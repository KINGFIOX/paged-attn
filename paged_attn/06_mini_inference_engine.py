"""Step 06 — A toy inference engine on top of paged attention.

So far we have:

    * a block-paged KV pool with a refcounting allocator      (step 03)
    * a numerically-correct paged attention forward            (step 04)
    * a fast Triton decode kernel                              (step 05)

What we still need is the *scheduler* that turns these primitives into a
serving engine.  This file builds the smallest possible one:

    * Continuous batching: every step we re-build the in-flight batch from
      whatever requests are alive, instead of padding to a fixed shape.
    * Prefix sharing: requests carrying the same prompt id are forked from
      a shared "prompt sequence", so their KV blocks are deduplicated until
      they generate divergent tokens.
    * Admission control: if free blocks are insufficient, new requests wait.

We then compare it against a "contiguous" baseline that pre-reserves
max_seq_len blocks per request — the same fragmentation pain we measured in
step 02, now expressed as "how many requests could we even admit?".

This is a *toy* model: there is no real transformer here, only one
attention-style layer.  But the scheduling story is exactly the same as in
vLLM / TensorRT-LLM / SGLang.

Run:
    uv run python paged_attn/06_mini_inference_engine.py
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import pathlib as _pl
import sys as _sys
import time
from dataclasses import dataclass, field
from typing import Optional

for _stub in ("/usr/local/cuda/lib64/stubs", "/usr/local/cuda-12.6/lib64/stubs"):
    if _pl.Path(_stub, "libcuda.so").exists():
        _os.environ["LIBRARY_PATH"] = f"{_stub}:{_os.environ.get('LIBRARY_PATH', '')}"
        break

import torch
from rich.console import Console
from rich.table import Table


def _load(name: str, path: _pl.Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_HERE = _pl.Path(__file__).resolve().parent
_bm = _load("paged_attn._03_block_manager", _HERE / "03_block_manager.py")
_naive = _load("paged_attn._04_paged_attention_naive", _HERE / "04_paged_attention_naive.py")
_triton = _load("paged_attn._05_paged_attention_triton", _HERE / "05_paged_attention_triton.py")
KVPool, BlockManager, Sequence, OutOfBlocks = (
    _bm.KVPool, _bm.BlockManager, _bm.Sequence, _bm.OutOfBlocks,
)
store_kv = _naive.store_kv
paged_attention_decode = _triton.paged_attention_decode
pack_batch = _triton.pack_batch


# ---------------------------------------------------------------------------
# Request bookkeeping.
# ---------------------------------------------------------------------------


@dataclass
class Request:
    req_id: int
    prompt_id: int            # requests sharing this id can share prefix blocks
    prompt_len: int
    max_new_tokens: int
    # Filled in by the engine:
    seq: Optional[Sequence] = None
    generated: int = 0
    arrival_step: int = 0
    finish_step: Optional[int] = None

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------


class MiniEngine:
    """One-attention-layer toy engine.

    The "model" is just `Q/K/V <- deterministic function of (prompt_id, pos)`
    so we can verify everything end-to-end without a tokenizer or real weights.
    """

    def __init__(self, *, num_blocks: int, block_size: int, n_heads: int, head_dim: int,
                 device: torch.device, dtype=torch.float16,
                 max_blocks_per_seq: int = 256):
        self.pool = KVPool(num_blocks=num_blocks, n_heads=n_heads,
                           block_size=block_size, head_dim=head_dim,
                           dtype=dtype, device=device)
        self.mgr = BlockManager(self.pool)
        self.device = device
        self.dtype = dtype
        self.n_heads, self.head_dim = n_heads, head_dim
        self.max_blocks_per_seq = max_blocks_per_seq

        # Cached prompt "sequences": one per unique prompt_id, holds the prompt
        # KV in shared blocks; children fork from it.
        self._prompt_seqs: dict[int, Sequence] = {}
        # How many distinct children are still using each prompt cache.
        self._prompt_refs: dict[int, int] = {}

        # In-flight requests (running this step).
        self.running: list[Request] = []
        # Pending requests (waiting for memory).
        self.pending: list[Request] = []
        self.finished: list[Request] = []

        # Worst-case future block reservation across all admitted requests.
        # The allocator can't OOM if we keep `_reserved_blocks <= num_blocks`.
        self._reserved_blocks = 0
        self._req_reservation: dict[int, int] = {}

        # Stats.
        self.step_count = 0
        self.tokens_produced = 0
        self.peak_blocks_used = 0

    # ---- "model" stubs: deterministic random K/V/Q per (prompt_id, pos). ----

    def _rng(self, prompt_id: int, pos: int) -> torch.Generator:
        g = torch.Generator(device="cpu")  # CPU for portability; we copy below.
        g.manual_seed(0xC0FFEE + 1000 * prompt_id + pos)
        return g

    def _make_kv(self, prompt_id: int, start_pos: int, length: int):
        g = self._rng(prompt_id, start_pos)
        shape = (self.n_heads, length, self.head_dim)
        k = torch.randn(shape, generator=g, dtype=torch.float32)
        v = torch.randn(shape, generator=g, dtype=torch.float32)
        return k.to(self.device, self.dtype), v.to(self.device, self.dtype)

    def _make_q(self, req: Request) -> torch.Tensor:
        # The decode query at position (prompt_len + generated) — uses req_id
        # rather than prompt_id so each child has independent queries.
        g = torch.Generator(device="cpu")
        g.manual_seed(0xBADC0DE + 1000 * req.req_id + req.total_len)
        q = torch.randn(self.n_heads, 1, self.head_dim, generator=g, dtype=torch.float32)
        return q.to(self.device, self.dtype)

    # ---- request lifecycle --------------------------------------------------

    def submit(self, req: Request) -> None:
        req.arrival_step = self.step_count
        self.pending.append(req)

    def _worst_case_blocks(self, req: Request, prompt_already_cached: bool) -> int:
        """Worst-case number of *new* pool blocks this request will consume.

        If the prompt is already in the pool, the child only needs its own
        decode blocks (plus one extra to account for a future copy-on-write
        of the partially-full last prompt block).
        """
        bs = self.pool.block_size
        if prompt_already_cached:
            return (req.max_new_tokens + bs - 1) // bs + 1
        return (req.prompt_len + req.max_new_tokens + bs - 1) // bs

    def _admit(self, req: Request) -> bool:
        """Try to admit `req` into the running set.  Returns False if OOM."""
        prompt_cached = req.prompt_id in self._prompt_seqs
        reservation = self._worst_case_blocks(req, prompt_cached)
        if self._reserved_blocks + reservation > self.pool.num_blocks:
            return False

        prompt_seq = self._prompt_seqs.get(req.prompt_id)
        if prompt_seq is None:
            # Fresh prompt: allocate blocks and fill them.
            ps = self.mgr.new_sequence()
            self.mgr.append_tokens(ps, req.prompt_len)
            k, v = self._make_kv(req.prompt_id, 0, req.prompt_len)
            store_kv(self.pool, ps, k, v, token_offset=0)
            self._prompt_seqs[req.prompt_id] = ps
            self._prompt_refs[req.prompt_id] = 0
            prompt_seq = ps

        # Fork (COW): child shares prompt blocks with refcount > 1.
        req.seq = self.mgr.fork(prompt_seq)
        self._prompt_refs[req.prompt_id] += 1
        self.running.append(req)

        self._reserved_blocks += reservation
        self._req_reservation[req.req_id] = reservation
        return True

    def _retire(self, req: Request) -> None:
        assert req.seq is not None
        self.mgr.free_sequence(req.seq)
        req.finish_step = self.step_count
        self.finished.append(req)

        # Release the worst-case reservation.
        self._reserved_blocks -= self._req_reservation.pop(req.req_id)

        # If we were the last child of this prompt cache, free the parent.
        self._prompt_refs[req.prompt_id] -= 1
        if self._prompt_refs[req.prompt_id] == 0 and not any(
            r.prompt_id == req.prompt_id for r in self.pending
        ):
            ps = self._prompt_seqs.pop(req.prompt_id)
            del self._prompt_refs[req.prompt_id]
            self.mgr.free_sequence(ps)

    # ---- one engine step ----------------------------------------------------

    def step(self) -> None:
        # 1. Admit as many pending requests as memory allows.
        still_pending: list[Request] = []
        for r in self.pending:
            if not self._admit(r):
                still_pending.append(r)
        self.pending = still_pending

        if not self.running:
            self.step_count += 1
            return

        # 2. Build the [B, H, D] query batch for this step.
        qs = [self._make_q(r).squeeze(1) for r in self.running]   # each [H, D]
        q_batch = torch.stack(qs, dim=0)                          # [B, H, D]
        block_tables, ctx_lens = pack_batch(
            [r.seq for r in self.running], device=self.device, dtype=torch.int32,
        )
        # Pad block_tables to a constant max for the kernel (one launch per step).
        if block_tables.shape[1] < self.max_blocks_per_seq:
            pad = torch.zeros(
                block_tables.shape[0],
                self.max_blocks_per_seq - block_tables.shape[1],
                device=self.device, dtype=block_tables.dtype,
            )
            block_tables = torch.cat([block_tables, pad], dim=1)

        # 3. Paged attention.  (We don't actually use the output to sample
        #    tokens in this toy — but in a real engine `out` would feed an LM
        #    head + sampling layer to produce the next token id.)
        _ = paged_attention_decode(q_batch, self.pool.buffer, block_tables, ctx_lens)

        # 4. For each running request: append one fresh K/V into the cache,
        #    increment generated count, retire if done.
        retire = []
        for r in self.running:
            self.mgr.append_tokens(r.seq, 1)
            k, v = self._make_kv(r.prompt_id, r.total_len, 1)
            store_kv(self.pool, r.seq, k, v, token_offset=r.total_len)
            r.generated += 1
            self.tokens_produced += 1
            if r.generated >= r.max_new_tokens:
                retire.append(r)
        for r in retire:
            self.running.remove(r)
            self._retire(r)

        used = self.pool.num_blocks - self.mgr.alloc.num_free
        self.peak_blocks_used = max(self.peak_blocks_used, used)
        self.step_count += 1

    def run_until_done(self) -> None:
        while self.pending or self.running:
            self.step()


# ---------------------------------------------------------------------------
# Workload + driver: show the win from prefix sharing and tight memory use.
# ---------------------------------------------------------------------------


def run_demo() -> None:
    assert torch.cuda.is_available(), "this demo wants a GPU"
    console = Console()
    device = torch.device("cuda")

    # Tiny but plausible config — A800 has plenty of room, we cap the pool on
    # purpose so admission control gets to do real work.
    block_size, n_heads, head_dim = 16, 8, 64
    num_blocks = 256       # capacity == 256 * 16 = 4096 tokens total
    max_seq_len = 256      # "model" context window

    engine = MiniEngine(num_blocks=num_blocks, block_size=block_size,
                        n_heads=n_heads, head_dim=head_dim, device=device,
                        max_blocks_per_seq=max_seq_len // block_size)

    # Construct a workload of 60 requests across 6 distinct prompts (so prompt
    # sharing has something to deduplicate).  Each request decodes 64 tokens.
    n_prompts = 6
    n_reqs = 60
    prompt_lens = {p: 64 + p * 16 for p in range(n_prompts)}
    requests = []
    for i in range(n_reqs):
        pid = i % n_prompts
        requests.append(Request(
            req_id=i, prompt_id=pid,
            prompt_len=prompt_lens[pid], max_new_tokens=64,
        ))

    # ---- Run the engine ----
    for r in requests:
        engine.submit(r)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    # ---- Stats ----
    bytes_per_token = 2 * n_heads * head_dim * 2  # K+V, fp16
    actual_kv_bytes_high_water = engine.peak_blocks_used * block_size * bytes_per_token

    # What the contiguous baseline (step 02 style) would have charged us:
    # every concurrent request reserves max_seq_len tokens, regardless of use.
    # We can't actually run the contiguous baseline in the same memory budget,
    # but we can compute the would-be size and how many requests would fit.
    contig_bytes_per_req = max_seq_len * bytes_per_token
    contig_capacity = (num_blocks * block_size * bytes_per_token) // contig_bytes_per_req

    table = Table(title=f"MiniEngine: {n_reqs} requests across {n_prompts} unique prompts")
    table.add_column("metric"); table.add_column("value", justify="right")
    table.add_row("steps run",              f"{engine.step_count}")
    table.add_row("tokens produced",        f"{engine.tokens_produced}")
    table.add_row("wall time",              f"{wall_s*1e3:.1f} ms")
    table.add_row("throughput",             f"{engine.tokens_produced/wall_s:,.0f} tok/s")
    table.add_row("peak blocks used",       f"{engine.peak_blocks_used}/{num_blocks}")
    table.add_row("peak KV memory",         f"{actual_kv_bytes_high_water/1024:.1f} KiB")
    table.add_row(
        "max concurrent (contig. baseline)",
        f"{contig_capacity} req (same pool, no paging)",
    )
    console.print(table)

    # Sanity: every request finished and produced exactly the right token count.
    assert len(engine.finished) == n_reqs
    for r in engine.finished:
        assert r.generated == r.max_new_tokens
    console.print(
        "[bold green]All requests finished cleanly; "
        "block pool drained back to empty.[/bold green]"
    )
    assert engine.mgr.alloc.num_free == num_blocks, \
        f"leaked blocks: {num_blocks - engine.mgr.alloc.num_free}"

    # Demonstrate prefix sharing by showing the shared prompt cache at work
    # mid-flight: rerun with 3 sibling requests under 1 prompt and inspect.
    console.print("\n[bold]Prefix-sharing inspection:[/bold]")
    engine2 = MiniEngine(num_blocks=64, block_size=16, n_heads=4, head_dim=32,
                         device=device, max_blocks_per_seq=32)
    siblings = [Request(req_id=i, prompt_id=42, prompt_len=80, max_new_tokens=8)
                for i in range(3)]
    for r in siblings:
        engine2.submit(r)
    # Just do the admission step and inspect refcounts before any decode.
    engine2.step()  # admits all three (they share blocks)
    parent = engine2._prompt_seqs[42]
    rc = [engine2.mgr.alloc.refcount(b) for b in parent.block_table]
    console.print(
        f"  prompt seq blocks = {parent.block_table}, refcounts = {rc} "
        f"(1 parent + {len(siblings)} children = {len(siblings)+1} per block)"
    )
    engine2.run_until_done()
    console.print(
        "[dim]After the children diverge, the prompt blocks stay shared "
        "until either (a) a child writes into them (COW) or (b) all "
        "children retire and the prompt cache is freed.[/dim]"
    )


if __name__ == "__main__":
    run_demo()
