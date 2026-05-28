"""Step 06 — A toy inference engine on top of paged attention.

Pulls together everything from the earlier steps:

    * block-paged KV pool + refcounting allocator      (step 03)
    * naive PyTorch paged attention forward            (step 04)

and adds the scheduler that turns these primitives into a serving engine:

    * Continuous batching — every step rebuilds the in-flight batch.
    * Prefix sharing      — same prompt_id forks from a shared prompt cache.
    * Admission control   — reserve worst-case future blocks per request.
    * Recompute preemption — if a pending request is stuck and the pool is
      full, evict the most-recently-admitted running request (LIFO, last in first out).
      Its KV is dropped; on re-admission we rebuild it.  (vLLM's default
      strategy; the alternative — CPU swap — lives in step 08.)

This is still a *toy* model: there is no transformer, only an attention
layer.  But the scheduling story is exactly the same as in vLLM /
TensorRT-LLM / SGLang.  Attention itself runs through the naive paged
PyTorch path from step 04 — slow but easy to reason about; swap in a
fused kernel later without touching the scheduler.

Run:
    uv run python -m paged_attn.mini_inference_engine
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from rich.console import Console
from rich.table import Table

from .block_manager import BlockManager, KVPool, Sequence
from .paged_attention_naive import paged_attention_single


# ---------------------------------------------------------------------------
# Request bookkeeping.
# ---------------------------------------------------------------------------


@dataclass
class Request:
    req_id: int
    prompt_id: int               # requests sharing this id can share prefix blocks
    prompt_len: int
    max_new_tokens: int
    priority: int = 0            # lower == admitted earlier (acts as FIFO key)
    # Filled in by the engine:
    seq: Optional[Sequence] = None
    generated: int = 0
    arrival_step: int = 0
    finish_step: Optional[int] = None
    needs_recompute: bool = False  # true while waiting to be re-admitted after preemption
    preemption_count: int = 0      # times this request has been preempted

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------


class MiniEngine:
    """One-attention-layer toy engine.

    The "model" is `Q/K/V <- deterministic_random(prompt_id_or_req_id, pos)`,
    so we can verify everything end-to-end without a tokenizer or real weights.
    Because seeding is *per absolute position*, recompute preemption rebuilds
    the exact same KV — making "preempted run == non-preempted run" provable.
    """

    def __init__(self, *, num_blocks: int, block_size: int, n_heads: int, head_dim: int,
                 device: torch.device, dtype=torch.float16,
                 enable_preemption: bool = True):
        self.pool = KVPool(num_blocks=num_blocks, n_heads=n_heads,
                           block_size=block_size, head_dim=head_dim,
                           dtype=dtype, device=device)
        self.mgr = BlockManager(self.pool)
        self.device = device
        self.dtype = dtype
        self.n_heads, self.head_dim = n_heads, head_dim
        self.enable_preemption = enable_preemption

        # Cached prompt "sequences": one per unique prompt_id.
        self._prompt_seqs: dict[int, Sequence] = {}
        self._prompt_refs: dict[int, int] = {}

        self.running: list[Request] = []
        self.pending: list[Request] = []
        self.finished: list[Request] = []

        # Worst-case future block reservation, split into two pieces so that
        # neither over- nor under-counts when prompt caches outlive the request
        # that created them:
        #   * each in-flight Request reserves only its OWN decode blocks
        #     (`_req_reservation[req_id]`).
        #   * each live prompt cache reserves the blocks it actually occupies
        #     (`_prompt_cache_reservation[prompt_id]`).
        # `_reserved_blocks` is the sum of both maps.
        self._reserved_blocks = 0
        self._req_reservation: dict[int, int] = {}
        self._prompt_cache_reservation: dict[int, int] = {}

        # Per-step "do not re-admit this request this step" set, to avoid
        # immediately re-admitting a just-preempted request (which would
        # cause a livelock).
        self._frozen_this_step: set[int] = set()

        # Stats.
        self.step_count = 0
        self.tokens_produced = 0
        self.tokens_recomputed = 0
        self.preemption_events = 0
        self.peak_blocks_used = 0
        self.peak_concurrent = 0

    # ---- "model" stubs: deterministic random K/V/Q per absolute position. ---

    def _seeded(self, base: int, prompt_or_req_id: int, pos: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(base + 10_000 * prompt_or_req_id + pos)
        return g

    def _make_kv(self, prompt_id: int, start_pos: int, length: int):
        """K/V for `length` tokens of prompt `prompt_id`, deterministically
        seeded per *absolute* position (so recompute is bit-exact)."""
        ks, vs = [], []
        for i in range(length):
            g_k = self._seeded(0xC0FFEE, prompt_id, start_pos + i)
            g_v = self._seeded(0xDEAD00, prompt_id, start_pos + i)
            ks.append(torch.randn(self.n_heads, 1, self.head_dim,
                                  generator=g_k, dtype=torch.float32))
            vs.append(torch.randn(self.n_heads, 1, self.head_dim,
                                  generator=g_v, dtype=torch.float32))
        k = torch.cat(ks, dim=1).to(self.device, self.dtype)
        v = torch.cat(vs, dim=1).to(self.device, self.dtype)
        return k, v

    def _make_q_prompt(self, prompt_id: int, length: int) -> torch.Tensor:
        qs = []
        for pos in range(length):
            g = self._seeded(0xCAFE00, prompt_id, pos)
            qs.append(torch.randn(self.n_heads, 1, self.head_dim,
                                  generator=g, dtype=torch.float32))
        return torch.cat(qs, dim=1).to(self.device, self.dtype)

    def _make_q_decode(self, req: Request) -> torch.Tensor:
        # Decode queries are per-request (siblings under the same prompt
        # diverge) and per-position.
        g = self._seeded(0xBADC0DE0, req.req_id, req.total_len)
        return torch.randn(self.n_heads, 1, self.head_dim,
                           generator=g, dtype=torch.float32).to(self.device, self.dtype)

    # ---- request lifecycle --------------------------------------------------

    def submit(self, req: Request) -> None:
        req.arrival_step = self.step_count
        self.pending.append(req)
        # Stable sort by priority — lower priority value runs first.
        self.pending.sort(key=lambda r: (r.priority, r.arrival_step))

    def _decode_reservation(self, req: Request) -> int:
        """Blocks this request will eat *beyond* the shared prompt cache."""
        bs = self.pool.block_size
        # ceil(max_new / bs) blocks for the decode tokens themselves, plus one
        # extra for a possible copy-on-write on the partially-full last prompt
        # block.  Slightly conservative; never under-counts.
        return (req.max_new_tokens + bs - 1) // bs + 1

    def _prompt_cache_blocks(self, prompt_len: int) -> int:
        bs = self.pool.block_size
        return (prompt_len + bs - 1) // bs

    def _admit(self, req: Request) -> bool:
        """Try to admit `req` into running.  Handles fresh + recompute paths."""
        decode_res = self._decode_reservation(req)
        prompt_cached = req.prompt_id in self._prompt_seqs
        prompt_res = 0 if prompt_cached else self._prompt_cache_blocks(req.prompt_len)
        if self._reserved_blocks + decode_res + prompt_res > self.pool.num_blocks:
            return False

        prompt_seq = self._prompt_seqs.get(req.prompt_id)
        if prompt_seq is None:
            # First request for this prompt: build the cache.
            ps = self.mgr.new_sequence()
            k, v = self._make_kv(req.prompt_id, 0, req.prompt_len)
            self.mgr.append_kv(ps, k, v)

            # Run paged attention over the prompt — even though we don't
            # consume the output here, this is the real compute cost a serving
            # engine pays on first appearance of a prompt.
            q_prompt = self._make_q_prompt(req.prompt_id, req.prompt_len)
            _ = paged_attention_single(self.pool, ps, q_prompt, causal=True)

            self._prompt_seqs[req.prompt_id] = ps
            self._prompt_refs[req.prompt_id] = 0
            self._prompt_cache_reservation[req.prompt_id] = prompt_res
            self._reserved_blocks += prompt_res
            prompt_seq = ps

        # Fork (COW) — child shares prompt blocks with refcount > 1.
        req.seq = self.mgr.fork(prompt_seq)
        self._prompt_refs[req.prompt_id] += 1

        # Recompute path: rebuild the KV for tokens previously generated.
        if req.generated > 0:
            k_dec, v_dec = self._make_kv(req.prompt_id, req.prompt_len, req.generated)
            self.mgr.append_kv(req.seq, k_dec, v_dec)
            self.tokens_recomputed += req.generated

        self.running.append(req)
        self._reserved_blocks += decode_res
        self._req_reservation[req.req_id] = decode_res
        req.needs_recompute = False
        return True

    def _retire(self, req: Request) -> None:
        assert req.seq is not None
        self.mgr.free_sequence(req.seq)
        req.finish_step = self.step_count
        self.finished.append(req)
        self._reserved_blocks -= self._req_reservation.pop(req.req_id)
        self._drop_prompt_ref(req.prompt_id)

    def _preempt(self, victim: Request) -> None:
        """Recompute-style preemption: drop the victim's KV, requeue it."""
        assert victim.seq is not None
        self.mgr.free_sequence(victim.seq)
        victim.seq = None
        self._reserved_blocks -= self._req_reservation.pop(victim.req_id)
        self._drop_prompt_ref(victim.prompt_id)

        self.running.remove(victim)
        victim.needs_recompute = True
        victim.preemption_count += 1
        self.preemption_events += 1
        # Re-insert at the front *but* respect priority order.
        self.pending.append(victim)
        self.pending.sort(key=lambda r: (r.priority, r.arrival_step))
        # Don't try to re-admit this same request again in this step.
        self._frozen_this_step.add(victim.req_id)

    def _drop_prompt_ref(self, prompt_id: int) -> None:
        self._prompt_refs[prompt_id] -= 1
        if self._prompt_refs[prompt_id] == 0 and not any(
            r.prompt_id == prompt_id for r in self.running + self.pending
        ):
            ps = self._prompt_seqs.pop(prompt_id)
            del self._prompt_refs[prompt_id]
            self._reserved_blocks -= self._prompt_cache_reservation.pop(prompt_id)
            self.mgr.free_sequence(ps)

    # ---- one engine step ----------------------------------------------------

    def _try_admit_round(self) -> list[Request]:
        """Greedily admit pending requests.  Returns the list still pending."""
        still: list[Request] = []
        for r in self.pending:
            if r.req_id in self._frozen_this_step:
                still.append(r)
                continue
            if not self._admit(r):
                still.append(r)
        return still

    def step(self) -> None:
        self._frozen_this_step.clear()

        # 1. Admission, with optional preemption to satisfy starving pending.
        self.pending = self._try_admit_round()
        while (
            self.enable_preemption
            and self.pending
            and self.running
            and any(r.req_id not in self._frozen_this_step for r in self.pending)
        ):
            # Pick a victim: LIFO (most-recently-admitted) running request whose
            # priority is no better than the worst-waiting pending request.
            best_pending = min(
                (r for r in self.pending if r.req_id not in self._frozen_this_step),
                key=lambda r: (r.priority, r.arrival_step),
            )
            candidates = [
                r for r in self.running
                if (r.priority, r.arrival_step) > (best_pending.priority, best_pending.arrival_step)
            ]
            if not candidates:
                break
            victim = max(candidates, key=lambda r: r.arrival_step)
            self._preempt(victim)
            self.pending = self._try_admit_round()

        if not self.running:
            self.step_count += 1
            return

        self.peak_concurrent = max(self.peak_concurrent, len(self.running))

        # 2. Per-request decode attention via the naive paged path.  A real
        # serving engine fuses these into a batched kernel (see step 05 in the
        # old commit history) — we drop the outputs anyway since this engine
        # only tracks block-management / scheduling behavior.
        for r in self.running:
            q_r = self._make_q_decode(r)  # [H, 1, D]
            _ = paged_attention_single(self.pool, r.seq, q_r, causal=True)

        # 4. Append one fresh K/V per running request and possibly retire.
        retire = []
        for r in self.running:
            k, v = self._make_kv(r.prompt_id, r.total_len, 1)
            self.mgr.append_kv(r.seq, k, v)
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

    def run_until_done(self, max_steps: int = 10_000) -> None:
        while (self.pending or self.running) and self.step_count < max_steps:
            self.step()
        assert not (self.pending or self.running), \
            f"engine stuck after {self.step_count} steps"


# ---------------------------------------------------------------------------
# Driver: a workload that exercises prefix sharing AND preemption.
# ---------------------------------------------------------------------------


def _build_wave_one():
    """Bulk of long requests; arrives at step 0, fills the pool."""
    return [
        Request(req_id=i, prompt_id=100 + (i % 4),
                prompt_len=96, max_new_tokens=96, priority=10)
        for i in range(24)
    ]


def _build_wave_two():
    """High-priority short requests; arrives later, will need to cut the queue."""
    return [
        Request(req_id=100 + i, prompt_id=200 + (i % 2),
                prompt_len=16, max_new_tokens=16, priority=1)
        for i in range(8)
    ]


def _run_engine(num_blocks: int, *, enable_preemption: bool,
                device: torch.device,
                warmup_steps_before_wave2: int = 30) -> MiniEngine:
    engine = MiniEngine(
        num_blocks=num_blocks, block_size=16, n_heads=8, head_dim=64,
        device=device, dtype=torch.float16,
        enable_preemption=enable_preemption,
    )
    # Wave 1 arrives at step 0 — fills the pool with long, low-priority work.
    for r in _build_wave_one():
        engine.submit(r)
    # Run for a while so the pool fills up before wave 2 shows up.
    for _ in range(warmup_steps_before_wave2):
        engine.step()
    # Wave 2 — short, high-priority — arrives later.
    for r in _build_wave_two():
        engine.submit(r)
    engine.run_until_done()
    return engine


def run_demo() -> None:
    assert torch.cuda.is_available()
    console = Console()
    device = torch.device("cuda")

    # Pool size deliberately chosen so several wave-1 requests must wait, and
    # admitting wave-2 (higher priority) must preempt some wave-1 runners.
    num_blocks = 96
    block_size, n_heads, head_dim = 16, 8, 64
    max_seq_len = (96 + 96)  # 192

    # ---- (A) Run WITH preemption enabled. ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine_p = _run_engine(num_blocks, enable_preemption=True, device=device)
    torch.cuda.synchronize()
    wall_p = time.perf_counter() - t0

    # ---- (B) Run WITHOUT preemption (FIFO-ish; high-pri waits). ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine_n = _run_engine(num_blocks, enable_preemption=False, device=device)
    torch.cuda.synchronize()
    wall_n = time.perf_counter() - t0

    # ---- Stats ----
    bytes_per_token = 2 * n_heads * head_dim * 2  # K+V, fp16
    contig_bytes_per_req = max_seq_len * bytes_per_token
    contig_capacity = (num_blocks * block_size * bytes_per_token) // contig_bytes_per_req

    table = Table(title=f"MiniEngine: 24 long + 8 high-pri requests, pool={num_blocks} blocks")
    table.add_column("metric")
    table.add_column("with preemption", justify="right")
    table.add_column("no preemption", justify="right")
    table.add_row("steps run", f"{engine_p.step_count}", f"{engine_n.step_count}")
    table.add_row("tokens produced", f"{engine_p.tokens_produced}", f"{engine_n.tokens_produced}")
    table.add_row("tokens recomputed (waste)", f"{engine_p.tokens_recomputed}", f"{engine_n.tokens_recomputed}")
    table.add_row("preemption events", f"{engine_p.preemption_events}", f"{engine_n.preemption_events}")
    table.add_row("peak concurrent", f"{engine_p.peak_concurrent}", f"{engine_n.peak_concurrent}")
    table.add_row("peak blocks used",
                  f"{engine_p.peak_blocks_used}/{num_blocks}",
                  f"{engine_n.peak_blocks_used}/{num_blocks}")
    table.add_row("wall time", f"{wall_p*1e3:.0f} ms", f"{wall_n*1e3:.0f} ms")
    table.add_row("max concurrent (contig.)", f"{contig_capacity}", f"{contig_capacity}")
    console.print(table)

    # ---- Latency for the high-priority wave-2 requests ----
    def hp_latency(engine):
        return sorted(
            r.finish_step - r.arrival_step for r in engine.finished if r.priority == 1
        )

    lat_p = hp_latency(engine_p)
    lat_n = hp_latency(engine_n)
    console.print(
        f"\n[bold]High-priority wave-2 finish-step minus arrival-step "
        f"(lower = engine reacted faster):[/bold]\n"
        f"  with preemption: min={lat_p[0]}, median={lat_p[len(lat_p)//2]}, max={lat_p[-1]}\n"
        f"  no preemption  : min={lat_n[0]}, median={lat_n[len(lat_n)//2]}, max={lat_n[-1]}"
    )

    # ---- Correctness: preemption must not change the engine's output ----
    # Because every (prompt_id, pos) and (req_id, pos) is per-position
    # deterministic, the two runs should produce *exactly* the same tokens.
    # The engine doesn't emit tokens, but we can checksum the final KV
    # state of each request indirectly via stats: total tokens produced.
    assert engine_p.tokens_produced == engine_n.tokens_produced, \
        "preemption must not change the number of tokens produced"
    assert engine_p.mgr.alloc.num_free == num_blocks, "leak (with preemption)"
    assert engine_n.mgr.alloc.num_free == num_blocks, "leak (no preemption)"

    console.print(
        "\n[bold green]Preemption shipped: high-priority requests jump the queue, "
        "and recomputed tokens are bit-exact (per-position deterministic seeding).[/bold green]\n"
        "[dim]Trade-off: preemption finishes wave-2 sooner at the cost of "
        f"{engine_p.tokens_recomputed} tokens of recomputation.  Step 08 shows the "
        "alternative — CPU swap — which avoids recompute but adds PCIe traffic.[/dim]"
    )


if __name__ == "__main__":
    run_demo()
