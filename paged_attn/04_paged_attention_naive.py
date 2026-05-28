"""Step 04 — Paged attention forward (pure PyTorch reference).

We finally combine the block manager from step 03 with real attention math.
The goal here is *correctness*, not speed:

    * Store K/V into the paged pool using `(block_id, slot)` addressing.
    * Run attention by gathering K/V from blocks via the block table.
    * Verify the answer matches step 01's contiguous KV cache to ~fp32 noise.

To make the KV-cache story visible, this file reuses step 01's
`MiniSelfAttention` (real `W_q/W_k/W_v` linear layers).  Both the contiguous
reference and the paged path run the same prefill + per-step decode loop;
the *only* difference is where each step's freshly-projected K/V land:

    * Contiguous: into `ContiguousKVCache.k/v[:, :, offset, :]`.
    * Paged    : into `pool.buffer[phys_block, :, :, slot, :]` via the
                 block table.

Once we trust this reference, step 05 will port the gather + dot product
into a fused Triton kernel.

Run:
    uv run python paged_attn/04_paged_attention_naive.py
"""

from __future__ import annotations

import torch

# We import by full path because the sibling files start with a digit.
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys


def _load(name: str, path: _pl.Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    # Dataclasses look up `cls.__module__` in `sys.modules`, so we MUST register
    # the module *before* exec_module runs the class bodies.
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_HERE = _pl.Path(__file__).resolve().parent
_bm = _load("paged_attn._03_block_manager", _HERE / "03_block_manager.py")
KVPool, BlockManager, Sequence = _bm.KVPool, _bm.BlockManager, _bm.Sequence
_std = _load("paged_attn._01_standard_attention", _HERE / "01_standard_attention.py")
scaled_dot_product_attention = _std.scaled_dot_product_attention
MiniSelfAttention = _std.MiniSelfAttention
ContiguousKVCache = _std.ContiguousKVCache


# ---------------------------------------------------------------------------
# Writing K/V into the paged pool.
# ---------------------------------------------------------------------------
#
# The common case "extend the sequence by L tokens with these K/V" lives on
# `BlockManager.append_kv` (step 03) so the block manager stays a complete
# KV-cache layer on its own.  We keep `store_kv` here as a lower-level
# primitive for the rare case where you need to *overwrite* an arbitrary
# offset that has already been reserved.


def store_kv(
    pool: KVPool, seq: Sequence, k: torch.Tensor, v: torch.Tensor, token_offset: int
) -> None:
    """Write `k`,`v` of shape `[n_heads, L, head_dim]` into the pool, starting
    at `token_offset` within `seq`'s virtual address space.

    `seq` must already have `token_offset + L` slots reserved via
    `BlockManager.reserve_slots` or `append_kv`.  Use this when you want to
    re-write existing slots (e.g. swap-in restore); for the normal append
    flow call `BlockManager.append_kv` directly.
    """
    n_heads, L, head_dim = k.shape
    bs = pool.block_size
    assert v.shape == k.shape
    assert token_offset + L <= seq.length, "write beyond reserved length"

    for i in range(L):
        global_pos = token_offset + i
        logical_block = global_pos // bs
        slot = global_pos % bs
        phys = seq.block_table[logical_block]
        # buffer shape: [num_blocks, 2, n_heads, block_size, head_dim]
        pool.buffer[phys, 0, :, slot, :] = k[:, i, :]
        pool.buffer[phys, 1, :, slot, :] = v[:, i, :]


# ---------------------------------------------------------------------------
# Gather a sequence's K/V back into a contiguous `[H, L, D]` tensor.
# ---------------------------------------------------------------------------


def gather_kv(pool: KVPool, seq: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the full K and V for one sequence.

    This is the *slow* path — useful for verification and as a stepping stone
    to a fused kernel.  The real PagedAttention kernel never materializes this;
    it does dot products directly against the block buffer.
    """
    L = seq.length
    H, D = pool.n_heads, pool.head_dim

    # gather along the block axis: shape [n_blocks_used, 2, H, block_size, D]
    table = torch.tensor(seq.block_table, device=pool.buffer.device, dtype=torch.long)
    blocks = pool.buffer.index_select(0, table)  # [Nb, 2, H, bs, D]
    # Re-arrange to one long [H, Nb*bs, D] tensor, then trim the tail.
    k = blocks[:, 0].permute(1, 0, 2, 3).reshape(H, -1, D)  # [H, Nb*bs, D]
    v = blocks[:, 1].permute(1, 0, 2, 3).reshape(H, -1, D)
    return k[:, :L], v[:, :L]


# ---------------------------------------------------------------------------
# Paged attention for a *single* sequence.
# ---------------------------------------------------------------------------


def paged_attention_single(
    pool: KVPool,
    seq: Sequence,
    q: torch.Tensor,  # [H, Lq, D]
    causal: bool = True,
) -> torch.Tensor:  # [H, Lq, D]
    k, v = gather_kv(pool, seq)  # [H, L, D]
    # Lift to [B=1, H, *, D] and reuse the standard kernel.
    out = scaled_dot_product_attention(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        causal=causal,
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# A "paged" drop-in for step 01's ContiguousKVCache.
# ---------------------------------------------------------------------------


class PagedKVCache:
    """The paged equivalent of step 01's `ContiguousKVCache`.

    In step 01 a `cache` packaged together "where to put K/V" + "how to append"
    + "how to read it back" for one sequence.  Here the K/V live in the shared
    `KVPool`, indexed by a per-sequence block table, so the natural cache
    object is the `(pool, manager, seq)` triple — wrapped up so callers don't
    have to plumb three handles through every call site.

    Mirrors `ContiguousKVCache.append` / `.view` semantics:

        * `append(k_new, v_new)` writes `L_new` newly-projected K/V rows into
          the pool and advances the sequence length.
        * `attend(q_new)` runs paged attention with `q_new` over the entire
          current cache (prefix + everything appended so far).

    Inputs use the `[B=1, H, L, head_dim]` shape so the call sites can stay
    bit-for-bit symmetric with the contiguous path.
    """

    def __init__(self, pool: KVPool, mgr: BlockManager, seq: Sequence):
        self.pool = pool
        self.mgr = mgr
        self.seq = seq

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        # `_project` gives `[B,H,L_new,hd]`; the paged layer is batch-free, so
        # drop the leading batch dim before scatter-writing into the pool.
        assert k_new.shape == v_new.shape
        assert k_new.dim() == 4 and k_new.shape[0] == 1, (
            "PagedKVCache is single-sequence; pass [B=1, H, L_new, head_dim]"
        )
        self.mgr.append_kv(self.seq, k_new.squeeze(0), v_new.squeeze(0))

    def attend(self, q_new: torch.Tensor, causal: bool = True) -> torch.Tensor:
        out = paged_attention_single(
            self.pool, self.seq, q_new.squeeze(0), causal=causal
        )
        return out.unsqueeze(0)  # back to [B=1, H, L_new, head_dim]


def forward_with_paged_cache(
    attn: "MiniSelfAttention",
    x_new: torch.Tensor,  # [B=1, L_new, d_model]
    paged_cache: PagedKVCache,
) -> torch.Tensor:  # [B=1, H, L_new, head_dim]
    """Paged twin of `MiniSelfAttention.forward_with_cache`.

    Project `x_new` to Q/K/V via the *same* `W_q/W_k/W_v` as the contiguous
    path, append the freshly-projected K/V into the paged cache, then attend
    over the full (cached + new) K/V history.

    Step 01:   `attn.forward_with_cache(x_new, contig_cache)`
    Step 04:   `forward_with_paged_cache(attn, x_new, paged_cache)`
    """
    q_new, k_new, v_new = attn._project(x_new)
    paged_cache.append(k_new, v_new)
    return paged_cache.attend(q_new)


# ---------------------------------------------------------------------------
# Demo: prefill + decode, paged side vs step 01's contiguous KV cache.
# ---------------------------------------------------------------------------
#
# Both paths share *the same* MiniSelfAttention (so the same W_q/W_k/W_v), and
# both iterate prefill -> decode_steps decode steps.  Every decode step only
# projects the single new token — that is the whole point of the KV cache.
# The only difference between the two paths is where that single new K/V row
# lands: in `ContiguousKVCache.k/v[:, :, offset]` for the reference, vs in
# `pool.buffer[phys, :, :, slot]` (via the block table) for the paged path.
# When everything works the two paths must agree to fp32 noise.


def run_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B = 1  # paged path is single-sequence in this file
    n_heads, d_model = 4, 64
    head_dim = d_model // n_heads
    block_size = 4
    prompt_len = 10  # deliberately not a multiple of block_size
    decode_steps = 6
    total_len = prompt_len + decode_steps

    # Same attention layer as step 01.  W_q/W_k/W_v live here.
    attn = MiniSelfAttention(d_model, n_heads).to(device=device, dtype=dtype)
    # Fake "input tokens": in a real transformer these come from the embedding
    # layer (or the previous block's output).  We materialise the whole stream
    # up front so that both paths consume the exact same inputs at each step.
    tokens = torch.randn(B, total_len, d_model, device=device, dtype=dtype)

    # ----------------------------------------------------------------------
    # Path A — contiguous KV cache.  Same loop as step 01 path B:
    #          one prefill call, then decode one token at a time, projecting
    #          only the single new token's Q/K/V on each step.
    # ----------------------------------------------------------------------
    cache = ContiguousKVCache.empty(B, n_heads, total_len, head_dim, device, dtype)
    outs_a: list[torch.Tensor] = []

    # forward_with_cache with _project inside
    out_p = attn.forward_with_cache(tokens[:, :prompt_len], cache)  # [B,H,Lp,hd]
    outs_a.append(out_p)
    for t in range(decode_steps):
        x_new = tokens[:, prompt_len + t : prompt_len + t + 1]  # [B,1,d]
        out_t = attn.forward_with_cache(x_new, cache)  # [B,H,1,hd]
        outs_a.append(out_t)
    out_contig = torch.cat(outs_a, dim=2).squeeze(0)  # [H,total,hd]

    # ----------------------------------------------------------------------
    # Path B — paged KV cache.  Same loop as Path A but with `PagedKVCache`
    #          as a drop-in for `ContiguousKVCache`.  The single line inside
    #          the loop hides projection (W_q/W_k/W_v) + scatter-write into
    #          the pool + paged attention, exactly like Path A hides the
    #          contiguous append + attention.
    # ----------------------------------------------------------------------
    pool = KVPool(
        num_blocks=16,
        n_heads=n_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    mgr = BlockManager(pool)
    paged_cache = PagedKVCache(pool, mgr, mgr.new_sequence())
    outs_b: list[torch.Tensor] = []  # a list of [B=1, H, L_new, head_dim]

    out_p = forward_with_paged_cache(attn, tokens[:, :prompt_len], paged_cache)
    outs_b.append(out_p)
    for t in range(decode_steps):
        x_new = tokens[:, prompt_len + t : prompt_len + t + 1]
        out_p = forward_with_paged_cache(attn, x_new, paged_cache)
        outs_b.append(out_p)
    out_paged = torch.cat(outs_b, dim=2).squeeze(0)  # [H,total,hd]

    seq = paged_cache.seq  # for the layout printouts below

    err = (out_contig - out_paged).abs().max().item()
    print(
        f"[04] max |contiguous KV cache - paged KV cache| = {err:.2e}  "
        f"(prompt_len={prompt_len}, decode_steps={decode_steps}, "
        f"block_size={block_size})"
    )
    assert err < 1e-4, "paged attention disagrees with contiguous reference!"

    # ---- (c) Show the block layout: tokens scattered across non-contiguous blocks ----
    print(f"[04] sequence length = {seq.length}")
    print(f"[04] logical -> physical block table: {seq.block_table}")
    print(
        f"[04] pool utilization: "
        f"{mgr.utilization()['blocks_used']}/{pool.num_blocks} blocks "
        f"({mgr.utilization()['block_util']:.0%} of the *used* blocks are populated)"
    )

    # ----------------------------------------------------------------------
    # (d) Bonus: prefix sharing.  Project the prompt ONCE, fork three children,
    #     decode each child with its own random "next token" stream, and check
    #     that all three children read the same prompt K/V (proving the shared
    #     blocks are wired up correctly).
    # ----------------------------------------------------------------------
    print("\n[04] Multi-sequence prefix-sharing sanity check:")
    pool2 = KVPool(
        num_blocks=32,
        n_heads=n_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    mgr2 = BlockManager(pool2)

    # Prompt: project once via `forward_with_paged_cache` so the prefix output
    # (Lq = Lk = prompt_len => standard lower-triangular mask) is recorded as
    # the reference that every child must reproduce on its prefix positions.
    prompt_cache = PagedKVCache(pool2, mgr2, mgr2.new_sequence())
    ref_prefix = forward_with_paged_cache( attn, tokens[:, :prompt_len], prompt_cache).squeeze(0)  # [H, prompt_len, hd]

    children = [mgr2.fork(prompt_cache.seq) for _ in range(3)]
    for ci, child in enumerate(children):
        # Wrap the child's seq in its own paged cache (the pool/manager are
        # shared, the block table is per-sequence with the prompt blocks
        # shared via fork).
        child_cache = PagedKVCache(pool2, mgr2, child)

        # Each child consumes its own random continuation, one token at a time,
        # via the SAME 1-line API as Path B.
        x_cont = torch.randn(B, decode_steps, d_model, device=device, dtype=dtype)
        decode_outs: list[torch.Tensor] = []
        for t in range(decode_steps):
            out_p = forward_with_paged_cache(attn, x_cont[:, t : t + 1], child_cache)
            decode_outs.append(out_p)

        # To verify the prefix part, project the prompt's Q again and run
        # paged attention with `q_full = [q_prompt, q_decode]` against the
        # child's full K/V.  With Lq == Lk the causal mask is just lower-
        # triangular, so the first `prompt_len` output rows correspond to the
        # prompt positions and must equal `ref_prefix`.
        # Lp: length of prompt
        q_prompt = attn._project(tokens[:, :prompt_len])[0]  # [B,H,Lp,hd]
        q_decode = attn._project(x_cont)[0]  # [B,H,decode,hd]
        q_full = torch.cat([q_prompt, q_decode], dim=2)  # [B,H,total,hd]
        out_child_full = child_cache.attend(q_full).squeeze(0)  # [H,total,hd]
        prefix_err = (ref_prefix - out_child_full[:, :prompt_len]).abs().max().item()
        print(
            f"  child {ci}: blocks={child.block_table}, len={child.length}, "
            f"prefix max|err|={prefix_err:.2e}"
        )
        assert prefix_err < 1e-4, "prefix-sharing math is broken"

    print(
        "[04] OK — iterative paged decode matches the iterative contiguous "
        "KV cache (both with real W_q/W_k/W_v), and prefix sharing reads "
        "identical K/V from the shared blocks."
    )


if __name__ == "__main__":
    run_demo()
