"""Step 04 — Paged attention forward (pure PyTorch reference).

We finally combine the block manager from step 03 with real attention math.
The goal here is *correctness*, not speed:

    * Store K/V into the paged pool using `(block_id, slot)` addressing.
    * Run attention by gathering K/V from blocks via the block table.
    * Verify the answer matches step 01's contiguous KV cache to ~fp32 noise.

Once we trust this reference, step 05 will port the gather + dot product
into a fused Triton kernel.

Run:
    uv run python paged_attn/04_paged_attention_naive.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence as SeqT

import torch
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Writing K/V into the paged pool, one token at a time.
# ---------------------------------------------------------------------------


def store_kv(pool: KVPool, seq: Sequence, k: torch.Tensor, v: torch.Tensor,
             token_offset: int) -> None:
    """Write `k`,`v` of shape `[n_heads, L, head_dim]` into the pool, starting
    at `token_offset` within `seq`'s virtual address space.

    The sequence must already have enough logical blocks reserved (the
    BlockManager.append_tokens call handles that).
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
    bs = pool.block_size
    L = seq.length
    H, D = pool.n_heads, pool.head_dim

    # gather along the block axis: shape [n_blocks_used, 2, H, block_size, D]
    table = torch.tensor(seq.block_table, device=pool.buffer.device, dtype=torch.long)
    blocks = pool.buffer.index_select(0, table)               # [Nb, 2, H, bs, D]
    # Re-arrange to one long [H, Nb*bs, D] tensor, then trim the tail.
    k = blocks[:, 0].permute(1, 0, 2, 3).reshape(H, -1, D)    # [H, Nb*bs, D]
    v = blocks[:, 1].permute(1, 0, 2, 3).reshape(H, -1, D)
    return k[:, :L], v[:, :L]


# ---------------------------------------------------------------------------
# Paged attention for a *single* sequence.
# ---------------------------------------------------------------------------


def paged_attention_single(
    pool: KVPool, seq: Sequence,
    q: torch.Tensor,            # [H, Lq, D]
    causal: bool = True,
) -> torch.Tensor:               # [H, Lq, D]
    k, v = gather_kv(pool, seq)  # [H, L, D]
    # Lift to [B=1, H, *, D] and reuse the standard kernel.
    out = scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=causal,
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# Demo: prefill + decode, paged side vs contiguous side, exact match.
# ---------------------------------------------------------------------------


def run_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    H, D = 4, 16
    block_size = 4
    prompt_len = 10           # not a multiple of block_size on purpose
    decode_steps = 6
    total_len = prompt_len + decode_steps

    # Pool sized to comfortably hold the whole sequence.
    pool = KVPool(num_blocks=16, n_heads=H, block_size=block_size, head_dim=D,
                  dtype=dtype, device=device)
    mgr = BlockManager(pool)
    seq = mgr.new_sequence()

    # Fake Q/K/V — generated once so we can compare paged vs contiguous on the
    # same numbers.
    full_q = torch.randn(H, total_len, D, device=device, dtype=dtype)
    full_k = torch.randn(H, total_len, D, device=device, dtype=dtype)
    full_v = torch.randn(H, total_len, D, device=device, dtype=dtype)

    # ---- (a) Contiguous reference (single sequence) ----
    ref_out = scaled_dot_product_attention(
        full_q.unsqueeze(0), full_k.unsqueeze(0), full_v.unsqueeze(0), causal=True
    ).squeeze(0)  # [H, total_len, D]

    # ---- (b) Paged: prefill ----
    mgr.append_tokens(seq, prompt_len)
    store_kv(pool, seq, full_k[:, :prompt_len], full_v[:, :prompt_len], token_offset=0)
    out_prefill = paged_attention_single(pool, seq, full_q[:, :prompt_len], causal=True)

    chunks = [out_prefill]

    # ---- (c) Paged: decode one token at a time ----
    for step in range(decode_steps):
        pos = prompt_len + step
        mgr.append_tokens(seq, 1)
        store_kv(pool, seq,
                 full_k[:, pos:pos + 1], full_v[:, pos:pos + 1],
                 token_offset=pos)
        out_t = paged_attention_single(pool, seq, full_q[:, pos:pos + 1], causal=True)
        chunks.append(out_t)

    paged_out = torch.cat(chunks, dim=1)  # [H, total_len, D]

    err = (ref_out - paged_out).abs().max().item()
    print(f"[04] max |contiguous - paged| = {err:.2e}  (block_size={block_size})")
    assert err < 1e-4, "paged attention disagrees with contiguous reference!"

    # ---- (d) Show the block layout: tokens scattered across non-contiguous blocks ----
    print(f"[04] sequence length = {seq.length}")
    print(f"[04] logical -> physical block table: {seq.block_table}")
    print(f"[04] pool utilization: "
          f"{mgr.utilization()['blocks_used']}/{pool.num_blocks} blocks "
          f"({mgr.utilization()['block_util']:.0%} of the *used* blocks are populated)")

    # ---- (e) Bonus: multiple sequences with prefix sharing ----
    print("\n[04] Multi-sequence prefix-sharing sanity check:")
    pool2 = KVPool(num_blocks=32, n_heads=H, block_size=block_size, head_dim=D,
                   dtype=dtype, device=device)
    mgr2 = BlockManager(pool2)
    prompt = mgr2.new_sequence()
    mgr2.append_tokens(prompt, prompt_len)
    store_kv(pool2, prompt, full_k[:, :prompt_len], full_v[:, :prompt_len], 0)

    children = [mgr2.fork(prompt) for _ in range(3)]
    # Three independent continuations: pretend each samples a different next K/V.
    for ci, child in enumerate(children):
        kc = torch.randn(H, decode_steps, D, device=device, dtype=dtype)
        vc = torch.randn(H, decode_steps, D, device=device, dtype=dtype)
        qc = torch.randn(H, decode_steps, D, device=device, dtype=dtype)
        for step in range(decode_steps):
            mgr2.append_tokens(child, 1)
            store_kv(pool2, child, kc[:, step:step+1], vc[:, step:step+1],
                     token_offset=prompt_len + step)
        # Run paged attention over the full sequence of this child.
        full_q_child = torch.cat(
            [full_q[:, :prompt_len], qc], dim=1
        )
        out_child = paged_attention_single(pool2, child, full_q_child, causal=True)
        # The first `prompt_len` rows of out_child should match the
        # *single-prompt* reference for the prefix — proves the shared blocks
        # are read correctly.
        ref_prefix = scaled_dot_product_attention(
            full_q[:, :prompt_len].unsqueeze(0),
            full_k[:, :prompt_len].unsqueeze(0),
            full_v[:, :prompt_len].unsqueeze(0), causal=True,
        ).squeeze(0)
        prefix_err = (ref_prefix - out_child[:, :prompt_len]).abs().max().item()
        print(f"  child {ci}: blocks={child.block_table}, len={child.length}, "
              f"prefix max|err|={prefix_err:.2e}")
        assert prefix_err < 1e-4, "prefix-sharing math is broken"

    print("[04] OK — paged attention agrees with contiguous baseline, "
          "and prefix sharing reads correct values.")


if __name__ == "__main__":
    run_demo()
