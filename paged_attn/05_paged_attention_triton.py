"""Step 05 — Paged attention as a Triton kernel (decode phase).

This is the file where things finally go fast.  The naive PyTorch path in
step 04 materializes K and V for the whole sequence before each call.  A real
serving engine does thousands of *decode* steps per second over thousands of
sequences — materialization would burn HBM bandwidth and dwarf the actual
math.

So we write a Triton kernel that:

    * Launches one program per **(sequence, head)** pair.
    * Loads the query vector once into registers / shared memory.
    * Walks the sequence's **block table**, loading one KV block at a time
      directly from the paged pool, and accumulates with an online softmax
      (Flash-Attention style: track running max `m` and normaliser `l`).

This is exactly the structure of vLLM's `paged_attention_v1` kernel — just
written for clarity instead of speed.  We only handle Lq == 1 (decode); the
prefill phase uses Flash-Attention on contiguous prompt tokens before they
are written into the pool, so it doesn't need paging.

Run:
    uv run python paged_attn/05_paged_attention_triton.py
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import pathlib as _pl
import sys as _sys
import time

# Make `gcc -lcuda` succeed when Triton JIT-compiles its CUDA helper.
# (libcuda.so.1 is in /usr/lib but the unversioned symlink lives in the SDK stubs.)
for _stub in ("/usr/local/cuda/lib64/stubs", "/usr/local/cuda-12.6/lib64/stubs"):
    if _pl.Path(_stub, "libcuda.so").exists():
        _os.environ["LIBRARY_PATH"] = f"{_stub}:{_os.environ.get('LIBRARY_PATH', '')}"
        break

import torch
import triton
import triton.language as tl


def _load(name: str, path: _pl.Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_HERE = _pl.Path(__file__).resolve().parent
_bm = _load("paged_attn._03_block_manager", _HERE / "03_block_manager.py")
_naive = _load("paged_attn._04_paged_attention_naive", _HERE / "04_paged_attention_naive.py")
KVPool, BlockManager = _bm.KVPool, _bm.BlockManager
store_kv = _naive.store_kv
paged_attention_single = _naive.paged_attention_single


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _paged_attn_decode_kernel(
    # Pointers
    q_ptr,            # [B, H, D]
    pool_ptr,         # [num_blocks, 2, H, BS, D]
    block_tables_ptr, # [B, max_blocks]
    context_lens_ptr, # [B]
    out_ptr,          # [B, H, D]
    # Strides (in elements, not bytes)
    sq_b, sq_h,
    sp_b, sp_kv, sp_h, sp_t,
    sbt_b,
    so_b, so_h,
    # Sizes / scales
    scale,
    max_num_blocks_per_seq,
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,   # KV block size (== pool.block_size)
    HEAD_DIM:   tl.constexpr,   # per-head dimension (e.g. 64, 128)
):
    """One program == one (sequence, head)."""

    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    d_offs = tl.arange(0, HEAD_DIM)
    t_offs = tl.arange(0, BLOCK_SIZE)

    # ---- Load the (single) query vector for this (seq, head). ----
    q = tl.load(q_ptr + pid_b * sq_b + pid_h * sq_h + d_offs).to(tl.float32)

    # ---- Running softmax state. ----
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    context_len = tl.load(context_lens_ptr + pid_b)
    # Ceil division without integer-overflow shenanigans.
    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    for blk_idx in range(0, num_blocks):
        # 1. Translate logical -> physical block.
        phys = tl.load(block_tables_ptr + pid_b * sbt_b + blk_idx)

        # 2. Causal mask along the block's BLOCK_SIZE positions.
        token_idx = blk_idx * BLOCK_SIZE + t_offs            # [BLOCK_SIZE]
        valid = token_idx < context_len                      # [BLOCK_SIZE]

        # 3. Compute pointers into the pool. Pool layout:
        #    [num_blocks, 2, H, BLOCK_SIZE, HEAD_DIM]
        # K plane = kv-axis 0, V plane = kv-axis 1.
        k_block_ptr = (
            pool_ptr
            + phys * sp_b
            + 0    * sp_kv          # K
            + pid_h * sp_h
            + t_offs[:, None] * sp_t
            + d_offs[None, :]
        )
        v_block_ptr = (
            pool_ptr
            + phys * sp_b
            + 1    * sp_kv          # V
            + pid_h * sp_h
            + t_offs[:, None] * sp_t
            + d_offs[None, :]
        )

        # 4. Load K, V for the block.  Masked rows are zeroed; we mask scores
        #    to -inf below so they contribute nothing to softmax / acc.
        k = tl.load(k_block_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=valid[:, None], other=0.0).to(tl.float32)

        # 5. Score = (K @ q) / sqrt(D), then causal/length mask.
        scores = tl.sum(k * q[None, :], axis=1) * scale       # [BLOCK_SIZE]
        scores = tl.where(valid, scores, -float("inf"))

        # 6. Online softmax update — the Flash-Attention trick.
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)                            # [BLOCK_SIZE]
        l_new = alpha * l_i + tl.sum(p, axis=0)
        # acc <- acc * alpha + p^T V
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i, l_i = m_new, l_new

    # ---- Finalize and write back. ----
    out = (acc / l_i).to(tl.load(q_ptr + pid_b * sq_b + pid_h * sq_h + d_offs).dtype)
    tl.store(out_ptr + pid_b * so_b + pid_h * so_h + d_offs, out)


# ---------------------------------------------------------------------------
# Python-side wrapper.
# ---------------------------------------------------------------------------


def paged_attention_decode(
    q: torch.Tensor,                # [B, H, D] (one token per sequence)
    pool_buffer: torch.Tensor,      # [num_blocks, 2, H, BLOCK_SIZE, D]
    block_tables: torch.Tensor,     # [B, max_blocks], int32/int64
    context_lens: torch.Tensor,     # [B], int32
) -> torch.Tensor:
    assert q.dim() == 3, f"q must be [B, H, D], got {q.shape}"
    B, H, D = q.shape
    NB, KV, Hp, BS, Dp = pool_buffer.shape
    assert KV == 2 and Hp == H and Dp == D, \
        f"pool/query shape mismatch: pool={pool_buffer.shape}, q={q.shape}"
    assert block_tables.shape[0] == B
    assert context_lens.shape == (B,)

    out = torch.empty_like(q)
    grid = (B, H)

    # Element strides (Triton wants element counts, not bytes).
    sq_b, sq_h, _ = q.stride()
    sp_b, sp_kv, sp_h, sp_t, _ = pool_buffer.stride()
    sbt_b = block_tables.stride(0)
    so_b, so_h, _ = out.stride()

    _paged_attn_decode_kernel[grid](
        q, pool_buffer, block_tables, context_lens, out,
        sq_b, sq_h,
        sp_b, sp_kv, sp_h, sp_t,
        sbt_b,
        so_b, so_h,
        scale=1.0 / (D ** 0.5),
        max_num_blocks_per_seq=block_tables.shape[1],
        BLOCK_SIZE=BS,
        HEAD_DIM=D,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers to pack a list of `Sequence` objects into the kernel's tensors.
# ---------------------------------------------------------------------------


def pack_batch(sequences, device: torch.device, dtype=torch.int32):
    B = len(sequences)
    max_blocks = max(len(s.block_table) for s in sequences)
    block_tables = torch.zeros(B, max_blocks, device=device, dtype=dtype)
    lengths = torch.zeros(B, device=device, dtype=dtype)
    for i, s in enumerate(sequences):
        block_tables[i, :len(s.block_table)] = torch.tensor(
            s.block_table, device=device, dtype=dtype
        )
        lengths[i] = s.length
    return block_tables, lengths


# ---------------------------------------------------------------------------
# Correctness + (tiny) benchmark.
# ---------------------------------------------------------------------------


def run_demo() -> None:
    assert torch.cuda.is_available(), "Triton kernel needs CUDA"
    device = torch.device("cuda")
    dtype = torch.float16

    B = 8                # concurrent sequences
    H = 8                # attention heads (MHA — easier to reason about)
    D = 64               # head dim
    block_size = 16
    seq_lens = [37, 64, 100, 12, 256, 511, 1000, 80]
    max_len = max(seq_lens)

    num_blocks_needed = sum((L + block_size - 1) // block_size for L in seq_lens)
    pool = KVPool(num_blocks=num_blocks_needed + 8,
                  n_heads=H, block_size=block_size, head_dim=D,
                  dtype=dtype, device=device)
    mgr = BlockManager(pool)

    # Build sequences: reserve blocks + fill K/V with random data.
    torch.manual_seed(7)
    sequences = []
    full_ks, full_vs, full_qs = [], [], []  # one query per sequence (decode step)
    for L in seq_lens:
        seq = mgr.new_sequence()
        mgr.append_tokens(seq, L)
        k = torch.randn(H, L, D, device=device, dtype=dtype)
        v = torch.randn(H, L, D, device=device, dtype=dtype)
        store_kv(pool, seq, k, v, token_offset=0)
        sequences.append(seq)
        full_ks.append(k); full_vs.append(v)
        full_qs.append(torch.randn(H, 1, D, device=device, dtype=dtype))

    # Stack the per-sequence decode queries into [B, H, D].
    q_batch = torch.stack([q.squeeze(1) for q in full_qs], dim=0)  # [B, H, D]

    block_tables, context_lens = pack_batch(sequences, device, dtype=torch.int32)

    # ---- Triton ----
    out_triton = paged_attention_decode(q_batch, pool.buffer, block_tables, context_lens)

    # ---- PyTorch reference (per-sequence) ----
    out_ref = torch.zeros_like(q_batch)
    for i, seq in enumerate(sequences):
        out_i = paged_attention_single(pool, seq, full_qs[i], causal=True)  # [H, 1, D]
        out_ref[i] = out_i.squeeze(1)

    max_abs = (out_triton.float() - out_ref.float()).abs().max().item()
    rel = max_abs / (out_ref.float().abs().max().item() + 1e-6)
    print(f"[05] max |triton - reference| = {max_abs:.3e}  (rel {rel:.1e})  dtype={dtype}")
    # fp16 attention accumulates a fair bit of rounding; 5e-2 absolute is safe.
    assert max_abs < 5e-2, "Triton kernel disagrees with PyTorch reference!"

    # ---- micro-benchmark ----
    def bench(fn, iters=200):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # microseconds

    def triton_call():
        paged_attention_decode(q_batch, pool.buffer, block_tables, context_lens)

    def pytorch_call():
        for i, seq in enumerate(sequences):
            paged_attention_single(pool, seq, full_qs[i], causal=True)

    t_triton = bench(triton_call)
    t_pytorch = bench(pytorch_call)
    print(
        f"[05] decode latency for batch={B}, seqs lens={seq_lens}:\n"
        f"     pytorch reference: {t_pytorch:8.1f} us/iter\n"
        f"     triton paged-attn: {t_triton:8.1f} us/iter "
        f"({t_pytorch / t_triton:.1f}x speedup)"
    )

    print("[05] OK — Triton kernel matches PyTorch reference within fp16 tolerance.")


if __name__ == "__main__":
    run_demo()
