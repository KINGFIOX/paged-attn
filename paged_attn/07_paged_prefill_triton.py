"""Step 07 — Paged *prefill* attention as a Triton kernel.

Step 05's decode kernel only handles Lq == 1.  Real serving systems also need
to compute attention with **Lq > 1**:

    * Fresh prefill: a brand-new request whose entire prompt has to be
      attended over (Lq = prompt_len, query_start = 0).
    * Chunked prefill: an in-flight prompt processed `C` tokens at a time
      (Lq = C, query_start = current_position).  Crucial for fitting long
      prompts into a fixed-shape kernel.
    * Speculative decoding verification: multiple draft tokens scored in one
      shot (Lq = num_draft).

The kernel is Flash-Attention-2 style with paged K/V access:

    grid = (n_heads, ceil(Lq / BLOCK_M))

Each program loads a `BLOCK_M`-row tile of Q, then iterates the sequence's
block table.  Per KV block we compute the [BLOCK_M, BLOCK_N] score tile,
apply the causal mask in *absolute* sequence coordinates (so chunked prefill
just works), and accumulate with the running max / normaliser.

Run:
    uv run python paged_attn/07_paged_prefill_triton.py
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os
import pathlib as _pl
import sys as _sys
import time

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
# Kernel
# ---------------------------------------------------------------------------


@triton.jit
def _paged_prefill_kernel(
    # Pointers
    q_ptr,            # [H, Lq, D]
    pool_ptr,         # [num_blocks, 2, H, BLOCK_N, D]
    block_table_ptr,  # [num_blocks_for_seq]
    out_ptr,          # [H, Lq, D]
    # Sizes
    Lq, context_len, query_start,
    # Strides (elements)
    sq_h, sq_l,
    sp_b, sp_kv, sp_h, sp_t,
    so_h, so_l,
    # Scalars / constants
    scale,
    BLOCK_M: tl.constexpr,   # query tile rows
    BLOCK_N: tl.constexpr,   # == KV pool block_size
    HEAD_DIM: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Query tile coordinates.
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    d_offs = tl.arange(0, HEAD_DIM)                      # [HEAD_DIM]
    n_offs = tl.arange(0, BLOCK_N)                       # [BLOCK_N]
    q_mask = m_offs < Lq                                 # mask out the tail tile

    # Absolute position of each row in this Q tile within the sequence.  This
    # is what makes chunked prefill work: same kernel, just pass a non-zero
    # `query_start` for the second / third / ... chunk.
    q_abs = query_start + m_offs                         # [BLOCK_M]

    # Load the Q tile (zero-padded out-of-range rows; they'll be masked off).
    q_ptrs = q_ptr + pid_h * sq_h + m_offs[:, None] * sq_l + d_offs[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax state — one running (m, l) pair per Q row.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    num_kv_blocks = (context_len + BLOCK_N - 1) // BLOCK_N
    for kv_blk in range(0, num_kv_blocks):
        phys = tl.load(block_table_ptr + kv_blk)

        k_abs = kv_blk * BLOCK_N + n_offs                # [BLOCK_N]
        kv_mask = k_abs < context_len                    # right edge of K cache

        # Load K, V for this paged block.
        k_ptrs = (pool_ptr + phys * sp_b + 0 * sp_kv + pid_h * sp_h
                  + n_offs[:, None] * sp_t + d_offs[None, :])
        v_ptrs = (pool_ptr + phys * sp_b + 1 * sp_kv + pid_h * sp_h
                  + n_offs[:, None] * sp_t + d_offs[None, :])
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Score tile [BLOCK_M, BLOCK_N].
        scores = tl.dot(q, tl.trans(k)) * scale

        # Combined mask: (1) Q row is in range, (2) K col is in range,
        # (3) causal — column j is only allowed if k_abs[j] <= q_abs[i].
        causal = k_abs[None, :] <= q_abs[:, None]
        valid = causal & kv_mask[None, :] & q_mask[:, None]
        scores = tl.where(valid, scores, -float("inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])              # [BLOCK_M, BLOCK_N]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = alpha * l_i + tl.sum(p, axis=1)
        m_i = m_new

    # Some rows may have received zero unmasked keys (e.g. the first query row
    # of a fresh prefill attending only to itself with a causal mask is fine,
    # but a *padded* tail row sees nothing).  l_i could be 0 for them; guard.
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    out = acc / safe_l[:, None]

    out_ptrs = out_ptr + pid_h * so_h + m_offs[:, None] * so_l + d_offs[None, :]
    tl.store(out_ptrs, out, mask=q_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def paged_attention_prefill(
    q: torch.Tensor,             # [H, Lq, D]
    pool_buffer: torch.Tensor,   # [num_blocks, 2, H, BS, D]
    block_table: torch.Tensor,   # [num_blocks_for_seq], int32
    context_len: int,
    query_start: int = 0,
    block_m: int = 32,
) -> torch.Tensor:
    """Paged prefill / chunked-prefill attention for one sequence.

    Args
    ----
    q            : queries for the `Lq` new tokens, shape `[H, Lq, D]`.
    pool_buffer  : full paged KV pool.
    block_table  : the sequence's physical block table.
    context_len  : how many tokens of K/V are currently valid in the pool
                   for this sequence (>= query_start + Lq).
    query_start  : absolute position of `q[:, 0]` in the sequence (0 for
                   fresh prefill; non-zero for chunked prefill).

    Each Q row attends only to K positions `<= query_start + row_idx`.
    """
    assert q.dim() == 3, f"q must be [H, Lq, D], got {q.shape}"
    H, Lq, D = q.shape
    NB, KV, Hp, BS, Dp = pool_buffer.shape
    assert KV == 2 and Hp == H and Dp == D, "pool/query shape mismatch"
    assert context_len >= query_start + Lq, "context must cover the queries"
    assert block_table.dim() == 1

    out = torch.empty_like(q)
    grid = (H, triton.cdiv(Lq, block_m))

    sq_h, sq_l, _ = q.stride()
    sp_b, sp_kv, sp_h, sp_t, _ = pool_buffer.stride()
    so_h, so_l, _ = out.stride()

    _paged_prefill_kernel[grid](
        q, pool_buffer, block_table, out,
        Lq, context_len, query_start,
        sq_h, sq_l,
        sp_b, sp_kv, sp_h, sp_t,
        so_h, so_l,
        scale=1.0 / (D ** 0.5),
        BLOCK_M=block_m,
        BLOCK_N=BS,
        HEAD_DIM=D,
    )
    return out


# ---------------------------------------------------------------------------
# Demo: fresh prefill + chunked prefill + benchmark
# ---------------------------------------------------------------------------


def run_demo() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.float16

    H, D = 8, 64
    block_size = 16
    prompt_len = 257            # not a multiple of block_size / BLOCK_M on purpose

    num_blocks = (prompt_len + block_size - 1) // block_size + 4
    pool = KVPool(num_blocks=num_blocks, n_heads=H, block_size=block_size,
                  head_dim=D, dtype=dtype, device=device)
    mgr = BlockManager(pool)

    seq = mgr.new_sequence()
    mgr.append_tokens(seq, prompt_len)

    torch.manual_seed(11)
    full_q = torch.randn(H, prompt_len, D, device=device, dtype=dtype)
    full_k = torch.randn(H, prompt_len, D, device=device, dtype=dtype)
    full_v = torch.randn(H, prompt_len, D, device=device, dtype=dtype)
    store_kv(pool, seq, full_k, full_v, token_offset=0)

    block_table = torch.tensor(seq.block_table, device=device, dtype=torch.int32)

    # ---- (a) PyTorch reference: paged + naive attention. ----
    ref = paged_attention_single(pool, seq, full_q, causal=True)        # [H, Lq, D]

    # ---- (b) Fresh prefill in one Triton launch. ----
    out_full = paged_attention_prefill(
        full_q, pool.buffer, block_table, context_len=prompt_len, query_start=0,
    )
    err_full = (ref.float() - out_full.float()).abs().max().item()
    print(f"[07] fresh prefill, Lq={prompt_len}: max |triton - reference| = {err_full:.3e}")
    assert err_full < 5e-2, f"prefill kernel error too large ({err_full})"

    # ---- (c) Chunked prefill: same answer, but processed `chunk` tokens at a time. ----
    chunk = 64
    chunks = []
    for q_start in range(0, prompt_len, chunk):
        q_end = min(q_start + chunk, prompt_len)
        q_chunk = full_q[:, q_start:q_end].contiguous()
        # During *real* chunked prefill, we would write K/V for this chunk
        # into the pool *before* the kernel call.  Here we already prefilled
        # the whole sequence, so we just pretend context_len grows.
        ctx_len = q_end
        out_c = paged_attention_prefill(
            q_chunk, pool.buffer, block_table,
            context_len=ctx_len, query_start=q_start,
        )
        chunks.append(out_c)
    out_chunked = torch.cat(chunks, dim=1)
    err_chunk = (ref.float() - out_chunked.float()).abs().max().item()
    print(f"[07] chunked prefill, chunk={chunk}: max |triton - reference| = {err_chunk:.3e}")
    assert err_chunk < 5e-2, f"chunked prefill error too large ({err_chunk})"

    # ---- (d) Round-trip with decode: prefill (Lq=prompt) + 4 decode steps. ----
    decode_steps = 4
    # Reserve more space, write new K/V, run decode kernel (step 05) — but that's
    # the *decode* kernel.  Here we use the prefill kernel itself with Lq=1
    # per step, which is valid and produces identical numerics (just slower).
    for s in range(decode_steps):
        mgr.append_tokens(seq, 1)
        pos = prompt_len + s
        k_new = torch.randn(H, 1, D, device=device, dtype=dtype)
        v_new = torch.randn(H, 1, D, device=device, dtype=dtype)
        store_kv(pool, seq, k_new, v_new, token_offset=pos)
        # Refresh the block_table tensor (may have grown).
        block_table = torch.tensor(seq.block_table, device=device, dtype=torch.int32)
        q_new = torch.randn(H, 1, D, device=device, dtype=dtype)
        out_d = paged_attention_prefill(
            q_new, pool.buffer, block_table,
            context_len=pos + 1, query_start=pos,
        )
        # Sanity vs paged_attention_single on the same q
        ref_d = paged_attention_single(pool, seq, q_new, causal=True)
        err_d = (ref_d.float() - out_d.float()).abs().max().item()
        assert err_d < 5e-2, f"decode-step error at step {s}: {err_d}"
    print(f"[07] decode via prefill kernel (Lq=1, query_start>0): "
          f"{decode_steps} steps all match reference")

    # ---- (e) Micro-benchmark on a longer prompt. ----
    def bench(fn, iters=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # ms

    # Build a longer prefill scenario.
    L = 2048
    nb = (L + block_size - 1) // block_size + 4
    pool2 = KVPool(num_blocks=nb, n_heads=H, block_size=block_size,
                   head_dim=D, dtype=dtype, device=device)
    mgr2 = BlockManager(pool2)
    seq2 = mgr2.new_sequence()
    mgr2.append_tokens(seq2, L)
    qb = torch.randn(H, L, D, device=device, dtype=dtype)
    kb = torch.randn(H, L, D, device=device, dtype=dtype)
    vb = torch.randn(H, L, D, device=device, dtype=dtype)
    store_kv(pool2, seq2, kb, vb, token_offset=0)
    bt = torch.tensor(seq2.block_table, device=device, dtype=torch.int32)

    def triton_prefill():
        paged_attention_prefill(qb, pool2.buffer, bt, context_len=L, query_start=0)

    def pytorch_prefill():
        paged_attention_single(pool2, seq2, qb, causal=True)

    t_t = bench(triton_prefill)
    t_p = bench(pytorch_prefill, iters=20)
    print(
        f"[07] prefill latency Lq=L={L}, H={H}, D={D}:\n"
        f"     pytorch reference: {t_p:7.2f} ms/iter\n"
        f"     triton prefill   : {t_t:7.2f} ms/iter "
        f"({t_p/t_t:.1f}x speedup)"
    )
    print("[07] OK — paged prefill kernel matches reference for full + chunked + decode-like calls.")


if __name__ == "__main__":
    run_demo()
