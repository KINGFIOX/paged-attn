"""Step 01 — Standard (contiguous) attention with a KV cache.

Goal of this file
-----------------
Before we can appreciate *paged* attention, we need a crisp mental model of the
"vanilla" inference path used by every transformer decoder:

    1. **Prefill**: feed the whole prompt of length L_p, compute attention in a
       single shot, and save K/V into a cache.
    2. **Decode**: at every step, append one new token's K/V to the cache and
       attend over the *full* cache (length grows by 1 per step).

We implement both phases here with one contiguous tensor per sequence
(`[B, H, max_len, D]`). That is the layout vLLM is trying to *replace*. We
will reuse this file as the numerical reference for every later optimization.

Run:
    uv run python paged_attn/01_standard_attention.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. A tiny multi-head attention module (no projections - we feed Q/K/V directly).
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,  # [B, H, Lq, D]
    k: torch.Tensor,  # [B, H, Lk, D]
    v: torch.Tensor,  # [B, H, Lk, D]
    causal: bool = True,
) -> torch.Tensor:
    """Reference attention.  No flash, no fused kernels — just math."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Lq, Lk]
    if causal:
        # Causal mask aligned to the *right* edge of K so it works for decode
        # (where Lq == 1 and Lk grows over time).
        Lq, Lk = scores.shape[-2], scores.shape[-1]
        # Each query at position (Lk - Lq + i) may only see keys [0, Lk - Lq + i].
        i = torch.arange(Lq, device=scores.device).view(Lq, 1)
        j = torch.arange(Lk, device=scores.device).view(1, Lk)
        mask = j > (Lk - Lq) + i  # True == disallowed
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)  # [B, H, Lq, D]


# ---------------------------------------------------------------------------
# 2. The "contiguous" KV cache: one fixed-size buffer per sequence.
# ---------------------------------------------------------------------------


@dataclass
class ContiguousKVCache:
    """KV cache laid out as `[B, H, max_len, D]` per layer.

    This is the classic HuggingFace/Megatron layout. The killer property — and
    the reason PagedAttention exists — is that we must *pre-allocate* `max_len`
    slots per sequence even if the sequence only uses a fraction of them.
    """

    k: torch.Tensor  # [B, H, max_len, D]
    v: torch.Tensor  # [B, H, max_len, D]
    lengths: torch.Tensor  # [B], current valid length per sequence

    @classmethod
    def empty(cls, batch: int, heads: int, max_len: int, head_dim: int,
              device: torch.device, dtype: torch.dtype) -> "ContiguousKVCache":
        return cls(
            k=torch.zeros(batch, heads, max_len, head_dim, device=device, dtype=dtype),
            v=torch.zeros(batch, heads, max_len, head_dim, device=device, dtype=dtype),
            lengths=torch.zeros(batch, dtype=torch.int32, device=device),
        )

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append L tokens (same L for every batch row) to the cache."""
        B, H, L, D = k_new.shape
        assert k_new.shape == v_new.shape
        # In real systems each row may have a different length; for clarity here
        # we assume the batch is aligned. (Step 06 lifts this assumption.)
        offset = int(self.lengths[0].item())
        assert torch.all(self.lengths == offset), "this toy cache assumes aligned lengths"
        self.k[:, :, offset:offset + L] = k_new
        self.v[:, :, offset:offset + L] = v_new
        self.lengths += L

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently-valid prefix of K/V (still padded shape)."""
        return self.k, self.v


# ---------------------------------------------------------------------------
# 3. Prefill + decode demo, with an equivalence check.
# ---------------------------------------------------------------------------


def run_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B, H, D = 2, 4, 16
    prompt_len = 5
    decode_steps = 7
    max_len = prompt_len + decode_steps

    # Fake Q/K/V tensors. We mimic what a transformer layer would produce after
    # the input projections.  Generating decode tokens one-by-one gives the same
    # numbers as running prefill over the entire sequence at once — that
    # equivalence is the whole point of a KV cache.
    full_q = torch.randn(B, H, max_len, D, device=device, dtype=dtype)
    full_k = torch.randn(B, H, max_len, D, device=device, dtype=dtype)
    full_v = torch.randn(B, H, max_len, D, device=device, dtype=dtype)

    # ---- (a) "Oracle": one-shot attention over the whole sequence. ----
    oracle = scaled_dot_product_attention(full_q, full_k, full_v, causal=True)

    # ---- (b) Prefill the prompt, then decode token-by-token. ----
    cache = ContiguousKVCache.empty(B, H, max_len, D, device=device, dtype=dtype)

    # Prefill: queries are the prompt itself.
    q_p = full_q[:, :, :prompt_len]
    k_p = full_k[:, :, :prompt_len]
    v_p = full_v[:, :, :prompt_len]
    cache.append(k_p, v_p)
    k_cache, v_cache = cache.view()
    out_prefill = scaled_dot_product_attention(
        q_p, k_cache[:, :, :prompt_len], v_cache[:, :, :prompt_len], causal=True,
    )

    out_chunks = [out_prefill]
    for step in range(decode_steps):
        pos = prompt_len + step
        q_t = full_q[:, :, pos:pos + 1]
        k_t = full_k[:, :, pos:pos + 1]
        v_t = full_v[:, :, pos:pos + 1]
        cache.append(k_t, v_t)
        cur_len = int(cache.lengths[0].item())
        k_cache, v_cache = cache.view()
        # NOTE: during decode the query length is 1, so the causal mask is a
        # no-op — every cached key is to the left of the new token.
        out_t = scaled_dot_product_attention(
            q_t, k_cache[:, :, :cur_len], v_cache[:, :, :cur_len], causal=True,
        )
        out_chunks.append(out_t)

    out_streamed = torch.cat(out_chunks, dim=2)

    # ---- (c) Sanity check: streaming == one-shot. ----
    max_err = (oracle - out_streamed).abs().max().item()
    print(f"[01] max |oracle - kv_cache_decode| = {max_err:.2e}")
    assert max_err < 1e-4, "KV cache decode disagrees with one-shot attention!"

    # ---- (d) Quantify the memory cost of contiguous layout. ----
    # vLLM observed that real workloads pre-allocate to max_model_len but rarely
    # use it.  Even in this toy run, look at the slack if max_len doubled:
    bytes_per_elem = torch.finfo(dtype).bits // 8
    used_bytes = B * H * cache.lengths[0].item() * D * bytes_per_elem * 2  # K and V
    pessimistic_max = 4 * max_len  # imagine a 4x safety factor
    reserved_bytes = B * H * pessimistic_max * D * bytes_per_elem * 2
    print(
        f"[01] used KV bytes = {used_bytes:>8d} | "
        f"pessimistically reserved = {reserved_bytes:>8d} | "
        f"waste = {1 - used_bytes / reserved_bytes:.0%}"
    )
    print("[01] OK — contiguous KV cache works, but every byte beyond `lengths` is wasted.")


if __name__ == "__main__":
    run_demo()
