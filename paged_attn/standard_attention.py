"""Step 01 — Standard (contiguous) attention with a KV cache.

Goal of this file
-----------------
Before we can appreciate *paged* attention, we need a crisp mental model of the
"vanilla" inference path used by every transformer decoder:

    1. **Prefill**: feed the whole prompt of length L_p, run it through the
       attention layer (W_q/W_k/W_v projections + self-attention) in one shot,
       and save K/V into a cache.
    2. **Decode**: at every step, *only project the single new token's Q/K/V*,
       append its K/V to the cache, and attend with Lq=1 over the *full* cache.

Both phases produce identical outputs to the "no-cache" baseline that
re-projects the entire sequence every step — but the costs differ wildly.
The KV cache trades a fixed-size buffer for never re-projecting old K/V.

Without W_q/W_k/W_v in the picture, there's nothing to save: that's why this
file uses a real `nn.Linear`-backed attention layer, not just a bare dot
product.  The cache then saves both:

    * **Projection FLOPs**:  3·L·d²    →    3·d²        per decode step.
    * **Attention FLOPs**:   (L+t)²·d  →    (L+t)·d     per decode step.

Run:
    uv run python -m paged_attn.standard_attention
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Reference scaled-dot-product attention (no flash, no fused kernels).
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,  # [B, H, Lq, D]
    k: torch.Tensor,  # [B, H, Lk, D]
    v: torch.Tensor,  # [B, H, Lk, D]
    causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale  # [B, H, Lq, Lk]
    if causal:
        # Mask aligned to the *right* edge of K so this works for decode too
        # (Lq == 1 and Lk grows over time).
        # Lq either equals to 1 or equals to the length of prompt
        Lq, Lk = scores.shape[-2], scores.shape[-1]
        i = torch.arange(Lq, device=scores.device).view(Lq, 1)
        j = torch.arange(Lk, device=scores.device).view(1, Lk)
        mask = j > (Lk - Lq) + i  # True == disallowed
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)  # [B, H, Lq, D]


# ---------------------------------------------------------------------------
# 2. A tiny self-attention layer WITH projections — this is where KV-cache wins.
# ---------------------------------------------------------------------------


class MiniSelfAttention(nn.Module):
    """One causal self-attention block: x -> (W_q,W_k,W_v) -> attention -> out.

    No output projection / no MLP — those don't affect the KV cache argument.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

    def _project(self, x: torch.Tensor):
        """`x: [B, L, d_model]`  ->  q,k,v: each `[B, H, L, head_dim]`."""
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    # ------------------------------------------------------------------
    # Path A: no cache.  Every call re-projects ALL the tokens in `x`.
    # ------------------------------------------------------------------
    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward over the whole sequence; returns `[B, H, L, head_dim]`."""
        q, k, v = self._project(x)
        return scaled_dot_product_attention(q, k, v, causal=True)

    # ------------------------------------------------------------------
    # Path B: with cache.  Only projects the NEW tokens; appends to cache.
    # ------------------------------------------------------------------
    def forward_with_cache(self, x_new: torch.Tensor,
                           cache: "ContiguousKVCache") -> torch.Tensor:
        """Project only `x_new`, append its K/V to `cache`, attend over the
        full (cached + new) K/V history.

        `x_new` is `[B, L_new, d_model]`.  Returns `[B, H, L_new, head_dim]`.
        """
        q_new, k_new, v_new = self._project(x_new)  # each [B, H, L_new, hd]
        cache.append(k_new, v_new)
        k_all, v_all = cache.view()
        return scaled_dot_product_attention(q_new, k_all, v_all, causal=True)


# ---------------------------------------------------------------------------
# 3. The "contiguous" KV cache: one fixed-size buffer per sequence per layer.
# ---------------------------------------------------------------------------


@dataclass
class ContiguousKVCache:
    """KV cache laid out as `[B, H, max_len, head_dim]`.

    This is the classic HuggingFace/Megatron layout.  The killer property — and
    the reason PagedAttention exists — is that we must *pre-allocate* `max_len`
    slots per sequence even if the sequence only uses a fraction of them.
    """

    k: torch.Tensor       # [B, H, max_len, head_dim]
    v: torch.Tensor       # [B, H, max_len, head_dim]
    lengths: torch.Tensor  # [B], current valid length per sequence

    @classmethod
    def empty(cls, batch: int, n_heads: int, max_len: int, head_dim: int,
              device: torch.device, dtype: torch.dtype) -> "ContiguousKVCache":
        return cls(
            k=torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype),
            v=torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype),
            lengths=torch.zeros(batch, dtype=torch.int32, device=device),
        )

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Append `L_new` tokens (same for every batch row) to the cache."""
        assert k_new.shape == v_new.shape
        B, H, L_new, D = k_new.shape
        # For clarity this toy assumes every batch row has the same cache length.
        offset = int(self.lengths[0].item())
        assert torch.all(self.lengths == offset), "this toy cache assumes aligned lengths"
        self.k[:, :, offset:offset + L_new] = k_new
        self.v[:, :, offset:offset + L_new] = v_new
        self.lengths += L_new

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently-valid prefix of K/V (sliced to current length)."""
        L = int(self.lengths[0].item())
        return self.k[:, :, :L], self.v[:, :, :L]


# ---------------------------------------------------------------------------
# 4. Demo: equivalence + cost — no cache vs with cache, with REAL projections.
# ---------------------------------------------------------------------------


def _flops_no_cache_step(L: int, d: int, n_heads: int) -> int:
    """Rough FLOPs of one full self-attention forward over `L` tokens.

    Projection part:   3 * (L * d * d)   (three GEMMs of [L, d] x [d, d])
    Attention scores:      L * L * d     (QK^T)
    Softmax * V:           L * L * d
    """
    return 3 * L * d * d + 2 * L * L * d


def _flops_with_cache_decode_step(L_total: int, d: int) -> int:
    """One decode step: project ONE new token, attend with Lq=1, Lk=L_total."""
    return 3 * 1 * d * d + 2 * 1 * L_total * d  # projections + (QK^T + attn*V)


def run_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    B, n_heads, d_model = 2, 4, 128
    head_dim = d_model // n_heads
    prompt_len = 32
    decode_steps = 64
    max_len = prompt_len + decode_steps

    attn = MiniSelfAttention(d_model, n_heads).to(device=device, dtype=dtype)

    # Fake "input tokens" — in a real transformer these come from the embedding
    # layer (or the previous block's output).  Different tokens for each batch row.
    tokens = torch.randn(B, max_len, d_model, device=device, dtype=dtype)

    # ----------------------------------------------------------------------
    # Path A — no KV cache.  At every step `t`, re-project the *entire*
    #          sequence-so-far and run full attention over it.
    # ----------------------------------------------------------------------
    last_outputs_a: list[torch.Tensor] = []
    flops_a = 0
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()

    # Prefill step: forward over the prompt.
    out_p = attn.forward_no_cache(tokens[:, :prompt_len])             # [B, H, Lp, hd]
    last_outputs_a.append(out_p)
    flops_a += _flops_no_cache_step(prompt_len, d_model, n_heads)

    for t in range(decode_steps):
        # The "true" generation loop: the model has consumed prompt + t tokens
        # and now wants to compute attention output at position prompt_len+t.
        L_so_far = prompt_len + t + 1
        out = attn.forward_no_cache(tokens[:, :L_so_far])             # [B, H, L, hd]
        last_outputs_a.append(out[:, :, -1:, :])                      # only new col
        flops_a += _flops_no_cache_step(L_so_far, d_model, n_heads)

    torch.cuda.synchronize() if device.type == "cuda" else None
    time_a = time.perf_counter() - t_start
    out_full_a = torch.cat(last_outputs_a, dim=2)                     # [B, H, max_len, hd]

    # ----------------------------------------------------------------------
    # Path B — with KV cache.  Prefill once; each decode step projects ONLY
    #          the single new token.
    # ----------------------------------------------------------------------
    cache = ContiguousKVCache.empty(B, n_heads, max_len, head_dim, device, dtype)
    last_outputs_b: list[torch.Tensor] = []
    flops_b = 0
    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()

    # Prefill
    out_p = attn.forward_with_cache(tokens[:, :prompt_len], cache)    # [B, H, Lp, hd]
    last_outputs_b.append(out_p)
    flops_b += _flops_no_cache_step(prompt_len, d_model, n_heads)     # prefill is the same

    for t in range(decode_steps):
        new_token = tokens[:, prompt_len + t:prompt_len + t + 1]      # [B, 1, d_model]
        out = attn.forward_with_cache(new_token, cache)               # [B, H, 1, hd]
        last_outputs_b.append(out)
        flops_b += _flops_with_cache_decode_step(prompt_len + t + 1, d_model)

    torch.cuda.synchronize() if device.type == "cuda" else None
    time_b = time.perf_counter() - t_start
    out_full_b = torch.cat(last_outputs_b, dim=2)                     # [B, H, max_len, hd]

    # ----------------------------------------------------------------------
    # (a) Sanity: both paths produce numerically identical outputs.
    # ----------------------------------------------------------------------
    max_err = (out_full_a - out_full_b).abs().max().item()
    print(f"[01] max |no_cache - kv_cache| over all decoded tokens = {max_err:.2e}")
    assert max_err < 1e-4, "KV cache decode disagrees with no-cache reference!"

    # ----------------------------------------------------------------------
    # (b) The whole point: with projections in the loop, the cache saves work.
    # ----------------------------------------------------------------------
    print(
        f"[01] d_model={d_model}, n_heads={n_heads}, "
        f"prompt_len={prompt_len}, decode_steps={decode_steps}"
    )
    print(
        f"[01] FLOPs (rough): no_cache = {flops_a/1e6:8.2f} M  |  "
        f"with_cache = {flops_b/1e6:8.2f} M  |  "
        f"ratio = {flops_a/flops_b:5.1f}x"
    )
    print(
        f"[01] wall time   : no_cache = {time_a*1e3:8.2f} ms |  "
        f"with_cache = {time_b*1e3:8.2f} ms |  "
        f"ratio = {time_a/time_b:5.1f}x"
    )

    # ----------------------------------------------------------------------
    # (c) Quantify the memory cost of the *contiguous* layout — this is the
    #     pain that step 02 will measure at GiB scale and step 03+ will fix.
    # ----------------------------------------------------------------------
    bytes_per_elem = torch.finfo(dtype).bits // 8
    used_bytes = 2 * B * n_heads * cache.lengths[0].item() * head_dim * bytes_per_elem
    pessimistic_max = 4 * max_len  # imagine a 4x safety factor (max_model_len ≫ used)
    reserved_bytes = 2 * B * n_heads * pessimistic_max * head_dim * bytes_per_elem
    print(
        f"[01] used KV bytes = {used_bytes:>8d} | "
        f"pessimistically reserved = {reserved_bytes:>8d} | "
        f"waste = {1 - used_bytes / reserved_bytes:.0%}"
    )
    print("[01] OK — KV cache cuts per-step work, but every byte beyond `lengths` is wasted.")


if __name__ == "__main__":
    run_demo()
