"""Step 02 — Why we need PagedAttention: quantifying KV-cache waste.

In step 01 we saw that the contiguous KV cache wastes every byte beyond the
sequence's current length.  That sounds small until you actually plug in a
real serving workload — 7B/13B/70B models, dozens of concurrent requests,
context windows of 4k–32k tokens, and sequence lengths that vary by 100x.

This file does no fancy attention math; it just *accounts*.  We pick a
Llama-like config, simulate a realistic mix of in-flight requests, and print
how badly the classic layout fragments memory.

Run:
    uv run python paged_attn/02_kv_cache_fragmentation.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table


@dataclass
class ModelConfig:
    name: str
    n_layers: int
    n_kv_heads: int     # GQA: K/V heads can be fewer than Q heads
    head_dim: int
    dtype_bytes: int = 2  # fp16 / bf16
    max_seq_len: int = 4096

    @property
    def bytes_per_token(self) -> int:
        # Two tensors (K and V), per layer, per KV head.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes


LLAMA2_7B = ModelConfig("Llama-2-7B", n_layers=32, n_kv_heads=32, head_dim=128)
LLAMA3_8B = ModelConfig("Llama-3-8B", n_layers=32, n_kv_heads=8,  head_dim=128)
LLAMA2_70B = ModelConfig("Llama-2-70B", n_layers=80, n_kv_heads=8, head_dim=128)


def simulate_batch(rng: np.random.Generator, n_seq: int, model: ModelConfig,
                   short_frac: float = 0.7):
    """Make a plausible mix of sequence lengths.

    Real serving traces are extremely skewed: most chat turns are short, a few
    are long.  We model that as a mixture of two log-normal distributions.
    """
    n_short = int(n_seq * short_frac)
    n_long = n_seq - n_short
    short = rng.lognormal(mean=np.log(120), sigma=0.5, size=n_short)
    long = rng.lognormal(mean=np.log(2000), sigma=0.6, size=n_long)
    lens = np.concatenate([short, long])
    lens = np.clip(lens, 1, model.max_seq_len).astype(np.int64)
    rng.shuffle(lens)
    return lens


def account_contiguous(lens: np.ndarray, model: ModelConfig) -> dict:
    """Mimic the HuggingFace-style layout: every request pre-allocates max_seq_len."""
    n = len(lens)
    reserved_tokens = n * model.max_seq_len
    used_tokens = int(lens.sum())
    return {
        "reserved_tokens": reserved_tokens,
        "used_tokens": used_tokens,
        "reserved_bytes": reserved_tokens * model.bytes_per_token,
        "used_bytes": used_tokens * model.bytes_per_token,
    }


def account_paged(lens: np.ndarray, model: ModelConfig, block_size: int) -> dict:
    """Mimic the vLLM layout: each request grabs ceil(len/block_size) blocks."""
    blocks_per_seq = np.ceil(lens / block_size).astype(np.int64)
    reserved_tokens = int(blocks_per_seq.sum()) * block_size
    used_tokens = int(lens.sum())
    return {
        "reserved_tokens": reserved_tokens,
        "used_tokens": used_tokens,
        "reserved_bytes": reserved_tokens * model.bytes_per_token,
        "used_bytes": used_tokens * model.bytes_per_token,
    }


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:6.2f} {u}"
        x /= 1024
    return f"{x:6.2f} PiB"


def run_demo() -> None:
    console = Console()
    rng = np.random.default_rng(42)

    n_seq = 64  # concurrent requests
    lens = simulate_batch(rng, n_seq, LLAMA3_8B)
    console.print(
        f"[bold]Workload[/bold]: {n_seq} concurrent requests, "
        f"length quantiles = "
        f"p50={int(np.percentile(lens, 50))}, "
        f"p90={int(np.percentile(lens, 90))}, "
        f"p99={int(np.percentile(lens, 99))} "
        f"(max_seq_len={LLAMA3_8B.max_seq_len})"
    )

    table = Table(title="KV cache memory: contiguous vs paged")
    table.add_column("Model")
    table.add_column("Layout")
    table.add_column("Reserved")
    table.add_column("Used")
    table.add_column("Utilization", justify="right")
    table.add_column("Slots wasted vs paged-16", justify="right")

    # We compare three models, three layouts each.
    paged_baseline = {}
    for model in (LLAMA2_7B, LLAMA3_8B, LLAMA2_70B):
        cont = account_contiguous(lens, model)
        paged16 = account_paged(lens, model, block_size=16)
        paged_baseline[model.name] = paged16

        for label, stats in [
            ("contiguous (max_seq_len)", cont),
            ("paged, block=16",          paged16),
            ("paged, block=128",         account_paged(lens, model, 128)),
        ]:
            util = stats["used_bytes"] / stats["reserved_bytes"]
            extra = stats["reserved_bytes"] - paged_baseline[model.name]["reserved_bytes"]
            table.add_row(
                model.name,
                label,
                _fmt_bytes(stats["reserved_bytes"]),
                _fmt_bytes(stats["used_bytes"]),
                f"{util:6.1%}",
                _fmt_bytes(max(extra, 0)),
            )
        table.add_section()

    console.print(table)

    # The takeaway line, made unmissable.
    cont_8b = account_contiguous(lens, LLAMA3_8B)["reserved_bytes"]
    paged_8b = account_paged(lens, LLAMA3_8B, 16)["reserved_bytes"]
    console.print(
        f"[bold green]Llama-3-8B saving from paging (block=16): "
        f"{_fmt_bytes(cont_8b - paged_8b)} "
        f"({(cont_8b - paged_8b) / cont_8b:.0%} of the contiguous footprint)[/bold green]"
    )
    console.print(
        "[dim]Notes:\n"
        " - Even paged layout wastes a sub-block tail per sequence (internal fragmentation).\n"
        " - Smaller block_size => less internal waste, but more block-table overhead and\n"
        "   poorer GPU memory coalescing.  vLLM defaults to 16; some kernels prefer 32 or 64.[/dim]"
    )


if __name__ == "__main__":
    run_demo()
