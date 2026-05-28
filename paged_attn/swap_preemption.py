"""Step 08 — Swap-based preemption (CPU swap pool).

What happens when the GPU pool is full but a higher-priority request shows up?
Two well-known strategies:

    1. **Recompute** — drop the victim's KV entirely, requeue it; when it
       resumes, re-run prefill from scratch.  Simple, no extra memory, but
       wastes compute.  This is what we'll wire into the engine (step 06).

    2. **Swap**   — copy the victim's KV blocks to a CPU "swap" pool, free
       its GPU blocks, and copy them back when there's room again.  No
       wasted compute; costs PCIe bandwidth and ~3x KV memory total.

Step 08 implements strategy #2 as a `SwappableBlockManager` that extends the
block manager from step 03 with two methods: `swap_out` and `swap_in`.

Constraints we adopt for clarity:
    - A swappable sequence must own its blocks (refcount == 1 everywhere).
      Shared (forked) prefix blocks would need to be swapped together with
      every refcount holder, which is doable but distracts from the idea.
    - Swap moves the K/V buffer only; the block table and length are
      preserved as-is — only the "device" of each entry changes.

Run:
    uv run python -m paged_attn.swap_preemption
"""

from __future__ import annotations

import torch
from rich.console import Console

from .block_manager import (
    BlockAllocator,
    BlockManager,
    KVPool,
    OutOfBlocks,
    Sequence,
)
from .paged_attention_naive import gather_kv


# ---------------------------------------------------------------------------
# The swappable block manager.
# ---------------------------------------------------------------------------


class SwappableBlockManager(BlockManager):
    """A BlockManager that can move a sequence's blocks between GPU and CPU.

    We deliberately use a separate `KVPool` for CPU storage — same shape, same
    dtype, just `device="cpu"`.  In real systems people often `pin_memory()`
    the CPU pool so D2H/H2D copies overlap with kernel execution; we keep that
    out for simplicity.
    """

    def __init__(self, gpu_pool: KVPool, cpu_pool: KVPool):
        super().__init__(gpu_pool)
        assert gpu_pool.block_size == cpu_pool.block_size
        assert gpu_pool.n_heads == cpu_pool.n_heads
        assert gpu_pool.head_dim == cpu_pool.head_dim
        assert gpu_pool.dtype == cpu_pool.dtype
        assert gpu_pool.device.type == "cuda" and cpu_pool.device.type == "cpu"
        self.cpu_pool = cpu_pool
        self.cpu_alloc = BlockAllocator(cpu_pool.num_blocks)
        self._on_cpu: dict[int, bool] = {}   # seq_id -> True iff currently swapped out

    # ---- queries ------------------------------------------------------------

    def is_swapped_out(self, seq: Sequence) -> bool:
        return self._on_cpu.get(seq.seq_id, False)

    def cpu_free_blocks(self) -> int:
        return self.cpu_alloc.num_free

    # ---- the two interesting operations -------------------------------------

    def swap_out(self, seq: Sequence) -> None:
        """Move `seq`'s blocks from GPU to CPU.  Releases GPU blocks."""
        if self.is_swapped_out(seq):
            return
        # Block must be owned exclusively — otherwise other sequences would
        # silently lose access to it.
        for b in seq.block_table:
            rc = self.alloc.refcount(b)
            if rc != 1:
                raise RuntimeError(
                    f"can't swap_out seq {seq.seq_id}: block {b} has refcount {rc} "
                    f"(shared with other sequences)"
                )
        # Reserve CPU blocks first so a partial failure can't leave us in a
        # half-swapped state.
        cpu_ids = [self.cpu_alloc.allocate() for _ in seq.block_table]
        # Bulk D2H copy.  We loop one block at a time to keep the code
        # transparent; you can vectorize with `index_select` for production.
        for gpu_b, cpu_b in zip(seq.block_table, cpu_ids):
            self.cpu_pool.buffer[cpu_b].copy_(self.pool.buffer[gpu_b])
        for gpu_b in seq.block_table:
            self.alloc.free(gpu_b)
        seq.block_table = cpu_ids
        self._on_cpu[seq.seq_id] = True

    def swap_in(self, seq: Sequence) -> None:
        """Move `seq`'s blocks back from CPU to GPU.  Releases CPU blocks."""
        if not self.is_swapped_out(seq):
            return
        if self.alloc.num_free < len(seq.block_table):
            raise OutOfBlocks(
                f"swap_in seq {seq.seq_id}: needs {len(seq.block_table)} GPU blocks, "
                f"have {self.alloc.num_free}"
            )
        gpu_ids = [self.alloc.allocate() for _ in seq.block_table]
        for cpu_b, gpu_b in zip(seq.block_table, gpu_ids):
            self.pool.buffer[gpu_b].copy_(self.cpu_pool.buffer[cpu_b])
        for cpu_b in seq.block_table:
            self.cpu_alloc.free(cpu_b)
        seq.block_table = gpu_ids
        self._on_cpu[seq.seq_id] = False


# ---------------------------------------------------------------------------
# Demo: out-of-memory => swap a victim out => admit a new request => later
# swap the victim back in => verify data integrity.
# ---------------------------------------------------------------------------


def _fingerprint_seq(pool: KVPool, seq: Sequence) -> tuple:
    """Return a (mean, std, l2) signature of a sequence's KV — used to check
    that swap_out + swap_in is lossless."""
    k, v = gather_kv(pool, seq)
    return (
        float(k.float().mean()),
        float(k.float().std()),
        float(v.float().norm()),
    )


def _print_pool(console: Console, mgr: SwappableBlockManager, label: str) -> None:
    util = mgr.utilization()
    console.print(f"\n[bold]{label}[/bold]")
    for sid, s in mgr.sequences.items():
        where = "cpu" if mgr.is_swapped_out(s) else "gpu"
        console.print(
            f"  seq {sid:>2} ({where}): len={s.length:>3}  blocks={s.block_table}"
        )
    console.print(
        f"  gpu pool: {util['blocks_used']}/{mgr.pool.num_blocks} used, "
        f"cpu pool: {mgr.cpu_pool.num_blocks - mgr.cpu_free_blocks()}"
        f"/{mgr.cpu_pool.num_blocks} used"
    )


def run_demo() -> None:
    console = Console()
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    block_size, H, D = 4, 2, 8
    gpu_pool = KVPool(num_blocks=4, n_heads=H, block_size=block_size, head_dim=D,
                      dtype=torch.float32, device=device)
    cpu_pool = KVPool(num_blocks=8, n_heads=H, block_size=block_size, head_dim=D,
                      dtype=torch.float32, device=torch.device("cpu"))
    mgr = SwappableBlockManager(gpu_pool, cpu_pool)

    console.print(
        f"[dim]GPU pool: {gpu_pool.num_blocks} blocks (capacity "
        f"{gpu_pool.num_blocks * block_size} tokens, "
        f"{gpu_pool.bytes()/1024:.1f} KiB).\n"
        f"CPU pool: {cpu_pool.num_blocks} blocks (capacity "
        f"{cpu_pool.num_blocks * block_size} tokens).[/dim]"
    )

    # ---- (a) Fill the GPU pool with two independent sequences. ----
    torch.manual_seed(0)
    seqs: list[Sequence] = []
    fingerprints: dict[int, tuple] = {}
    for prio, L in enumerate([6, 7]):
        s = mgr.new_sequence()
        k = torch.randn(H, L, D, device=device)
        v = torch.randn(H, L, D, device=device)
        mgr.append_kv(s, k, v)
        seqs.append(s)
        fingerprints[s.seq_id] = _fingerprint_seq(gpu_pool, s)
    _print_pool(console, mgr, "After admitting seq 0 (len 6) and seq 1 (len 7)")
    # 6 tokens fit in 2 blocks; 7 in 2 blocks; total 4 blocks => pool is full.

    # ---- (b) A high-priority sequence (len 5 = 2 blocks) needs to be admitted. ----
    console.print("\n[bold yellow]A new high-priority request (len 5) arrives.[/bold yellow]")
    try:
        bad = mgr.new_sequence()
        mgr.reserve_slots(bad, 5)
    except OutOfBlocks as e:
        console.print(f"  [red]OutOfBlocks on GPU pool, as expected: {e}[/red]")
        mgr.free_sequence(bad)

    # ---- (c) Pick a victim and swap it out to CPU. ----
    victim = seqs[0]
    console.print(f"\nPreempting victim = seq {victim.seq_id}: swap_out -> CPU")
    mgr.swap_out(victim)
    _print_pool(console, mgr, "After swap_out")

    # ---- (d) Now there's room. Admit the new request. ----
    new_seq = mgr.new_sequence()
    k_new = torch.randn(H, 5, D, device=device)
    v_new = torch.randn(H, 5, D, device=device)
    mgr.append_kv(new_seq, k_new, v_new)
    fingerprints[new_seq.seq_id] = _fingerprint_seq(gpu_pool, new_seq)
    _print_pool(console, mgr, "After admitting the new request")

    # ---- (e) New request finishes; free it; swap the victim back in. ----
    mgr.free_sequence(new_seq)
    console.print(f"\nNew request done.  Swap victim seq {victim.seq_id} back -> GPU.")
    mgr.swap_in(victim)
    _print_pool(console, mgr, "After swap_in")

    # ---- (f) Verify the victim's K/V is bit-equal to what was stored. ----
    fp_after = _fingerprint_seq(gpu_pool, victim)
    fp_before = fingerprints[victim.seq_id]
    diffs = [abs(a - b) for a, b in zip(fp_after, fp_before)]
    console.print(
        f"\nFingerprint(victim) before swap_out  = "
        f"mean={fp_before[0]:+.6f} std={fp_before[1]:.6f} ||v||={fp_before[2]:.6f}\n"
        f"Fingerprint(victim) after swap_in    = "
        f"mean={fp_after[0]:+.6f} std={fp_after[1]:.6f} ||v||={fp_after[2]:.6f}\n"
        f"max(abs diff) = {max(diffs):.2e}  "
        f"(should be exactly 0 — copy is lossless)"
    )
    assert max(diffs) == 0.0, "swap is supposed to be lossless"

    # ---- (g) Negative test: swapping a shared (refcount>1) sequence is refused. ----
    # Free everything first so we can demo the refcount check in isolation.
    for s in list(mgr.sequences.values()):
        mgr.free_sequence(s)
    console.print("\n[bold]Negative test: swapping a shared (refcount>1) sequence is refused.[/bold]")
    parent = mgr.new_sequence()
    mgr.reserve_slots(parent, 4)
    child = mgr.fork(parent)
    try:
        mgr.swap_out(parent)
    except RuntimeError as e:
        console.print(f"  [yellow]correctly rejected: {e}[/yellow]")
    mgr.free_sequence(child)
    mgr.free_sequence(parent)

    console.print(
        "\n[bold green]Swap preemption works.[/bold green]\n"
        "[dim]Engine integration sketch:\n"
        "  - On admission failure, pick the lowest-priority running seq with\n"
        "    refcount==1 everywhere; swap it out.\n"
        "  - When other seqs retire and the pool has room, swap the victim back in\n"
        "    and put it back into the running set.\n"
        "  - If no swap victim qualifies (e.g. all running seqs are shared), fall\n"
        "    back to recompute preemption (see step 06).\n"
        "  - Production: pin CPU memory + overlap copies with kernels via streams.[/dim]"
    )


if __name__ == "__main__":
    run_demo()
