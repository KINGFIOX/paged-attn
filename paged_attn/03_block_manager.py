"""Step 03 — The block manager: KV pool, block tables, and copy-on-write.

The PagedAttention idea is borrowed straight from operating-system virtual
memory:

    * Physical memory = one giant pre-allocated KV "pool" tensor on the GPU,
      chopped into fixed-size **blocks** (e.g. 16 tokens per block).
    * Virtual memory  = each sequence's logical KV history.
    * Page table      = a per-sequence list of physical block IDs
                        (`block_table[seq_id][logical_block_idx] = phys_block`).

This file builds *only* the bookkeeping layer (no attention math yet):

    KVPool                — owns the physical buffer.
    BlockAllocator        — free list with reference counting.
    Sequence              — virtual view: a list of logical blocks + length.
    BlockManager          — glues them together; supports append, fork (COW),
                            and free.

Run:
    uv run python paged_attn/03_block_manager.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from rich.console import Console


# ---------------------------------------------------------------------------
# KV pool: one big tensor, treated as `num_blocks` independent slots.
# ---------------------------------------------------------------------------


@dataclass
class KVPool:
    """A single global buffer of shape `[num_blocks, 2, n_heads, block_size, head_dim]`.

    The leading axis is the physical block ID.  Axis 1 is K (0) or V (1).  We
    keep K and V together so a block looks like one cache-friendly chunk.
    """

    num_blocks: int
    n_heads: int
    block_size: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    buffer: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = torch.zeros(
            self.num_blocks, 2, self.n_heads, self.block_size, self.head_dim,
            device=self.device, dtype=self.dtype,
        )

    def k(self, block_id: int) -> torch.Tensor:
        return self.buffer[block_id, 0]  # [n_heads, block_size, head_dim]

    def v(self, block_id: int) -> torch.Tensor:
        return self.buffer[block_id, 1]

    def bytes(self) -> int:
        return self.buffer.element_size() * self.buffer.numel()


# ---------------------------------------------------------------------------
# Block allocator: a free list with refcounts (for copy-on-write).
# ---------------------------------------------------------------------------


class OutOfBlocks(RuntimeError):
    pass


@dataclass
class BlockAllocator:
    num_blocks: int
    _free: list[int] = field(init=False)
    _refcount: list[int] = field(init=False)

    def __post_init__(self) -> None:
        # LIFO free list: most-recently-freed blocks are reused first (warm in
        # GPU caches).  Production allocators usually do a FIFO or random order
        # to spread wear; LIFO is fine for learning.
        self._free = list(range(self.num_blocks - 1, -1, -1))
        self._refcount = [0] * self.num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks(f"no free blocks (capacity={self.num_blocks})")
        block_id = self._free.pop()
        self._refcount[block_id] = 1
        return block_id

    def incref(self, block_id: int) -> None:
        assert self._refcount[block_id] > 0
        self._refcount[block_id] += 1

    def free(self, block_id: int) -> None:
        assert self._refcount[block_id] > 0, f"double-free of block {block_id}"
        self._refcount[block_id] -= 1
        if self._refcount[block_id] == 0:
            self._free.append(block_id)

    def refcount(self, block_id: int) -> int:
        return self._refcount[block_id]


# ---------------------------------------------------------------------------
# Sequence: the "virtual address space" for one in-flight request.
# ---------------------------------------------------------------------------


@dataclass
class Sequence:
    seq_id: int
    block_size: int
    block_table: list[int] = field(default_factory=list)   # logical -> physical
    length: int = 0                                        # tokens currently stored

    def num_logical_blocks(self) -> int:
        return len(self.block_table)

    def slots_in_last_block(self) -> int:
        return self.length - (self.num_logical_blocks() - 1) * self.block_size \
            if self.block_table else 0

    def free_slots_in_last_block(self) -> int:
        if not self.block_table:
            return 0
        return self.block_size - self.slots_in_last_block()


# ---------------------------------------------------------------------------
# Block manager: append tokens, fork (copy-on-write), free.
# ---------------------------------------------------------------------------


class BlockManager:
    """The brains that decide which physical blocks back which sequence."""

    def __init__(self, pool: KVPool):
        self.pool = pool
        self.alloc = BlockAllocator(pool.num_blocks)
        self.sequences: dict[int, Sequence] = {}
        self._next_seq_id = 0

    # ---- lifecycle ----------------------------------------------------------

    def new_sequence(self) -> Sequence:
        seq = Sequence(seq_id=self._next_seq_id, block_size=self.pool.block_size)
        self._next_seq_id += 1
        self.sequences[seq.seq_id] = seq
        return seq

    def free_sequence(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.alloc.free(b)
        seq.block_table.clear()
        seq.length = 0
        del self.sequences[seq.seq_id]

    # ---- growth -------------------------------------------------------------

    def append_tokens(self, seq: Sequence, n: int) -> list[int]:
        """Reserve room for `n` more tokens.  Returns the physical-block-id list
        for every newly-allocated block (for tests / visualization)."""
        newly_allocated: list[int] = []
        remaining = n
        # First, fill the tail of the last block if there's room.
        if seq.block_table:
            tail_block = seq.block_table[-1]
            # Copy-on-write: if the last block is shared with another sequence,
            # we cannot write into it.  Allocate a fresh copy first.
            if self.alloc.refcount(tail_block) > 1 and seq.free_slots_in_last_block() > 0:
                new_block = self._cow(seq, len(seq.block_table) - 1)
                newly_allocated.append(new_block)
            free = seq.free_slots_in_last_block()
            take = min(free, remaining)
            seq.length += take
            remaining -= take
        # Then allocate whole new blocks as needed.
        while remaining > 0:
            block_id = self.alloc.allocate()
            seq.block_table.append(block_id)
            newly_allocated.append(block_id)
            take = min(self.pool.block_size, remaining)
            seq.length += take
            remaining -= take
        return newly_allocated

    # ---- copy-on-write fork (prefix sharing) --------------------------------

    def fork(self, parent: Sequence) -> Sequence:
        """Create a child that *shares* the parent's blocks.

        Used by beam search, n-best sampling, or any time multiple in-flight
        requests share the same prompt prefix.  No memory is copied yet — only
        when one of the participants writes into a shared block.
        """
        child = self.new_sequence()
        child.block_table = list(parent.block_table)
        child.length = parent.length
        for block_id in child.block_table:
            self.alloc.incref(block_id)
        return child

    def _cow(self, seq: Sequence, logical_idx: int) -> int:
        """Copy-on-write a single block; returns the new physical id."""
        old = seq.block_table[logical_idx]
        new = self.alloc.allocate()
        # GPU-side copy of the block contents.
        self.pool.buffer[new].copy_(self.pool.buffer[old])
        seq.block_table[logical_idx] = new
        self.alloc.free(old)  # decref the shared block
        return new

    # ---- introspection ------------------------------------------------------

    def utilization(self) -> dict:
        used_blocks = self.pool.num_blocks - self.alloc.num_free
        used_tokens = sum(s.length for s in self.sequences.values())
        block_capacity_tokens = used_blocks * self.pool.block_size
        return {
            "blocks_used":     used_blocks,
            "blocks_free":     self.alloc.num_free,
            "tokens_stored":   used_tokens,
            "block_capacity":  block_capacity_tokens,
            "block_util":      used_tokens / max(block_capacity_tokens, 1),
        }


# ---------------------------------------------------------------------------
# Demo: append, fork (prefix share), divergent writes (COW), free.
# ---------------------------------------------------------------------------


def _show(console: Console, mgr: BlockManager, label: str) -> None:
    util = mgr.utilization()
    console.print(f"\n[bold]{label}[/bold]")
    for sid, seq in mgr.sequences.items():
        rc = [mgr.alloc.refcount(b) for b in seq.block_table]
        console.print(
            f"  seq {sid:>2}: len={seq.length:>3}  "
            f"blocks={seq.block_table}  refcounts={rc}"
        )
    console.print(
        f"  pool: used={util['blocks_used']}/{mgr.pool.num_blocks} blocks, "
        f"tokens={util['tokens_stored']}/{util['block_capacity']} "
        f"({util['block_util']:.0%})"
    )


def run_demo() -> None:
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pool = KVPool(num_blocks=8, n_heads=2, block_size=4, head_dim=8,
                  dtype=torch.float16, device=device)
    mgr = BlockManager(pool)
    console.print(f"[dim]Pool: {pool.num_blocks} blocks * block_size {pool.block_size} "
                  f"= capacity {pool.num_blocks * pool.block_size} tokens "
                  f"({pool.bytes() / 1024:.1f} KiB).[/dim]")

    # 1) A prompt of 10 tokens => ceil(10/4) = 3 blocks.
    parent = mgr.new_sequence()
    mgr.append_tokens(parent, 10)
    _show(console, mgr, "After prefill (parent: 10 tokens)")

    # 2) Fork twice: two beams share the parent's blocks (refcount goes up).
    child_a = mgr.fork(parent)
    child_b = mgr.fork(parent)
    _show(console, mgr, "After two forks (prefix shared, no copies yet)")

    # 3) Each child appends 3 tokens.  The shared last block gets copy-on-write.
    mgr.append_tokens(child_a, 3)
    mgr.append_tokens(child_b, 5)
    _show(console, mgr, "After child_a += 3 tokens, child_b += 5 tokens (COW kicks in)")

    # 4) Drop child_a and the parent; only child_b's blocks should remain.
    mgr.free_sequence(parent)
    mgr.free_sequence(child_a)
    _show(console, mgr, "After freeing parent + child_a")

    # 5) Try to run out of memory on purpose, just to see the allocator complain.
    try:
        big = mgr.new_sequence()
        mgr.append_tokens(big, 99)  # way more than capacity
    except OutOfBlocks as e:
        console.print(f"\n[red]OutOfBlocks (expected): {e}[/red]")

    console.print(
        "\n[dim]Key takeaways:\n"
        "  - A sequence's KV history is a list of *physical* blocks, not a contiguous tensor.\n"
        "  - Two sequences can share a prefix as long as nobody writes; refcount = 2 means shared.\n"
        "  - On the first divergent write into a shared block, we copy it (COW) and bump pointers.\n"
        "  - Freeing a sequence just decrements refcounts; blocks return to the pool when they hit 0.[/dim]"
    )


if __name__ == "__main__":
    run_demo()
