# src/kvcache/page_pool.py — paged KV cache
from collections import deque
import torch
import sys


PAGE_SIZE = 16   # tokens per page (vLLM default)


class PagePool:
    """
    Physical KV cache: two tensors [max_pages, PAGE_SIZE, n_heads, head_dim].
    Free-list tracks available physical pages (O(1) alloc + free).
    Page table maps (seq_id, logical_page) → physical_page.
    """
    def __init__(self, max_pages: int, n_heads: int, head_dim: int,
                 device: str = "cuda"):
        # Physical storage — allocated ONCE, never reallocated
        self.K = torch.zeros(max_pages, PAGE_SIZE, n_heads, head_dim, device=device)
        self.V = torch.zeros(max_pages, PAGE_SIZE, n_heads, head_dim, device=device)
        self.max_pages = max_pages
        self.n_heads   = n_heads
        self.head_dim  = head_dim
        self.device    = device

        # Free-list: all pages initially available
        self.free_list: deque[int] = deque(range(max_pages))   # O(1) alloc + free

        # Page table: seq_id → list of physical page indices (logical order)
        self.page_table: dict[int, list[int]] = {}

        # Token count per sequence (for slot computation)
        self.seq_lens: dict[int, int] = {}

    def allocate_page(self, seq_id: int) -> int:
        """Pop one physical page from the free list and append to seq's page table."""
        if not self.free_list:
            raise MemoryError("KV cache full — no free pages available")
        phys = self.free_list.popleft()                         # O(1)
        self.page_table.setdefault(seq_id, []).append(phys)
        return phys

    def free_sequence(self, seq_id: int) -> None:
        """Return all of seq_id's pages to the free list."""
        for phys in self.page_table.pop(seq_id, []):
            self.free_list.append(phys)                         # O(1) per page
        self.seq_lens.pop(seq_id, None)

    @property
    def pages_used(self) -> int:
        return self.max_pages - len(self.free_list)

    @property
    def utilisation(self) -> float:
        """Fraction of pool capacity actually storing tokens (not wasted)."""
        tokens_stored = sum(self.seq_lens.values())
        pool_capacity = self.max_pages * PAGE_SIZE
        return tokens_stored / pool_capacity if pool_capacity > 0 else 0.0


def append_token(pool: PagePool, seq_id: int,
                 k: torch.Tensor, v: torch.Tensor) -> None:
    """
    Write (k, v) for the next token of seq_id into the KV pool.
    k, v: [n_heads, head_dim]
    Allocates a new page if the current one is full.
    """
    t = pool.seq_lens.get(seq_id, 0)           # current sequence length (before this token)

    if t % PAGE_SIZE == 0:
        # Need a new page (either the very first token, or we just crossed a boundary)
        pool.allocate_page(seq_id)

    # Compute physical location: which page and which slot within it
    logical_page = t // PAGE_SIZE
    slot         = t % PAGE_SIZE
    phys_page    = pool.page_table[seq_id][logical_page]

    pool.K[phys_page, slot] = k                # write key   to pool
    pool.V[phys_page, slot] = v                # write value to pool
    pool.seq_lens[seq_id]   = t + 1            # advance sequence length


def assert_pool_correct(max_pages: int = 64, n_seqs: int = 4,
                        n_heads: int = 4, head_dim: int = 32) -> None:
    """Append variable-length sequences; verify page table and utilisation."""
    pool   = PagePool(max_pages, n_heads, head_dim)
    lens   = [10, 35, 17, 50][:n_seqs]          # deliberately straddle page boundaries
    for seq_id, length in enumerate(lens):
        for _ in range(length):
            k = torch.randn(n_heads, head_dim, device="cuda")
            v = torch.randn(n_heads, head_dim, device="cuda")
            append_token(pool, seq_id, k, v)
        assert pool.seq_lens[seq_id] == length, \
            f"seq {seq_id}: expected len {length}, got {pool.seq_lens[seq_id]}"
        n_pages = len(pool.page_table[seq_id])
        expected_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
        assert n_pages == expected_pages, \
            f"seq {seq_id}: expected {expected_pages} pages, got {n_pages}"
    used   = pool.pages_used
    tokens = sum(lens)
    util   = pool.utilisation
    print(f"[ok] page pool correct: {n_seqs} seqs, lens={lens}, "
          f"{used} pages used, utilisation={util:.2f}")
    return pool, lens


def run_pool_checks() -> None:
    """Correctness gate. Called from __main__ before the reuse demo."""
    pool, lens = assert_pool_correct()
    print(f"\nPre-allocation baseline (for comparison):")
    max_len = 64                                  # hypothetical max sequence length
    pre_alloc_util = sum(lens) / (len(lens) * max_len)
    print(f"  actual tokens: {sum(lens)}, pre-allocated: {len(lens)*max_len}")
    print(f"  pre-alloc utilisation: {pre_alloc_util:.2f}  "
          f"vs  paged utilisation: {pool.utilisation:.2f}")


def demonstrate_page_reuse() -> None:
    pool = PagePool(max_pages=8, n_heads=2, head_dim=16)
    # Fill two sequences
    for seq_id in range(2):
        for _ in range(PAGE_SIZE * 2):          # 2 pages each
            append_token(pool, seq_id,
                         torch.zeros(2, 16, device="cuda"),
                         torch.zeros(2, 16, device="cuda"))
    before_free = pool.pages_used
    pool.free_sequence(0)                        # return seq 0's pages
    after_free  = pool.pages_used
    # Allocate a new sequence that needs 3 pages
    for _ in range(PAGE_SIZE * 3):
        append_token(pool, 2,
                     torch.zeros(2, 16, device="cuda"),
                     torch.zeros(2, 16, device="cuda"))
    after_alloc = pool.pages_used
    print(f"[ok] page reuse: {before_free} pages used → "
          f"{after_free} after free → {after_alloc} after realloc")
    assert after_free < before_free, "Free didn't reduce page count"
    print("     Freed pages are immediately available to new sequences.")


if __name__ == "__main__":
    assert_pool_correct()
    run_pool_checks()
    demonstrate_page_reuse()
