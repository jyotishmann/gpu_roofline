# src/kvcache/paged_attn.py — paged attention
import math, torch
from gpu_roofline.kvcache.page_pool import PagePool, append_token, PAGE_SIZE


def cuda_kernel_note() -> None:
    """
    This attention uses torch.einsum (gather + matmul). Real production paged attention
    (vLLM's PagedAttention kernel) replaces this with a custom CUDA kernel that:

    1. Reads the page table from GPU memory (not Python dict).
    2. Each CUDA thread block handles one (query_head, page) pair.
    3. Within the page, K/V are contiguous → standard coalesced loads.
    4. Across pages, thread blocks run in parallel → high GPU utilisation.

    The kernel is an extension of p03/03 (flash_attn_fwd) with:
    - page_table: [seq_len // PAGE_SIZE] tensor argument
    - inner loop: for p in range(n_pages): phys = page_table[p]; load K_pool[phys]...

    The online softmax recurrence (m, l, acc) is identical.
    """
    print("[note] Custom CUDA kernel = p03/03 flash_attn_fwd + page_table lookup.")
    print("       The recurrence is unchanged; only the memory address computation differs.")


def paged_attention(q: torch.Tensor,          # [n_heads, head_dim] — current query token
                    pool: PagePool,
                    seq_id: int,
                    seq_len: int,             # number of KV tokens cached (past + current)
                    scale: float | None = None) -> torch.Tensor:
    """
    Attention for one query token over a paged KV cache.
    Implements the online softmax recurrence, page-by-page.

    Map to Python:
      outer loop   kv_start → outer loop over logical pages p
      Kj, Vj      → gathered from pool via page_table lookup  ← the one new line
      m, l, O     → same scalar/vector state variables
    """
    n_heads, head_dim = q.shape
    scale   = scale or head_dim ** -0.5
    q_scaled = q * scale                       # [n_heads, head_dim]

    m   = torch.full((n_heads,), float("-inf"), device=q.device)  # [n_heads]
    l   = torch.zeros(n_heads,               device=q.device)     # [n_heads]
    acc = torch.zeros(n_heads, head_dim,     device=q.device)     # [n_heads, head_dim]

    n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    for p in range(n_pages):
        phys_page = pool.page_table[seq_id][p]      # ← the page-table lookup (new vs p03)
        page_start = p * PAGE_SIZE
        page_end   = min(page_start + PAGE_SIZE, seq_len)
        kv_len     = page_end - page_start

        # Gather K and V for this page (contiguous within the page)
        K_page = pool.K[phys_page, :kv_len]         # [kv_len, n_heads, head_dim]
        V_page = pool.V[phys_page, :kv_len]

        # QK dot products for this page: [kv_len, n_heads]
        # q_scaled: [n_heads, head_dim], K_page: [kv_len, n_heads, head_dim]
        S_page = torch.einsum("hd,khd->kh", q_scaled, K_page)   # [kv_len, n_heads]
        S_page = S_page.T                                         # [n_heads, kv_len]

        # Online softmax update (identical to p03/02 Cell 2.3)
        m_new = torch.maximum(m, S_page.max(dim=1).values)
        sf    = torch.exp(m - m_new)                              # [n_heads]
        P     = torch.exp(S_page - m_new.unsqueeze(1))           # [n_heads, kv_len]
        l     = sf * l + P.sum(dim=1)
        # acc update: [n_heads, head_dim] += [n_heads, kv_len] @ [kv_len, head_dim]
        acc   = sf.unsqueeze(1) * acc + \
                torch.einsum("hk,khd->hd", P, V_page)
        m     = m_new

    return acc / l.unsqueeze(1)                                   # [n_heads, head_dim]


def contiguous_attention(q: torch.Tensor,       # [n_heads, head_dim]
                          K_cont: torch.Tensor, # [seq_len, n_heads, head_dim]
                          V_cont: torch.Tensor, # [seq_len, n_heads, head_dim]
                          scale: float | None = None) -> torch.Tensor:
    """Reference attention using a contiguous KV cache."""
    n_heads, head_dim = q.shape
    scale = scale or head_dim ** -0.5
    # [n_heads, 1, head_dim] × [n_heads, head_dim, seq_len] → [n_heads, seq_len]
    S = torch.einsum("hd,khd->hk", q * scale, K_cont)
    P = torch.softmax(S, dim=-1)                               # [n_heads, seq_len]
    return torch.einsum("hk,khd->hd", P, V_cont)              # [n_heads, head_dim]


def assert_paged_attn_correct(seq_len: int = 55, n_heads: int = 4,
                               head_dim: int = 32) -> None:
    """paged_attention must agree with contiguous_attention to atol=1e-5."""
    torch.manual_seed(5)
    pool = PagePool(max_pages=32, n_heads=n_heads, head_dim=head_dim)

    # Generate and append KV tokens to the pool
    K_list, V_list = [], []
    for t in range(seq_len):
        k = torch.randn(n_heads, head_dim, device="cuda")
        v = torch.randn(n_heads, head_dim, device="cuda")
        append_token(pool, seq_id=0, k=k, v=v)
        K_list.append(k); V_list.append(v)

    # Stack into contiguous tensors for the reference
    K_cont = torch.stack(K_list)                               # [seq_len, n_heads, head_dim]
    V_cont = torch.stack(V_list)

    # Generate a random query and compute both
    q = torch.randn(n_heads, head_dim, device="cuda")
    got = paged_attention(q, pool, seq_id=0, seq_len=seq_len)
    ref = contiguous_attention(q, K_cont, V_cont)

    assert torch.allclose(got, ref, atol=1e-5), \
        f"paged_attn mismatch: max err {(got-ref).abs().max():.2e}"
    print(f"[ok] paged_attention == contiguous_attention  "
          f"(seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim})")
    print(f"     Pages used: {pool.pages_used} = ceil({seq_len}/{PAGE_SIZE})")


if __name__ == "__main__":
    assert_paged_attn_correct(seq_len=16)   # exactly one page
    assert_paged_attn_correct(seq_len=17)   # crosses into second page
    assert_paged_attn_correct(seq_len=55)   # crosses four page boundaries
    assert_paged_attn_correct(seq_len=100)
    print("\nPaged attention correct for seq_len in {16, 17, 55, 100}.")
