# src/kernels/triton_attn.py — FlashAttention in Triton
import torch
import triton # type: ignore
import triton.language as tl # type: ignore
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.kernels.attn_naive import (assert_attention_correct, flash_hbm_bytes,
                                 attention_flops, benchmark_attention,
                                 print_attention_report, persist_result)


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    N, HEAD_DIM: tl.constexpr,         # ← tl.constexpr = template<int HEAD_DIM>
    scale,
    BLOCK_M: tl.constexpr,             # ← query tile size  (= Bq in p03's Python)
    BLOCK_N: tl.constexpr,             # ← key/value tile size (= Bkv)
):
    # ── Grid: one program per query tile ──────────────────────────────────────
    start_m = tl.program_id(0) * BLOCK_M          # = blockIdx.x * Bq  (p03 CUDA: qi = blockIdx.x)

    # ── Pointer arithmetic for this tile's Q block ────────────────────────────
    offs_m = start_m + tl.arange(0, BLOCK_M)      # [BLOCK_M] global query indices
    offs_d = tl.arange(0, HEAD_DIM)               # [HEAD_DIM] head-dimension indices
    Q_ptrs = Q_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :]  # [BLOCK_M, HEAD_DIM]

    # ── Load Q tile into registers (persists across all KV iterations) ─────────
    # p03 CUDA: const float q = Q[qi*D + tx] * scale;   (one element per thread)
    Q = tl.load(Q_ptrs, mask=offs_m[:, None] < N, other=0.0) * scale   # [BLOCK_M, HEAD_DIM]

    # ── Online-softmax register state ─────────────────────────────────────────
    # p03 CUDA: float m = -1e9f, l = 0.f, acc = 0.f;   (scalar/register per thread)
    m   = tl.full([BLOCK_M],           float("-inf"), dtype=tl.float32)  # [BLOCK_M]
    l   = tl.zeros([BLOCK_M],                         dtype=tl.float32)  # [BLOCK_M]
    acc = tl.zeros([BLOCK_M, HEAD_DIM],               dtype=tl.float32)  # [BLOCK_M, HEAD_DIM]

    # ── KV tile loop ──────────────────────────────────────────────────────────
    for kv_start in range(0, N, BLOCK_N):
        offs_n = kv_start + tl.arange(0, BLOCK_N)   # [BLOCK_N] global KV indices
        K_ptrs = K_ptr + offs_n[:, None] * HEAD_DIM + offs_d[None, :]  # [BLOCK_N, HEAD_DIM]
        V_ptrs = V_ptr + offs_n[:, None] * HEAD_DIM + offs_d[None, :]

        # Load K and V tiles — p03 CUDA: 2-line cooperative smem load + __syncthreads()
        kv_mask = offs_n[:, None] < N
        K = tl.load(K_ptrs, mask=kv_mask, other=0.0)   # [BLOCK_N, HEAD_DIM]
        V = tl.load(V_ptrs, mask=kv_mask, other=0.0)   # [BLOCK_N, HEAD_DIM]

        # QK dot product — p03 CUDA: 10-line two-warp shfl_down_sync reduction
        S = tl.dot(Q, tl.trans(K))                      # [BLOCK_M, BLOCK_N]

        # Online softmax update — identical to Cell 2.3 Python and Cell 3.2 CUDA
        m_new = tl.maximum(m, tl.max(S, 1))             # [BLOCK_M]  ← tl.max = manual loop in CUDA
        sf    = tl.exp(m - m_new)                        # [BLOCK_M]  ← the rescaling factor
        P     = tl.exp(S - m_new[:, None])               # [BLOCK_M, BLOCK_N]
        l     = sf * l + tl.sum(P, 1)                   # [BLOCK_M]  ← tl.sum = manual loop in CUDA
        acc   = sf[:, None] * acc + tl.dot(P, V)        # [BLOCK_M, HEAD_DIM]  ← tl.dot for PV
        m     = m_new
        # No __syncthreads() — Triton inserts barriers automatically

    # ── Final normalisation and output write ──────────────────────────────────
    acc = acc / l[:, None]                               # [BLOCK_M, HEAD_DIM]
    lse = m + tl.log(l)                                  # [BLOCK_M]  logsumexp

    # Write O and LSE — p03 CUDA: O[qi*D+tx] = acc/l;  if(tx==0) LSE[qi] = m+log(l);
    out_ptrs  = O_ptr   + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    lse_ptrs  = LSE_ptr + offs_m
    out_mask  = offs_m[:, None] < N
    tl.store(out_ptrs, acc, mask=out_mask)
    tl.store(lse_ptrs, lse, mask=offs_m < N)


def flash_attn_triton_v1(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          scale: float | None = None,
                          BLOCK_M: int = 64, BLOCK_N: int = 64) -> torch.Tensor:
    """Non-causal Triton attention with fixed tile sizes."""
    N, D = Q.shape
    scale = scale or D ** -0.5
    O   = torch.zeros(N, D, device=Q.device, dtype=Q.dtype)
    LSE = torch.zeros(N,    device=Q.device, dtype=Q.dtype)
    grid = (triton.cdiv(N, BLOCK_M),)   # one program per query tile
    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, LSE, N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return O


if __name__ == "__main__":
    dev = probe_device()
    assert_attention_correct(flash_attn_triton_v1, causal=False, atol=1e-5)
    print()
    N, D = 2048, 64
    Q = torch.randn(N, D, device="cuda")
    K = torch.randn(N, D, device="cuda")
    V = torch.randn(N, D, device="cuda")
    r = benchmark_attention("triton_attn_v1 (BLOCK_M=64, BLOCK_N=64, no autotune)",
                             lambda: flash_attn_triton_v1(Q, K, V),
                             Q, K, V, N, D, dev, iters=50)
    print_attention_report(r, dev)
    persist_result(r)
    print("\nFirst call compiles the kernel (~1–5 s); subsequent calls use the cached CUBIN.")
    print("Run p04/03 (autotune) for the tuned speed.")
