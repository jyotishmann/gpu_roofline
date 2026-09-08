# src/kernels/sgemm_tiled.py — tiled SGEMM with shared memory

import torch
import sys
from gpu_roofline.harness.timing import benchmark
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .sgemm_naive import (assert_sgemm_correct, gemm_bytes_and_flops,
                           persist_level, print_gemm_report, _tf32_off)

_TILED_KERNEL = r"""
#define TILE 32
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K, float alpha, float beta,
                              int lda, int ldb, int ldc) {
    extern __shared__ float smem[];            // ADR-202: dynamic smem, sized 2*TILE*TILE*4 bytes at launch
    float* As = smem;                          // A_tile[TILE][TILE] — flat in row-major order
    float* Bs = smem + TILE * TILE;            // B_tile[TILE][TILE] — immediately after A_tile

    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += TILE) {
        // ── DRAM → shared memory: each thread loads one element of As and one of Bs ──
        As[ty * TILE + tx] = (row < M && k_base + tx < K) ? A[row * lda + k_base + tx] : 0.0f;
        Bs[ty * TILE + tx] = (k_base + ty < K && col < N) ? B[(k_base + ty) * ldb + col] : 0.0f;
        __syncthreads();   // fence 1: all loads visible before any thread reads a neighbour's element

        // ── shared memory → register: BK=TILE inner MACs, no global memory touched ──
        for (int k = 0; k < TILE; ++k)
            acc += As[ty * TILE + k] * Bs[k * TILE + tx];
        __syncthreads();   // fence 2: all reads done before next tile overwrites smem
    }

    if (row < M && col < N)
        C[row * ldc + col] = alpha * acc + beta * C[row * ldc + col];
}
"""


_TILED_LAUNCHER = r"""
void sgemm_tiled_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                         double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "sgemm: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: 2D tensors only");
    const int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(1);
    TORCH_CHECK((int)B.size(0) == K, "sgemm: K mismatch");
    TORCH_CHECK((int)C.size(0) == M && (int)C.size(1) == N, "sgemm: C shape mismatch");
    const int lda = (int)A.stride(0), ldb = (int)B.stride(0), ldc = (int)C.stride(0);
    const int T = 32;
    const size_t smem_bytes = 2 * T * T * sizeof(float);   // As[T][T] + Bs[T][T]
    const dim3 block(T, T);
    const dim3 grid((N + T - 1) / T, (M + T - 1) / T);
    sgemm_tiled<<<grid, block, smem_bytes>>>(              // third arg: dynamic smem size in bytes
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, (float)alpha, (float)beta, lda, ldb, ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_mod_tiled = load_inline(
    name="p02_sgemm_tiled",
    cpp_sources="void sgemm_tiled_launch(torch::Tensor,torch::Tensor,torch::Tensor,double,double);",
    cuda_sources=_INCLUDES + _TILED_KERNEL + _TILED_LAUNCHER,
    functions=["sgemm_tiled_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def sgemm_tiled(A: torch.Tensor, B: torch.Tensor,
                alpha: float = 1.0, beta: float = 0.0,
                C: torch.Tensor | None = None) -> torch.Tensor:
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_tiled.sgemm_tiled_launch(A.contiguous(), B.contiguous(), C, alpha, beta)
    return C   # same thin-wrapper pattern as p02/01; correctness imported unchanged


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(sgemm_tiled)
    N = 2048; M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    bytes_, flops, ai_algo = gemm_bytes_and_flops(M, N, K)
    r = benchmark("sgemm_tiled", lambda: sgemm_tiled(A, B, C=C), bytes_, flops, dev, M * N, iters=20)
    print_gemm_report("sgemm_tiled (TILE=32)", M, N, K, r, dev)
    print(f"  algo AI         : {ai_algo:.0f} FLOP/byte  (tile reuse brings actual traffic → algo minimum)")
    print(f"  ridge           : {dev.ridge_flop_per_byte:.1f} FLOP/byte  "
          f"({'ABOVE — compute-bound' if ai_algo > dev.ridge_flop_per_byte else 'below — BW-bound by sync/occupancy'})")
    persist_level(r, level=2)  # keyed by level for p02/07 KPI chart
