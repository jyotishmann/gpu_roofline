# src/kernels/sgemm_coarsened.py — thread-coarsened SGEMM

import torch
import sys
from gpu_roofline.harness.timing import benchmark
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .sgemm_naive import (assert_sgemm_correct, gemm_bytes_and_flops,
                           persist_level, print_gemm_report)

_BM, _BN, _BK, _WM, _WN = 128, 128, 8, 4, 4  # tile/warp-tile constants exposed for the launcher


_COARSENED_KERNEL = r"""
#define BM 128
#define BN 128
#define BK   8
#define WM   4
#define WN   4
#define WS  32   // warp size == block dim in each direction == BM/WM == BN/WN

__global__ void sgemm_coarsened(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K, float alpha, float beta,
                                  int lda, int ldb, int ldc) {
    extern __shared__ float smem[];
    float* As = smem;          // A_tile[BM][BK] = 128×8 floats
    float* Bs = smem + BM*BK; // B_tile[BK][BN] = 8×128 floats
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int tid = ty * WS + tx;     // linear thread id 0..1023
    float C_reg[WM][WN] = {};         // 16 register accumulators, zero-initialised

    for (int k_base = 0; k_base < K; k_base += BK) {
        // ── Cooperative load A_tile[BM][BK] (1 element per thread) ──
        // tid → row = tid/BK, col = tid%BK inside As
        int a_row = tid / BK, a_col = tid % BK;
        As[a_row * BK + a_col] =
            (blockIdx.y*BM + a_row < M && k_base + a_col < K)
            ? A[(blockIdx.y*BM + a_row)*lda + k_base + a_col] : 0.0f;
        // ── Cooperative load B_tile[BK][BN] (1 element per thread) ──
        // tid → row = tid/BN, col = tid%BN inside Bs; consecutive tx → coalesced global read
        int b_row = tid / BN, b_col = tid % BN;
        Bs[b_row * BN + b_col] =
            (k_base + b_row < K && blockIdx.x*BN + b_col < N)
            ? B[(k_base + b_row)*ldb + blockIdx.x*BN + b_col] : 0.0f;
        __syncthreads();

        // ── Outer-product update: WM A-regs × WN B-regs → WM×WN accumulations ──
        for (int k = 0; k < BK; ++k) {
            float a_reg[WM], b_reg[WN];
            for (int i = 0; i < WM; ++i)
                a_reg[i] = As[(ty + i*WS)*BK + k];        // broadcast within warp (same ty,i,k for all tx)
            for (int j = 0; j < WN; ++j)
                b_reg[j] = Bs[k*BN + tx + j*WS];          // stride-1 tx → 32 distinct banks, no conflict
            for (int i = 0; i < WM; ++i)
                for (int j = 0; j < WN; ++j)
                    C_reg[i][j] += a_reg[i] * b_reg[j];   // 4×4 = 16 MACs from 4+4 = 8 smem reads
        }
        __syncthreads();
    }

    // ── Write output: strided rows and cols match the ownership model ──
    for (int i = 0; i < WM; ++i)
        for (int j = 0; j < WN; ++j) {
            int grow = blockIdx.y*BM + ty + i*WS;
            int gcol = blockIdx.x*BN + tx + j*WS;
            if (grow < M && gcol < N)
                C[grow*ldc + gcol] = alpha * C_reg[i][j] + beta * C[grow*ldc + gcol];
        }
}
"""


_COARSENED_LAUNCHER = r"""
void sgemm_coarsened_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                              double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "sgemm: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: 2D tensors only");
    const int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(1);
    TORCH_CHECK((int)B.size(0) == K, "sgemm: K mismatch");
    TORCH_CHECK((int)C.size(0) == M && (int)C.size(1) == N, "sgemm: C shape mismatch");
    const int lda = (int)A.stride(0), ldb = (int)B.stride(0), ldc = (int)C.stride(0);
    const int bm = 128, bn = 128, bk = 8;
    const size_t smem_bytes = (bm*bk + bk*bn) * sizeof(float);  // 8 KB — same as TILE=32
    const dim3 block(32, 32);                                    // 1024 threads; WS=32 in each dim
    const dim3 grid((N + bn - 1)/bn, (M + bm - 1)/bm);          // one block per BM×BN output tile
    sgemm_coarsened<<<grid, block, smem_bytes>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, (float)alpha, (float)beta, lda, ldb, ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_mod_crs = load_inline(
    name="p02_sgemm_coarsened",
    cpp_sources="void sgemm_coarsened_launch(torch::Tensor,torch::Tensor,torch::Tensor,double,double);",
    cuda_sources=_INCLUDES + _COARSENED_KERNEL + _COARSENED_LAUNCHER,
    functions=["sgemm_coarsened_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def sgemm_coarsened(A: torch.Tensor, B: torch.Tensor,
                    alpha: float = 1.0, beta: float = 0.0,
                    C: torch.Tensor | None = None) -> torch.Tensor:
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_crs.sgemm_coarsened_launch(A.contiguous(), B.contiguous(), C, alpha, beta)
    return C   # same thin-wrapper pattern; four-case correctness gate imported unchanged


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(sgemm_coarsened)
    N = 2048; M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    bytes_, flops, ai_algo = gemm_bytes_and_flops(M, N, K)
    r = benchmark("sgemm_coarsened", lambda: sgemm_coarsened(A, B, C=C),
                  bytes_, flops, dev, M * N, iters=20)
    print_gemm_report("sgemm_coarsened (BM=BN=128, WM=WN=4)", M, N, K, r, dev)
    print(f"  AI = BM·BN/(2·(BM+BN)) : {_BM*_BN / (2*(_BM+_BN)):.0f} FLOP/byte  "
          f"({'above' if _BM*_BN/(2*(_BM+_BN)) > dev.ridge_flop_per_byte else 'below'} "
          f"ridge {dev.ridge_flop_per_byte:.1f})")
    persist_level(r, level=3)   # keyed by level for p02/07 chart
