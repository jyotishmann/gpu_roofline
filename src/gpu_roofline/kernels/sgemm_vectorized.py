# src/kernels/sgemm_vectorized.py — float4-loaded SGEMM

import torch
from torch.utils.cpp_extension import load_inline
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import benchmark
from .sgemm_naive import (assert_sgemm_correct, gemm_bytes_and_flops,
                           persist_level, print_gemm_report)

_BM, _BN, _BK, _WM, _WN = 128, 128, 32, 4, 4   # BK=32 to pair with float4 loading


_VECTORIZED_KERNEL = r"""
#define BM 128
#define BN 128
#define BK  32    // increased from 8: 1 float4 per thread per tile; 4× fewer barriers
#define WM   4
#define WN   4
#define WS  32

__global__ void sgemm_vectorized(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K, float alpha, float beta,
                                   int lda, int ldb, int ldc) {
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + BM * BK;   // As[128][32] then Bs[32][128], total 32 KB
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int tid = ty * WS + tx;
    float C_reg[WM][WN] = {};

    for (int k_base = 0; k_base < K; k_base += BK) {
        // ── float4 load A_tile[BM][BK] (1 float4 per thread) ──
        {
            int a_row = tid / (BK / 4);             // tid/8  → row in [0,BM)
            int a_col4 = tid % (BK / 4);            // tid%8  → float4 column index in [0,BK/4)
            int grow = blockIdx.y * BM + a_row;
            int gcol = k_base + a_col4 * 4;
            float4 a4 = (grow < M && gcol + 3 < K)
                ? *reinterpret_cast<const float4*>(&A[grow * lda + gcol])
                : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(&As[a_row * BK + a_col4 * 4]) = a4;  // 128-bit smem store
        }
        // ── float4 load B_tile[BK][BN] (1 float4 per thread) ──
        {
            int b_row = tid / (BN / 4);             // tid/32 → k-tile row in [0,BK)
            int b_col4 = tid % (BN / 4);            // tid%32 → float4 column index in [0,BN/4)
            int grow = k_base + b_row;
            int gcol = blockIdx.x * BN + b_col4 * 4;
            float4 b4 = (grow < K && gcol + 3 < N)
                ? *reinterpret_cast<const float4*>(&B[grow * ldb + gcol])
                : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(&Bs[b_row * BN + b_col4 * 4]) = b4;  // 128-bit smem store
        }
        __syncthreads();

        // ── Outer-product update (identical to p02/03; BK=32 → 32 k-steps) ──
        for (int k = 0; k < BK; ++k) {
            float a_reg[WM], b_reg[WN];
            for (int i = 0; i < WM; ++i) a_reg[i] = As[(ty + i*WS)*BK + k];
            for (int j = 0; j < WN; ++j) b_reg[j] = Bs[k*BN + tx + j*WS];
            for (int i = 0; i < WM; ++i)
                for (int j = 0; j < WN; ++j)
                    C_reg[i][j] += a_reg[i] * b_reg[j];   // outer product unchanged from p02/03
        }
        __syncthreads();
    }
    for (int i = 0; i < WM; ++i)
        for (int j = 0; j < WN; ++j) {
            int grow = blockIdx.y*BM + ty + i*WS;
            int gcol = blockIdx.x*BN + tx + j*WS;
            if (grow < M && gcol < N)
                C[grow*ldc + gcol] = alpha * C_reg[i][j] + beta * C[grow*ldc + gcol];
        }
}
"""


_VECTORIZED_LAUNCHER = r"""
void sgemm_vectorized_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                               double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "sgemm: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: 2D only");
    const int M = (int)A.size(0), K = (int)A.size(1), N = (int)B.size(1);
    TORCH_CHECK((int)B.size(0) == K && (int)C.size(0) == M && (int)C.size(1) == N,
                "sgemm: shape mismatch");
    TORCH_CHECK(K % 4 == 0 && N % 4 == 0,
                "sgemm_vectorized: K and N must be multiples of 4 for float4 alignment");
    const int lda = (int)A.stride(0), ldb = (int)B.stride(0), ldc = (int)C.stride(0);
    const int bm = 128, bn = 128, bk = 32;
    const size_t smem_bytes = (bm*bk + bk*bn) * sizeof(float);  // 32 KB
    const dim3 block(32, 32);
    const dim3 grid((N + bn - 1)/bn, (M + bm - 1)/bm);
    sgemm_vectorized<<<grid, block, smem_bytes>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, (float)alpha, (float)beta, lda, ldb, ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
_mod_vec = load_inline(
    name="p02_sgemm_vectorized",
    cpp_sources="void sgemm_vectorized_launch(torch::Tensor,torch::Tensor,torch::Tensor,double,double);",
    cuda_sources=_INCLUDES + _VECTORIZED_KERNEL + _VECTORIZED_LAUNCHER,
    functions=["sgemm_vectorized_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True, verbose=False,
)


def sgemm_vectorized(A, B, alpha=1.0, beta=0.0, C=None):
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_vec.sgemm_vectorized_launch(A.contiguous(), B.contiguous(), C, alpha, beta)
    return C


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(sgemm_vectorized)       # all four cases have K,N divisible by 4 ✓
    N = 2048; M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    bytes_, flops, ai = gemm_bytes_and_flops(M, N, K)
    r = benchmark("sgemm_vectorized", lambda: sgemm_vectorized(A, B, C=C),
                  bytes_, flops, dev, M*N, iters=20)
    print_gemm_report("sgemm_vectorized (BK=32, float4)", M, N, K, r, dev)
    persist_level(r, level=4)   # for the p02/07 KPI chart
