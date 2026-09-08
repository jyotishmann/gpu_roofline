# src/kernels/sgemm_db.py — register-prefetch double-buffer

import torch
from torch.utils.cpp_extension import load_inline
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import benchmark
from .sgemm_naive import (assert_sgemm_correct, gemm_bytes_and_flops,
                           persist_level, print_gemm_report)

_BM, _BN, _BK, _WM, _WN = 128, 128, 32, 4, 4


_DB_KERNEL = r"""
#define BM 128
#define BN 128
#define BK  32
#define WM   4
#define WN   4
#define WS  32

__global__ void sgemm_db(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K, float alpha, float beta,
                           int lda, int ldb, int ldc) {
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + BM * BK;
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int tid = ty * WS + tx;
    const int ar = tid / (BK/4), ac4 = tid % (BK/4);   // A loading indices
    const int br = tid / (BN/4), bc4 = tid % (BN/4);   // B loading indices
    float C_reg[WM][WN] = {};

    // ── PROLOGUE: load first tile into smem ──────────────────────────────────
    {
        int ga = blockIdx.y*BM+ar, ka = ac4*4;
        *(float4*)(&As[ar*BK+ac4*4]) = (ga<M && ka+3<K) ?
            *(const float4*)(&A[ga*lda+ka]) : make_float4(0.f,0.f,0.f,0.f);
        int gb = br, kb = blockIdx.x*BN+bc4*4;
        *(float4*)(&Bs[br*BN+bc4*4]) = (gb<K && kb+3<N) ?
            *(const float4*)(&B[gb*ldb+kb]) : make_float4(0.f,0.f,0.f,0.f);
    }
    __syncthreads();

    // ── MAIN LOOP: prefetch next tile, compute current tile ──────────────────
    for (int k_base = 0; k_base < K - BK; k_base += BK) {
        // Issue global loads NOW — before the outer product starts (latency hiding)
        float4 a_pf, b_pf;
        { int ga = blockIdx.y*BM+ar, ka = k_base+BK+ac4*4;
          a_pf = (ga<M && ka+3<K) ? *(const float4*)(&A[ga*lda+ka])
                                   : make_float4(0.f,0.f,0.f,0.f); }
        { int gb = k_base+BK+br,    kb = blockIdx.x*BN+bc4*4;
          b_pf = (gb<K && kb+3<N)  ? *(const float4*)(&B[gb*ldb+kb])
                                   : make_float4(0.f,0.f,0.f,0.f); }
        // Outer product from smem — runs while global loads are in flight
        for (int k = 0; k < BK; ++k) {
            float a_reg[WM], b_reg[WN];
            for (int i = 0; i < WM; ++i) a_reg[i] = As[(ty+i*WS)*BK+k]; // broadcast
            for (int j = 0; j < WN; ++j) b_reg[j] = Bs[k*BN+tx+j*WS];   // conflict-free
            for (int i = 0; i < WM; ++i)
                for (int j = 0; j < WN; ++j)
                    C_reg[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads(); // (1) smem reads done; (2) prefetch registers have landed
        *(float4*)(&As[ar*BK+ac4*4]) = a_pf; // write next tile to smem
        *(float4*)(&Bs[br*BN+bc4*4]) = b_pf;
        __syncthreads(); // smem ready for next iteration's outer product
    }

    // ── EPILOGUE: compute the last tile (already in smem from the loop's final write) ──
    for (int k = 0; k < BK; ++k) {
        float a_reg[WM], b_reg[WN];
        for (int i = 0; i < WM; ++i) a_reg[i] = As[(ty+i*WS)*BK+k];
        for (int j = 0; j < WN; ++j) b_reg[j] = Bs[k*BN+tx+j*WS];
        for (int i = 0; i < WM; ++i)
            for (int j = 0; j < WN; ++j)
                C_reg[i][j] += a_reg[i] * b_reg[j];
    }
    for (int i = 0; i < WM; ++i)
        for (int j = 0; j < WN; ++j) {
            int grow = blockIdx.y*BM + ty + i*WS;
            int gcol = blockIdx.x*BN + tx + j*WS;
            if (grow < M && gcol < N)
                C[grow*ldc+gcol] = alpha * C_reg[i][j] + beta * C[grow*ldc+gcol];
        }
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_DB_LAUNCHER = r"""
// Production escalation point: replace the float4 register prefetch above with
// cuda::memcpy_async + cuda::pipeline (CUDA 11+, sm_75+) for __pipeline_memcpy_async,
// or cp.async (sm_80+) for true hardware-async copy. Same structure, different ISA.
void sgemm_db_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                      double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "sgemm: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: 2D only");
    const int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(1);
    TORCH_CHECK((int)B.size(0)==K && (int)C.size(0)==M && (int)C.size(1)==N, "shape mismatch");
    TORCH_CHECK(K%32==0 && N%4==0, "sgemm_db: K must be mult of 32 (BK), N mult of 4 (float4)");
    const int lda=(int)A.stride(0), ldb=(int)B.stride(0), ldc=(int)C.stride(0);
    const size_t smem = (128*32 + 32*128)*sizeof(float);  // 32 KB — same as p02/04
    sgemm_db<<<dim3((N+127)/128,(M+127)/128), dim3(32,32), smem>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, (float)alpha, (float)beta, lda, ldb, ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod_db = load_inline(
    name="p02_sgemm_db",
    cpp_sources="void sgemm_db_launch(torch::Tensor,torch::Tensor,torch::Tensor,double,double);",
    cuda_sources=_INCLUDES + _DB_KERNEL + _DB_LAUNCHER,
    functions=["sgemm_db_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True, verbose=False,
)


def sgemm_db(A, B, alpha=1.0, beta=0.0, C=None):
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_db.sgemm_db_launch(A.contiguous(), B.contiguous(), C, alpha, beta)
    return C


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(sgemm_db)    # all four cases: K ∈ {128,64,4096,64} — all div by 32 ✓
    N = 2048; M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    bytes_, flops, _ = gemm_bytes_and_flops(M, N, K)
    r = benchmark("sgemm_db", lambda: sgemm_db(A, B, C=C), bytes_, flops, dev, M*N, iters=20)
    print_gemm_report("sgemm_db (register-prefetch double buffer)", M, N, K, r, dev)
    print("  note: gain vs p02/04 is modest on a fully compute-bound kernel (~3–10%);")
    print("        double-buffer pays most at low occupancy or the BW/compute boundary.")
    print("  production escalation: cuda::pipeline + cp.async (sm_80+) for true async copy.")
    persist_level(r, level=5)   # for the p02/07 KPI chart
