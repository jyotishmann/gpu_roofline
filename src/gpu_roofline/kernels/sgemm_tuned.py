# src/kernels/sgemm_tuned.py — autotuned SGEMM
import itertools
import json, pathlib
from dataclasses import asdict

import torch
from torch.utils.cpp_extension import load_inline
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import benchmark
from .sgemm_naive import (assert_sgemm_correct, gemm_bytes_and_flops,
                           persist_level, print_gemm_report)

# Configuration space: (WM, WN, BK).  BM = 32*WM, BN = 32*WN (block always 32×32).
_CANDIDATES = list(itertools.product([2, 4, 8], [2, 4, 8], [8, 16, 32]))

_TUNED_KERNEL = r"""
template<int WM, int WN, int BK>
__global__ void sgemm_tuned(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K, float alpha, float beta,
                              int lda, int ldb, int ldc) {
    constexpr int BM = WM * 32;   // derived from template: block is always 32×32
    constexpr int BN = WN * 32;
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + BM * BK;   // compile-time offset: no runtime multiply
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int tid = ty * 32 + tx;
    float C_reg[WM][WN] = {};

    for (int k_base = 0; k_base < K; k_base += BK) {
        // Cooperative scalar load (one float per thread, multiple floats for large tiles)
        const int ELEMS_A = BM * BK / 1024;   // floats per thread for A tile
        const int ELEMS_B = BK * BN / 1024;
        for (int e = 0; e < ELEMS_A; ++e) {
            int idx = tid + e * 1024;
            int row = idx / BK, col = idx % BK;
            int grow = blockIdx.y*BM + row, gcol = k_base + col;
            As[row * BK + col] = (grow < M && gcol < K) ? A[grow*lda + gcol] : 0.f;
        }
        for (int e = 0; e < ELEMS_B; ++e) {
            int idx = tid + e * 1024;
            int row = idx / BN, col = idx % BN;
            int grow = k_base + row, gcol = blockIdx.x*BN + col;
            Bs[row * BN + col] = (grow < K && gcol < N) ? B[grow*ldb + gcol] : 0.f;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {               // compiler unrolls: BK is a compile-time const
            float a_reg[WM], b_reg[WN];
            for (int i = 0; i < WM; ++i) a_reg[i] = As[(ty + i*32)*BK + k];
            for (int j = 0; j < WN; ++j) b_reg[j] = Bs[k*BN + tx + j*32];
            for (int i = 0; i < WM; ++i)
                for (int j = 0; j < WN; ++j)
                    C_reg[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < WM; ++i)
        for (int j = 0; j < WN; ++j) {
            int grow = blockIdx.y*BM + ty + i*32;
            int gcol = blockIdx.x*BN + tx + j*32;
            if (grow < M && gcol < N)
                C[grow*ldc+gcol] = alpha * C_reg[i][j] + beta * C[grow*ldc+gcol];
        }
}
"""

_DISPATCH = r"""
void sgemm_tuned_dispatch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                           double alpha, double beta, int wm, int wn, int bk) {
    TORCH_CHECK(A.is_cuda()&&B.is_cuda()&&C.is_cuda(), "cuda required");
    TORCH_CHECK(A.scalar_type()==torch::kFloat32, "float32 only");
    const int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(1);
    TORCH_CHECK((int)B.size(0)==K&&(int)C.size(0)==M&&(int)C.size(1)==N, "shape mismatch");
    const int lda=(int)A.stride(0), ldb=(int)B.stride(0), ldc=(int)C.stride(0);
    const float* a=A.data_ptr<float>(); const float* b=B.data_ptr<float>();
    float* c=C.data_ptr<float>();
    const size_t smem = (wm*32*bk + bk*wn*32)*sizeof(float);
    const dim3 block(32,32);
    const dim3 grid((N+wn*32-1)/(wn*32), (M+wm*32-1)/(wm*32));
#define LAUNCH(WM_,WN_,BK_) \
    sgemm_tuned<WM_,WN_,BK_><<<grid,block,smem>>>(a,b,c,M,N,K,(float)alpha,(float)beta,lda,ldb,ldc)
    if      (wm==2&&wn==2&&bk== 8){LAUNCH(2,2, 8);}
    else if (wm==2&&wn==2&&bk==16){LAUNCH(2,2,16);}
    else if (wm==2&&wn==2&&bk==32){LAUNCH(2,2,32);}
    else if (wm==4&&wn==4&&bk== 8){LAUNCH(4,4, 8);}
    else if (wm==4&&wn==4&&bk==16){LAUNCH(4,4,16);}
    else if (wm==4&&wn==4&&bk==32){LAUNCH(4,4,32);}
    else if (wm==8&&wn==8&&bk== 8){LAUNCH(8,8, 8);}
    else if (wm==8&&wn==8&&bk==16){LAUNCH(8,8,16);}
    else { TORCH_CHECK(false,"unsupported (wm,wn,bk): add to dispatch table"); }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def smem_bytes(wm, wn, bk) -> int:
    return (wm*32*bk + bk*wn*32) * 4   # (As + Bs) in bytes

def is_valid(wm, wn, bk, dev, K: int, smem_limit: int = 48 * 1024) -> bool:
    """Filter configs by smem budget, K divisibility, and basic register sanity."""
    if smem_bytes(wm, wn, bk) > smem_limit: return False
    if K % bk != 0:                          return False
    if wm * wn > 64:                         return False   # heuristic: >64 regs for C_reg → likely spill
    return True


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
_mod_tuned = load_inline(
    name="p02_sgemm_tuned",
    cpp_sources="void sgemm_tuned_dispatch(torch::Tensor,torch::Tensor,torch::Tensor,double,double,int,int,int);",
    cuda_sources=_INCLUDES + _TUNED_KERNEL + _DISPATCH,
    functions=["sgemm_tuned_dispatch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True, verbose=False,
)


def sgemm_tuned(A, B, wm=4, wn=4, bk=32, alpha=1.0, beta=0.0, C=None):
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_tuned.sgemm_tuned_dispatch(A.contiguous(), B.contiguous(), C, alpha, beta, wm, wn, bk)
    return C


def run_autotune_sweep(dev, N: int = 2048, iters: int = 20) -> dict:
    M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    bytes_, flops, _ = gemm_bytes_and_flops(M, N, K)
    rows, best = [], None
    for wm, wn, bk in _CANDIDATES:
        if not is_valid(wm, wn, bk, dev, K): continue
        fn = lambda wm=wm, wn=wn, bk=bk: sgemm_tuned(A, B, wm=wm, wn=wn, bk=bk, C=C)
        r  = benchmark(f"tuned_wm{wm}wn{wn}bk{bk}", fn, bytes_, flops, dev, M*N, iters=iters)
        rows.append({"wm": wm, "wn": wn, "bk": bk, "bm": wm*32, "bn": wn*32,
                     "smem_kb": smem_bytes(wm,wn,bk)//1024, **asdict(r)})
        if best is None or r.gflops > best["gflops"]: best = rows[-1]
    return rows, best


def print_sweep_table(rows, dev) -> None:
    header = f"{'config':<22}│{'BM×BN×BK':>12}│{'smem':>6}│{'GFLOP/s':>9}│{'% peak FP32':>12}"
    print(header); print("─"*len(header))
    for r in rows:
        cfg = f"WM={r['wm']} WN={r['wn']} BK={r['bk']}"
        tile = f"{r['bm']}×{r['bn']}×{r['bk']}"
        print(f"{cfg:<22}│{tile:>12}│{r['smem_kb']:>4} KB│{r['gflops']:>9.0f}│{r['pct_peak_fp32']:>11.1f}%")
    print(f"\nbest: WM={best['wm']} WN={best['wn']} BK={best['bk']} "
          f"→ {best['gflops']:.0f} GFLOP/s ({best['pct_peak_fp32']:.1f}% peak FP32)")


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(lambda A,B: sgemm_tuned(A, B, wm=4, wn=4, bk=32))
    rows, best = run_autotune_sweep(dev)
    print_sweep_table(rows, dev)
    # Persist winner as level 6 and to its own config file
    from gpu_roofline.harness.timing import BenchResult
    r_best = benchmark("sgemm_tuned_best",
                       lambda: sgemm_tuned(
                           torch.randn(2048,2048,device="cuda"),
                           torch.randn(2048,2048,device="cuda"),
                           wm=best['wm'], wn=best['wn'], bk=best['bk']),
                       *gemm_bytes_and_flops(2048,2048,2048)[:2], dev, 2048*2048, iters=20)
    persist_level(r_best, level=6)
    p = pathlib.Path("benchmarks/p02_best_config.json")
    p.write_text(json.dumps({"wm": best['wm'], "wn": best['wn'], "bk": best['bk'],
                              "device": dev.name, "pct_peak_fp32": best['pct_peak_fp32']}, indent=2))
    print(f"\nBest config written to {p}")
