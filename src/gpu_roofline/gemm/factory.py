# src/gemm/factory.py — kernel factory and autotune implementation
import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.gemm.api import DeviceSpec, KernelConfig

import statistics, sys, json, pathlib, time
from gpu_roofline.gemm.perf_model import predict_configs
from gpu_roofline.gemm.tiles import enumerate_candidate_configs

_KERNEL_TEMPLATE = r"""
template<int WM, int WN, int BK>
__global__ void gemm_lib_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K,
                                  float alpha, float beta, int lda, int ldb, int ldc) {{
    constexpr int BM = WM * 32, BN = WN * 32;
    extern __shared__ float smem[];
    float* As = smem; float* Bs = smem + BM * BK;
    const int ty = threadIdx.y, tx = threadIdx.x, tid = ty*32+tx;
    float C_reg[WM][WN] = {{}};
    for (int k_base = 0; k_base < K; k_base += BK) {{
        const int ar=tid/(BK/4), ac4=tid%(BK/4), br=tid/(BN/4), bc4=tid%(BN/4);
        int ga=blockIdx.y*BM+ar, ka=k_base+ac4*4;
        *(float4*)(&As[ar*BK+ac4*4]) = (ga<M && ka+3<K)?*(const float4*)(&A[ga*lda+ka]):make_float4(0,0,0,0);
        int gb=k_base+br, kb=blockIdx.x*BN+bc4*4;
        *(float4*)(&Bs[br*BN+bc4*4]) = (gb<K && kb+3<N)?*(const float4*)(&B[gb*ldb+kb]):make_float4(0,0,0,0);
        __syncthreads();
        for (int k=0;k<BK;++k) {{
            float a[WM],b[WN];
            for(int i=0;i<WM;++i) a[i]=As[(ty+i*32)*BK+k];
            for(int j=0;j<WN;++j) b[j]=Bs[k*BN+tx+j*32];
            for(int i=0;i<WM;++i) for(int j=0;j<WN;++j) C_reg[i][j]+=a[i]*b[j];
        }}
        __syncthreads();
    }}
    for(int i=0;i<WM;++i) for(int j=0;j<WN;++j) {{
        int gr=blockIdx.y*BM+ty+i*32, gc=blockIdx.x*BN+tx+j*32;
        if(gr<M && gc<N) C[gr*ldc+gc]=alpha*C_reg[i][j]+beta*C[gr*ldc+gc];
    }}
}}
"""

_compiled_cache: dict[KernelConfig, object] = {}


def compile_config(cfg: KernelConfig) -> object:
    """Compile a kernel for the given config; cache to avoid recompilation."""
    if cfg in _compiled_cache:
        return _compiled_cache[cfg]
    dispatch = f"""
void gemm_lib_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                      double alpha, double beta, long long bm, long long bn, long long bk,
                      long long wm, long long wn) {{
    const int M=(int)A.size(0),K=(int)A.size(1),N=(int)B.size(1);
    const int lda=(int)A.stride(0),ldb=(int)B.stride(0),ldc=(int)C.stride(0);
    const size_t smem=({cfg.BM}*{cfg.BK}+{cfg.BK}*{cfg.BN})*sizeof(float);
    dim3 block(32,32); dim3 grid((N+{cfg.BN}-1)/{cfg.BN},(M+{cfg.BM}-1)/{cfg.BM});
    gemm_lib_kernel<{cfg.WM},{cfg.WN},{cfg.BK}><<<grid,block,smem>>>(
        A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),
        M,N,K,(float)alpha,(float)beta,lda,ldb,ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}"""
    mod = load_inline(
        name=f"gemm_lib_{cfg.BM}x{cfg.BN}x{cfg.BK}_wm{cfg.WM}wn{cfg.WN}",
        cpp_sources="void gemm_lib_launch(torch::Tensor,torch::Tensor,torch::Tensor,"
                    "double,double,long long,long long,long long,long long,long long);",
        cuda_sources="#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
                     + _KERNEL_TEMPLATE + dispatch,
        functions=["gemm_lib_launch"],
        extra_cuda_cflags=["-O3","--use_fast_math"], with_cuda=True, verbose=False,
    )
    _compiled_cache[cfg] = mod
    return mod


_autotune_cache: dict[tuple, KernelConfig] = {}


def autotune(dev: DeviceSpec, M: int, N: int, K: int,
             warmup: int = 10, iters: int = 50) -> KernelConfig:
    """Implements the p08/01 stub: model → benchmark top-k → cache winner."""
    key = (dev.name, M, N, K)
    if key in _autotune_cache:
        return _autotune_cache[key]
    candidates = predict_configs(dev, M, N, K, top_k=5)
    A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda")
    best_gf, best_cfg = -1.0, candidates[0]
    for cfg in candidates:
        mod = compile_config(cfg)
        C   = torch.zeros(M, N, device="cuda")
        for _ in range(warmup):
            mod.gemm_lib_launch(A.contiguous(), B.contiguous(), C, 1.0, 0.0,
                                 cfg.BM, cfg.BN, cfg.BK, cfg.WM, cfg.WN)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            mod.gemm_lib_launch(A.contiguous(), B.contiguous(), C, 1.0, 0.0,
                                 cfg.BM, cfg.BN, cfg.BK, cfg.WM, cfg.WN)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        gf = 2*M*N*K / (statistics.median(ts) * 1e9)
        if gf > best_gf:
            best_gf, best_cfg = gf, cfg
    _autotune_cache[key] = best_cfg
    print(f"[autotuned] {dev.name} M={M} N={N} K={K} → {best_cfg}  {best_gf:.0f} GFLOP/s")
    return best_cfg


def run(A: torch.Tensor, B: torch.Tensor,
        config: KernelConfig | None = None,
        dev: DeviceSpec | None = None) -> torch.Tensor:
    """Implements the p08/01 stub: autotune-on-demand or use provided config."""
    if not A.is_cuda or A.dtype != torch.float32:
        raise RuntimeError("A must be CUDA float32")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"K mismatch: A.shape={A.shape}, B.shape={B.shape}")
    M, K = A.shape; N = B.shape[1]
    dev = dev or DeviceSpec.from_probe()
    cfg = config or autotune(dev, M, N, K)
    mod = compile_config(cfg)
    C   = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    mod.gemm_lib_launch(A.contiguous(), B.contiguous(), C, 1.0, 0.0,
                         cfg.BM, cfg.BN, cfg.BK, cfg.WM, cfg.WN)
    return C


if __name__ == "__main__":
    import torch
    from gpu_roofline.harness.device import probe_device as p00_probe
    from gpu_roofline.gemm.api import DeviceSpec
    dev = DeviceSpec.from_probe()
    # Correctness: run must agree with torch.mm
    torch.manual_seed(0)
    A = torch.randn(512, 512, device="cuda"); B = torch.randn(512, 512, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = torch.mm(A, B)
    got = run(A, B, dev=dev)
    assert torch.allclose(got, ref, atol=1e-3), \
        f"gemm_lib.run disagrees with torch.mm: max err {(got-ref).abs().max():.2e}"
    print("[ok] gemm_lib.run correct vs torch.mm  (TF32 off)")
    # Demo: run a larger problem (autotune fires on first call)
    C = run(torch.randn(2048,2048,device="cuda"), torch.randn(2048,2048,device="cuda"), dev=dev)
    print(f"[ok] gemm_lib.run({2048}x{2048}) returned shape {tuple(C.shape)}")
