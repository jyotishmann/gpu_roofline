# src/kernels/sgemm_naive.py — naïve SGEMM
import contextlib
import json
import pathlib
from dataclasses import asdict

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device

import sys
from gpu_roofline.harness.timing import benchmark


@contextlib.contextmanager
def _tf32_off():
    """Disable TF32 for the duration of the block, restoring the original state unconditionally."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 off for honest fp32 reference
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def assert_sgemm_correct(gemm_fn, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    """Correctness gate reused across all SGEMM levels: allclose vs torch.mm with TF32 off."""
    torch.manual_seed(0)
    cases = [
        (128, 128, 128, "square 128"),
        (256, 128, 64,  "non-square 256×128×64"),
        (1,   1,   4096, "K-accumulation stress"),
        (64,  64,  64,  "prime-adjacent — partial tile exercise"),
    ]
    with _tf32_off():
        for M, N, K, label in cases:
            A = torch.randn(M, K, device="cuda")
            B = torch.randn(K, N, device="cuda")
            got = gemm_fn(A, B)
            ref = torch.mm(A, B)                   # torch.mm with TF32 off = honest fp32 reference
            ok  = torch.allclose(got, ref, atol=atol, rtol=rtol)
            if not ok:
                max_err = (got - ref).abs().max().item()
                raise AssertionError(f"sgemm wrong on '{label}': max error {max_err:.2e} "
                                     f"(atol={atol}, rtol={rtol})")
            print(f"[ok] {label}  (M={M}, N={N}, K={K})")


_NAIVE_KERNEL = r"""
#define BX 32
#define BY 32
__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K, float alpha, float beta,
                             int lda, int ldb, int ldc) {
    int row = (int)blockIdx.y * BY + (int)threadIdx.y;
    int col = (int)blockIdx.x * BX + (int)threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * lda + k] * B[k * ldb + col];   // B[k,col]: coalesced within warp (col varies)
    C[row * ldc + col] = alpha * acc + beta * C[row * ldc + col];  // no shared memory → 8MNK DRAM bytes
}
"""


_NAIVE_LAUNCHER = r"""
void sgemm_naive_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                         double alpha, double beta) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "sgemm: CUDA tensors required");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "sgemm: float32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "sgemm: 2D tensors only");
    const int M = (int)A.size(0), K = (int)A.size(1);
    const int N = (int)B.size(1);
    TORCH_CHECK(B.size(0) == K, "sgemm: A columns must match B rows");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "sgemm: C shape must be (M,N)");
    // Leading dimensions from strides — handles both contiguous and .T views correctly
    const int lda = (int)(A.stride(0));  // elements between consecutive rows of A
    const int ldb = (int)(B.stride(0));  // elements between consecutive rows of B
    const int ldc = (int)(C.stride(0));  // elements between consecutive rows of C
    const int BX = 32, BY = 32;
    const dim3 block(BX, BY);
    const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);  // x→cols, y→rows: must match kernel indexing
    sgemm_naive<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, (float)alpha, (float)beta, lda, ldb, ldc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_mod_naive = load_inline(
    name="p02_sgemm_naive",
    cpp_sources="void sgemm_naive_launch(torch::Tensor,torch::Tensor,torch::Tensor,double,double);",
    cuda_sources=_INCLUDES + _NAIVE_KERNEL + _NAIVE_LAUNCHER,
    functions=["sgemm_naive_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],   # fused MACs + fp reassociation for fair perf comparison
    with_cuda=True,
    verbose=False,
)


def sgemm(A: torch.Tensor, B: torch.Tensor,
          alpha: float = 1.0, beta: float = 0.0,
          C: torch.Tensor | None = None) -> torch.Tensor:
    """Frontend: allocate C if needed, launch the kernel, return C."""
    if C is None:
        C = torch.zeros(A.shape[0], B.shape[1], device=A.device, dtype=A.dtype)
    _mod_naive.sgemm_naive_launch(A.contiguous(), B.contiguous(), C, alpha, beta)
    return C


def gemm_bytes_and_flops(M: int, N: int, K: int) -> tuple[int, int, float]:
    """Algorithmic minimum bytes, total FLOPs, and arithmetic intensity for an MxNxK GEMM."""
    bytes_ = 4 * (M * K + K * N + M * N)   # read A, read B, write C — once each
    flops  = 2 * M * N * K                  # one multiply-add per (m,n,k) triple
    ai     = flops / bytes_
    return bytes_, flops, ai


def print_gemm_report(name: str, M: int, N: int, K: int, r, dev) -> None:
    """Print both KPIs so the p02/01→p02/02 metric shift is visible on the terminal."""
    _, _, ai = gemm_bytes_and_flops(M, N, K)
    print(f"{name}  ({M}×{N}×{K})")
    print(f"  time (median)   : {r.median_ms:.2f} ms")
    print(f"  GFLOP/s         : {r.gflops:.1f}   ({r.pct_peak_fp32:.1f}% of peak FP32)  ← headline from p02/02")
    print(f"  eff BW          : {r.eff_bw_gbps:.1f} GB/s  ({r.pct_peak_bw:.1f}% of peak BW)")
    print(f"  algo AI         : {ai:.1f} FLOP/byte  (actual AI = 0.25 for naïve — AI = algo only if BW = algo bytes)")


def persist_level(res, level: int, path: str = "benchmarks/p02_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    rec  = {**asdict(res), "level": level}
    data = [d for d in data if d.get("level") != level] + [rec]
    p.write_text(json.dumps(sorted(data, key=lambda d: d["level"]), indent=2))


if __name__ == "__main__":
    dev = probe_device()
    assert_sgemm_correct(sgemm)
    N = 2048                                             # representative square; fits on T4 (3 × 2048² × 4 ≈ 48 MB)
    M = K = N
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")                # pre-allocate so allocation stays out of timed region
    bytes_, flops, ai = gemm_bytes_and_flops(M, N, K)
    r = benchmark("sgemm_naive", lambda: sgemm(A, B, C=C), bytes_, flops, dev, M * N, iters=20)
    print_gemm_report("sgemm_naive", M, N, K, r, dev)
    excess = (8 * M * N * K * 4) / bytes_               # how many × more DRAM traffic than algorithmic min
    print(f"  actual traffic  : ~{excess:.0f}× the algorithmic minimum "
          f"(naïve fetches 8MNK = {8*M*N*K*4/1e9:.1f} GB; algo min = {bytes_/1e6:.0f} MB)")
    persist_level(r, level=1)                            # level=1 for the p02/07 KPI comparison
