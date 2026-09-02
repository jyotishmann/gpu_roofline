# src/kernels/saxpy_copy.py — SAXPY and copy kernels

import torch
from torch.utils.cpp_extension import load_inline

import json
import pathlib
from dataclasses import asdict

from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import benchmark, BenchResult
from gpu_roofline.kernels.vector_add import add as vector_add


_SAXPY_KERNEL = r"""
__global__ void saxpy_kernel(float alpha, const float* __restrict__ x,
                             float* __restrict__ y, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = alpha * x[i] + y[i];  // one FMA per element: 2 FLOP on 3*4 bytes → AI = 1/6
}
"""

_COPY_KERNEL = r"""
__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];  // zero arithmetic, 2*4 bytes/element — the purest bandwidth probe
}
"""

_LAUNCHERS = r"""
void saxpy(float alpha, torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(),  "saxpy: tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "saxpy: float32 only");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "saxpy: must be contiguous");
    TORCH_CHECK(x.numel() == y.numel(), "saxpy: size mismatch");
    TORCH_CHECK(x.data_ptr() != y.data_ptr(), "saxpy: x and y must not alias — __restrict__ requires it");
    const long long n = x.numel();
    const int block = 256;
    const int grid  = (int)((n + block - 1) / block);
    saxpy_kernel<<<grid, block>>>(alpha, x.data_ptr<float>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_buf(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(),  "copy_buf: tensors must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32, "copy_buf: float32 only");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "copy_buf: must be contiguous");
    TORCH_CHECK(src.numel() == dst.numel(), "copy_buf: size mismatch");
    const long long n = src.numel();
    const int block = 256;
    const int grid  = (int)((n + block - 1) / block);
    copy_kernel<<<grid, block>>>(src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
_CPP_DECL = """
void saxpy(float alpha, torch::Tensor x, torch::Tensor y);
void copy_buf(torch::Tensor src, torch::Tensor dst);
"""
_CUDA_SRC = _INCLUDES + _SAXPY_KERNEL + _COPY_KERNEL + _LAUNCHERS

_mod = load_inline(
    name="p00_saxpy_copy",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["saxpy", "copy_buf"],   # both kernels in one module → one compile wait per session
    with_cuda=True,
    verbose=False,
)


def saxpy(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """In-place y = alpha*x + y. Returns y (mutated)."""
    _mod.saxpy(alpha, x, y)
    return y   # returns y for chaining, but mutation has already happened


def copy_buf(src: torch.Tensor, dst: torch.Tensor | None = None) -> torch.Tensor:
    """dst = src; allocates dst if None. Returns dst."""
    if dst is None:
        dst = torch.empty_like(src)
    _mod.copy_buf(src, dst)
    return dst


def _check_correctness() -> None:
    torch.manual_seed(42)
    n, alpha = 1_000_003, 2.5   # prime N to test partial block, as in p00/02
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    y_orig = y.clone()          # must snapshot before in-place mutation

    saxpy(alpha, x, y)
    ref = alpha * x + y_orig    # unfused on CPU path → may differ by 1 ulp from GPU FMA
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-4), "saxpy disagrees with torch reference"
    print(f"[ok] saxpy allclose to alpha*x+y  (N={n}, alpha={alpha})")

    a = torch.randn(n, device="cuda")
    c = copy_buf(a)
    assert torch.equal(c, a), "copy_buf disagrees with src — pure data movement must be exact"
    print(f"[ok] copy_buf exactly equals src  (N={n})")


if __name__ == "__main__":
    _check_correctness()


_RESULTS_PATH = "benchmarks/p00_results.json"


def _upsert(res: BenchResult) -> None:
    p = pathlib.Path(_RESULTS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    data = [d for d in data if d["name"] != res.name] + [asdict(res)]
    p.write_text(json.dumps(data, indent=2))


def run_saxpy_copy_benchmarks(dev=None, iters: int = 100) -> list[BenchResult]:
    dev = dev or probe_device()
    # Working set > 8× L2 for each kernel; 3-array kernels need more elements than 2-array copy
    n3 = int(dev.l2_bytes * 8 / (3 * 4))   # 3-array working set ≈ 8× L2 (SAXPY, vector add)
    n2 = int(dev.l2_bytes * 8 / (2 * 4))   # 2-array working set ≈ 8× L2 (copy)

    # --- vector add (re-run for the side-by-side table; result already in JSON) ---
    a = torch.randn(n3, device="cuda")
    b = torch.randn(n3, device="cuda")
    oa = torch.empty_like(a)
    r_add = benchmark("vector_add", lambda: vector_add(a, b, oa), 3*n3*4, n3, dev, n3, iters=iters)

    # --- SAXPY ---
    x = torch.randn(n3, device="cuda")
    y = torch.randn(n3, device="cuda")
    r_sax = benchmark("saxpy", lambda: saxpy(2.5, x, y), 3*n3*4, 2*n3, dev, n3, iters=iters)
    # note: y is mutated each iteration; that's fine — we're timing steady-state throughput

    # --- copy ---
    src = torch.randn(n2, device="cuda")
    dst = torch.empty_like(src)
    r_cpy = benchmark("copy_buf", lambda: copy_buf(src, dst), 2*n2*4, 0, dev, n2, iters=iters)

    results = [r_cpy, r_add, r_sax]
    for r in results:
        _upsert(r)
    return results


def _print_table(results: list[BenchResult], dev) -> None:
    header = f"{'kernel':<14}│{'AI (FLOP/B)':>13} │{'eff BW (GB/s)':>15} │{'% peak BW':>11} │ note"
    print(header)
    print("─" * len(header))
    notes = {"copy_buf": "purest BW probe", "vector_add": "3 arrays, 1 FLOP",
             "saxpy":    "3 arrays, 1 FMA"}
    for r in results:
        ai = f"{r.arithmetic_intensity:.3f}"
        print(f"{r.name:<14}│{ai:>13} │{r.eff_bw_gbps:>14.1f}  │{r.pct_peak_bw:>10.1f}% │ {notes[r.name]}")
    print(f"\npeak bandwidth ceiling: {dev.peak_bw_gbps:.0f} GB/s  "
          f"(ridge at {dev.ridge_flop_per_byte:.1f} FLOP/byte — all three kernels are far left of it)")


if __name__ == "__main__":
    dev = probe_device()
    _check_correctness()                      # correctness before benchmarks — always
    results = run_saxpy_copy_benchmarks(dev)
    _print_table(results, dev)
