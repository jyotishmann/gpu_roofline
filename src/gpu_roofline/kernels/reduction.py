# src/kernels/reduction.py — parallel reduction, built level-by-level
import torch

import json
import pathlib
from dataclasses import asdict

import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import benchmark

from torch.utils.cpp_extension import load_inline


def reference_sum(x: torch.Tensor) -> float:
    """Trustworthy reference: sum in float64 so the reference's own rounding is negligible."""
    return x.double().sum().item()  # fp32 torch.sum is just one ordering — float64 is the honest yardstick


def assert_reduction_correct(reduce_fn, n: int = 1_000_003, rtol: float = 1e-3) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda")               # mean-0, O(1): well-conditioned so tolerance stays meaningful
    got = reduce_fn(x).item()
    ref = reference_sum(x)
    assert abs(got - ref) <= rtol * max(1.0, abs(ref)), f"reduction wrong: {got} vs {ref} (rtol={rtol})"
    print(f"[ok] reduction matches float64 reference (N={n}): {got:.5f} vs {ref:.5f}")


_L1_KERNEL = r"""
#define BLOCK 256
__global__ void reduce_l1(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];  // interleaved + DIVERGENT branch: the baseline pathology
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
"""


_LAUNCHER = r"""
void reduce_l1_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256;
    const int grid  = (int)((n + block - 1) / block);
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold exactly gridDim partial sums");
    reduce_l1<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);  // one partial per block
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
_mod = load_inline(
    name="p01_reduce_l1",
    cpp_sources="void reduce_l1_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _L1_KERNEL + _LAUNCHER,
    functions=["reduce_l1_launch"],
    with_cuda=True,
    verbose=False,
)

BLOCK = 256


"""def make_reducer(n: int, device: str = "cuda", launch=_mod.reduce_l1_launch):
    '''Return a closure that reduces an n-element tensor via pre-allocated ping-pong buffers.'''
    g1 = (n + BLOCK - 1) // BLOCK
    buf_a = torch.empty(g1, device=device, dtype=torch.float32)
    buf_b = torch.empty((g1 + BLOCK - 1) // BLOCK, device=device, dtype=torch.float32)

    def reduce(x: torch.Tensor) -> torch.Tensor:
        src, a, b = x.contiguous(), buf_a, buf_b
        while src.numel() > 1:
            grid = (src.numel() + BLOCK - 1) // BLOCK
            dst = a[:grid]                       # prefix slice stays contiguous; sizes only shrink, so 2 buffers suffice
            launch(src, dst)
            src, a, b = dst, b, a                # ping-pong
        return src
    return reduce
"""


# make_reducer with this tile-aware version (default tile=BLOCK)
def make_reducer(n, launch=_mod.reduce_l1_launch, tile: int = BLOCK, device: str = "cuda"):
    """Two-pass reducer; `tile` = elements one block reduces (BLOCK for L1–3, 2*BLOCK for L4+)."""
    g1 = (n + tile - 1) // tile
    buf_a = torch.empty(g1, device=device, dtype=torch.float32)
    buf_b = torch.empty((g1 + tile - 1) // tile, device=device, dtype=torch.float32)

    def reduce(x):
        src, a, b = x.contiguous(), buf_a, buf_b
        while src.numel() > 1:
            grid = (src.numel() + tile - 1) // tile
            dst = a[:grid]
            launch(src, dst)
            src, a, b = dst, b, a
        return src
    return reduce  # tile defaults to BLOCK, so L1–3 callers are unaffected


def benchmark_reduction(name: str, reduce_fn, x: torch.Tensor, dev, iters: int = 100):
    n = x.numel()
    return benchmark(name, lambda: reduce_fn(x), 4 * n, n, dev, n, iters=iters)


def persist_level(res, level: int, path: str = "benchmarks/p01_results.json") -> None:
    """Upsert one level's result, keyed by level, for the p01/09 climb chart."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    rec = {**asdict(res), "level": level}
    data = [d for d in data if d.get("level") != level] + [rec]  # upsert by level number
    p.write_text(json.dumps(sorted(data, key=lambda d: d["level"]), indent=2))


if __name__ == "__main__":
    dev = probe_device()
    reduce = make_reducer(1_000_003)
    assert_reduction_correct(reduce)
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")   # working set ≫ L2 → measure HBM, not cache
    r = benchmark_reduction("reduce_l1", make_reducer(x.numel()), x, dev)
    print(f"reduce_l1  N={x.numel():,}  {r.median_ms:.3f} ms  "
          f"{r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% of peak)  ← the floor")
    persist_level(r, level=1)
