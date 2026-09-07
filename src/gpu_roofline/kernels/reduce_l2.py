# src/kernels/reduce_l2.py — Level 2: non-divergent interleaved addressing
import json
import pathlib

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .reduction import (make_reducer, assert_reduction_correct,
                        benchmark_reduction, persist_level, _INCLUDES)


_L2_KERNEL = r"""
#define BLOCK 256
__global__ void reduce_l2(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        unsigned index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];  // contiguous active lanes → whole warps retire, no divergence
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
"""


_L2_LAUNCHER = r"""
void reduce_l2_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)((n + block - 1) / block);
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold gridDim partials");
    reduce_l2<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod2 = load_inline(
    name="p01_reduce_l2",
    cpp_sources="void reduce_l2_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _L2_KERNEL + _L2_LAUNCHER,
    functions=["reduce_l2_launch"],
    with_cuda=True,
    verbose=False,
)  # same (in,out) signature as L1, so make_reducer drives it unchanged


def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


if __name__ == "__main__":
    dev = probe_device()
    assert_reduction_correct(make_reducer(1_000_003, launch=_mod2.reduce_l2_launch))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    r = benchmark_reduction("reduce_l2", make_reducer(x.numel(), launch=_mod2.reduce_l2_launch), x, dev)
    prev = _level_bw(1)
    ratio = f"{r.pct_peak_bw / prev:.2f}× L1" if prev else "n/a"
    print(f"reduce_l2  {r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% peak)  {ratio}  "
          f"← divergence gone; bank conflicts now dominate")
    persist_level(r, level=2)
