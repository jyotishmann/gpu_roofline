# src/kernels/reduce_l4.py — Level 4: first add during load (each thread reduces 2 elements)

import json
import pathlib

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .reduction import (make_reducer, assert_reduction_correct,
                        benchmark_reduction, persist_level, _INCLUDES, BLOCK)

_L4_KERNEL = r"""
#define BLOCK 256
__global__ void reduce_l4(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float v = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) v += g_in[i + blockDim.x];   // the first add happens during load — the idle half now works
    sdata[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
"""


_L4_LAUNCHER = r"""
void reduce_l4_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)((n + block * 2 - 1) / (block * 2));  // half the blocks: each reduces 2*blockDim
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold gridDim partials");
    reduce_l4<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod4 = load_inline(
    name="p01_reduce_l4",
    cpp_sources="void reduce_l4_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _L4_KERNEL + _L4_LAUNCHER,
    functions=["reduce_l4_launch"],
    with_cuda=True,
    verbose=False,
)


def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


if __name__ == "__main__":
    dev = probe_device()
    tile = 2 * BLOCK
    assert_reduction_correct(make_reducer(1_000_003, launch=_mod4.reduce_l4_launch, tile=tile))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    r = benchmark_reduction("reduce_l4",
                            make_reducer(x.numel(), launch=_mod4.reduce_l4_launch, tile=tile), x, dev)
    l3 = _level_bw(3)
    ratio = f"{r.pct_peak_bw / l3:.2f}× L3" if l3 else "n/a"
    print(f"reduce_l4  {r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% peak)  {ratio}  "
          f"← idle threads recovered; remaining gap is overhead, not waste")
    persist_level(r, level=4)
