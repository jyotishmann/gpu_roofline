# src/kernels/reduce_l5.py — Level 5: Volta-safe warp-shuffle tail

import json
import pathlib

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .reduction import (make_reducer, assert_reduction_correct,
                        benchmark_reduction, persist_level, _INCLUDES, BLOCK)

_WARP_REDUCE = r"""
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);  // register-to-register, explicit lane mask = the sync (Volta-safe)
    return val;
}
"""


_L5_KERNEL = r"""
#define BLOCK 256
__global__ void reduce_l5(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float v = (i < n) ? g_in[i] : 0.0f;
    if (i + blockDim.x < n) v += g_in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float w = warpReduceSum(sdata[tid] + sdata[tid + 32]);  // last 64→1 in one warp, registers only, no barriers
        if (tid == 0) g_out[blockIdx.x] = w;
    }
}
"""


_L5_LAUNCHER = r"""
void reduce_l5_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)((n + block * 2 - 1) / (block * 2));
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold gridDim partials");
    reduce_l5<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod5 = load_inline(
    name="p01_reduce_l5",
    cpp_sources="void reduce_l5_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _WARP_REDUCE + _L5_KERNEL + _L5_LAUNCHER,  # helper + kernel + launcher in one TU
    functions=["reduce_l5_launch"],
    with_cuda=True,
    verbose=False,
)


def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


if __name__ == "__main__":
    dev = probe_device()
    tile = 2 * BLOCK
    assert_reduction_correct(make_reducer(1_000_003, launch=_mod5.reduce_l5_launch, tile=tile))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    r = benchmark_reduction("reduce_l5",
                            make_reducer(x.numel(), launch=_mod5.reduce_l5_launch, tile=tile), x, dev)
    l4 = _level_bw(4)
    ratio = f"{r.pct_peak_bw / l4:.2f}× L4" if l4 else "n/a"
    print(f"reduce_l5  {r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% peak)  {ratio}  "
          f"← Volta-safe warp tail; correctness gate is the real proof here")
    persist_level(r, level=5)
