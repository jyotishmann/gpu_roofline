# src/kernels/reduce_l3.py — Level 3: sequential addressing

import json
import pathlib

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .reduction import (make_reducer, assert_reduction_correct,
                        benchmark_reduction, persist_level, _INCLUDES)


_L3_KERNEL = r"""
#define BLOCK 256
__global__ void reduce_l3(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[BLOCK];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];  // sequential addressing: contiguous words → 32 distinct banks → conflict-free
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
"""


_L3_LAUNCHER = r"""
void reduce_l3_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)((n + block - 1) / block);
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold gridDim partials");
    reduce_l3<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod3 = load_inline(
    name="p01_reduce_l3",
    cpp_sources="void reduce_l3_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _L3_KERNEL + _L3_LAUNCHER,
    functions=["reduce_l3_launch"],
    with_cuda=True,
    verbose=False,
)  # same signature → make_reducer drives it unchanged





def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


if __name__ == "__main__":
    dev = probe_device()
    assert_reduction_correct(make_reducer(1_000_003, launch=_mod3.reduce_l3_launch))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    r = benchmark_reduction("reduce_l3", make_reducer(x.numel(), launch=_mod3.reduce_l3_launch), x, dev)
    l1, l2 = _level_bw(1), _level_bw(2)
    tail = (f"{r.pct_peak_bw / l2:.2f}× L2, {r.pct_peak_bw / l1:.2f}× floor" if l1 and l2 else "n/a")
    print(f"reduce_l3  {r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% peak)  {tail}  "
          f"← conflict-free; half the threads still idle after load")
    persist_level(r, level=3)
