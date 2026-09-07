# src/kernels/reduce_l6.py — Level 6: templated full unroll (compile-time tree)

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
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}
"""

_L6_KERNEL = r"""
template <unsigned int blockSize>
__global__ void reduce_l6(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[blockSize];
    unsigned tid = threadIdx.x;
    long long i = (long long)blockIdx.x * (blockSize * 2) + threadIdx.x;
    float v = (i < n) ? g_in[i] : 0.0f;
    if (i + blockSize < n) v += g_in[i + blockSize];
    sdata[tid] = v;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }  // compile-time guard prunes dead steps
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) {
        float w = warpReduceSum(sdata[tid] + sdata[tid + 32]);
        if (tid == 0) g_out[blockIdx.x] = w;
    }
}
"""


_L6_LAUNCHER = r"""
void reduce_l6_launch(torch::Tensor in, torch::Tensor out) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)((n + block * 2 - 1) / (block * 2));
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold gridDim partials");
    const float* a = in.data_ptr<float>(); float* o = out.data_ptr<float>();
    switch (block) {                                             // runtime block → compile-time template instantiation
        case 512: reduce_l6<512><<<grid, 512>>>(a, o, n); break;
        case 256: reduce_l6<256><<<grid, 256>>>(a, o, n); break;
        case 128: reduce_l6<128><<<grid, 128>>>(a, o, n); break;
        default:  TORCH_CHECK(false, "reduce_l6: unsupported block size (want 128/256/512)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod6 = load_inline(
    name="p01_reduce_l6",
    cpp_sources="void reduce_l6_launch(torch::Tensor in, torch::Tensor out);",
    cuda_sources=_INCLUDES + _WARP_REDUCE + _L6_KERNEL + _L6_LAUNCHER,
    functions=["reduce_l6_launch"],
    with_cuda=True,
    verbose=False,
)



def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


if __name__ == "__main__":
    dev = probe_device()
    tile = 2 * BLOCK
    assert_reduction_correct(make_reducer(1_000_003, launch=_mod6.reduce_l6_launch, tile=tile))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    r = benchmark_reduction("reduce_l6",
                            make_reducer(x.numel(), launch=_mod6.reduce_l6_launch, tile=tile), x, dev)
    l5 = _level_bw(5)
    ratio = f"{r.pct_peak_bw / l5:.2f}× L5" if l5 else "n/a"
    print(f"reduce_l6  {r.eff_bw_gbps:.0f} GB/s  ({r.pct_peak_bw:.1f}% peak)  {ratio}  "
          f"← loop overhead gone; only the cascade (L7) remains")
    persist_level(r, level=6)
