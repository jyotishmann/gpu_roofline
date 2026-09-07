# src/kernels/reduce_shuffle.py — full warp-shuffle reduction + library-parity benchmark

import json
import pathlib
from dataclasses import asdict

import torch
from torch.utils.cpp_extension import load_inline

from gpu_roofline.harness.device import probe_device
from gpu_roofline.kernels.reduction import (
    make_cascade_reducer,
    assert_reduction_correct,
    benchmark_reduction,
    persist_level,
    _INCLUDES,
    BLOCK,
)


_WARP_REDUCE = r"""
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}
"""

_SHUFFLE_KERNEL = r"""
#define WARP_SIZE 32
__global__ void reduce_shuffle(const float* __restrict__ g_in,
                                float* __restrict__ g_out, long long n) {
    __shared__ float warp_sums[WARP_SIZE];   // 8 slots used; 32 allocated for warp-safe access
    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    long long gridSize = (long long)blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridSize)
        sum += g_in[j];                      // grid-stride accumulate: many elements per thread
    sum = warpReduceSum(sum);                // each warp → its lane-0 holds the warp sum
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();                         // the ONE barrier: makes 8 warp sums visible block-wide
    sum = (lane < blockDim.x / WARP_SIZE) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) sum = warpReduceSum(sum);   # warp 0 reduces the 8 warp sums → block total
    if (threadIdx.x == 0) g_out[blockIdx.x] = sum;
}
"""

_SHUFFLE_LAUNCHER = r"""
void reduce_shuffle_launch(torch::Tensor in, torch::Tensor out, long long grid_ll) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)grid_ll;
    TORCH_CHECK(out.numel() == grid, "reduce: out must hold exactly `grid` partials");
    reduce_shuffle<<<grid, block>>>(in.data_ptr<float>(), out.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod_shuf = load_inline(
    name="p01_reduce_shuffle",
    cpp_sources="void reduce_shuffle_launch(torch::Tensor in, torch::Tensor out, long long grid);",
    cuda_sources=_INCLUDES + _WARP_REDUCE + _SHUFFLE_KERNEL + _SHUFFLE_LAUNCHER,
    functions=["reduce_shuffle_launch"],
    with_cuda=True,
    verbose=False,
)

# Cooperative-groups drop-in — conceptual note, not compiled separately
_COOP_GROUPS_SNIPPET = r"""
// Production-quality spelling of the warp reduce using cooperative groups (CUDA 9+).
// Compiles to the same SASS as the hand-written __shfl_down_sync version above,
// but tile size and operator are type-checked at compile time and the mask is
// correct-by-construction (derived from active lanes, not hardcoded 0xffffffff).
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Replace the warpReduceSum call and the warp_sums write with:
auto block = cg::this_thread_block();
auto warp  = cg::tiled_partition<32>(block);   // compile-time tile; adapts if warpSize changes
sum = cg::reduce(warp, sum, cg::plus<float>{}); // mask derived from active lanes, not 0xffffffff
if (warp.thread_rank() == 0) warp_sums[warp.meta_group_rank()] = sum;
block.sync();                                   // one barrier, in terms of the group
// ... same from here
"""


_CUB_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>

long long cub_probe_bytes(long long n) {
    void*  d_temp = nullptr;
    size_t bytes  = 0;
    float* dummy  = nullptr;
    cub::DeviceReduce::Sum(d_temp, bytes, dummy, dummy, (int)n);  // null-temp probe: returns required scratch size
    return (long long)bytes;
}

void cub_sum(torch::Tensor in, torch::Tensor out, torch::Tensor temp) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && temp.is_cuda(), "cub_sum: CUDA tensors required");
    const long long n = in.numel();
    void*  d_temp     = temp.data_ptr<uint8_t>();
    size_t temp_bytes = (size_t)temp.numel();
    cub::DeviceReduce::Sum(d_temp, temp_bytes,
                           in.data_ptr<float>(), out.data_ptr<float>(), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod_cub = load_inline(
    name="p01_cub_reduce",
    cpp_sources="long long cub_probe_bytes(long long n); "
                "void cub_sum(torch::Tensor,torch::Tensor,torch::Tensor);",
    cuda_sources=_CUB_SRC,
    functions=["cub_probe_bytes", "cub_sum"],
    with_cuda=True,
    verbose=False,
)


def make_cub_reducer(n: int, device: str = "cuda"):
    """Pre-allocate CUB scratch (probe once, never inside the timed region)."""
    temp_bytes = _mod_cub.cub_probe_bytes(n)
    out  = torch.empty(1, device=device, dtype=torch.float32)
    temp = torch.empty(int(temp_bytes) + 1, device=device, dtype=torch.uint8)  # +1: avoids zero-alloc edge

    def reduce(x: torch.Tensor) -> torch.Tensor:
        _mod_cub.cub_sum(x.contiguous(), out, temp)
        return out
    return reduce


def _torch_reducer(device: str = "cuda"):
    """Thin closure around torch.sum for apples-to-apples benchmarking."""
    out = torch.empty(1, device=device)

    def reduce(x: torch.Tensor) -> torch.Tensor:
        out[0] = torch.sum(x)   # torch.sum is the library ground truth
        return out
    return reduce


def _make_shuffle_reducer(n: int, dev, device: str = "cuda"):
    """Shuffle-based two-pass cascade at the sweep-winner grid."""
    grid     = max(1, 4 * dev.sm_count)   # same grid heuristic as Level-7 winner area
    partials = torch.empty(grid, device=device, dtype=torch.float32)
    final    = torch.empty(1,    device=device, dtype=torch.float32)

    def reduce(x: torch.Tensor) -> torch.Tensor:
        _mod_shuf.reduce_shuffle_launch(x.contiguous(), partials, grid)
        _mod_shuf.reduce_shuffle_launch(partials, final, 1)
        return final
    return reduce


def run_comparison(dev, n: int | None = None, iters: int = 100):
    n = n or int(dev.l2_bytes * 8 / 4)
    x = torch.randn(n, device="cuda")
    candidates = {
        "shuffle_kernel":    _make_shuffle_reducer(n, dev),
        "level7_cascade":    make_cascade_reducer(n, grid=max(1, 4 * dev.sm_count)),
        "cub_device_reduce": make_cub_reducer(n),
        "torch_sum":         _torch_reducer(),
    }
    results = {}
    for name, fn in candidates.items():
        results[name] = benchmark_reduction(name, fn, x, dev, iters=iters)
    return n, results


def write_comparison_table(n: int, results: dict, dev,
                           out: str = "benchmarks/p01_comparison.md") -> str:
    lines = [
        "# p01 — Library-parity comparison\n",
        f"**Device:** {dev.name}  **N:** {n:,}  **Peak BW:** {dev.peak_bw_gbps:.0f} GB/s\n",
        "| kernel | eff BW (GB/s) | % peak BW | median (ms) |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r.eff_bw_gbps:.1f} | {r.pct_peak_bw:.1f}% | {r.median_ms:.3f} |"
        )
    best  = max(results.values(), key=lambda r: r.eff_bw_gbps)
    worst = min(results.values(), key=lambda r: r.eff_bw_gbps)
    spread = (best.eff_bw_gbps - worst.eff_bw_gbps) / best.eff_bw_gbps * 100
    lines += [
        f"\n**Spread:** {spread:.1f}% (best to worst). "
        f"A spread ≤ 5% means all candidates are in the same performance class.\n",
        "_Generated by `kernels/reduce_shuffle.py`. Measurement not presentation._",
    ]
    pathlib.Path(out).write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    dev = probe_device()
    # Correctness first: both passes of the shuffle reducer must match the float64 reference
    assert_reduction_correct(_make_shuffle_reducer(1_000_003, dev))
    n, results = run_comparison(dev)
    for name, r in results.items():
        print(f"{name:<24} {r.eff_bw_gbps:6.1f} GB/s  "
              f"({r.pct_peak_bw:.1f}% peak)  {r.median_ms:.3f} ms")
    tbl = write_comparison_table(n, results, dev)
    print(f"\nComparison table written to {tbl}")
    persist_level(results["shuffle_kernel"], level=8)
