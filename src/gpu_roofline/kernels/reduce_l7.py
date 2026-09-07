# src/kernels/reduce_l7.py — Level 7: grid-stride cascade (tunable elements per thread)

import json
import pathlib

import torch
from torch.utils.cpp_extension import load_inline
from gpu_roofline.harness.device import probe_device
from .reduction import assert_reduction_correct, benchmark_reduction, persist_level, _INCLUDES

_WARP_REDUCE = r"""
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}
"""

_L7_KERNEL = r"""
template <unsigned int blockSize>
__global__ void reduce_l7(const float* __restrict__ g_in, float* __restrict__ g_out, long long n) {
    __shared__ float sdata[blockSize];
    unsigned tid = threadIdx.x;
    long long gridSize = (long long)blockSize * gridDim.x;
    float sum = 0.0f;
    for (long long j = (long long)blockIdx.x * blockSize + tid; j < n; j += gridSize) sum += g_in[j];  // cascade: many elements per thread
    sdata[tid] = sum;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) {
        float w = warpReduceSum(sdata[tid] + sdata[tid + 32]);
        if (tid == 0) g_out[blockIdx.x] = w;
    }
}
"""


_L7_LAUNCHER = r"""
void reduce_l7_launch(torch::Tensor in, torch::Tensor out, long long grid_ll) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda(), "reduce: CUDA tensors required");
    TORCH_CHECK(in.scalar_type() == torch::kFloat32, "reduce: float32 only");
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous(), "reduce: contiguous required");
    const long long n = in.numel();
    const int block = 256, grid = (int)grid_ll;
    TORCH_CHECK(grid >= 1 && out.numel() == grid, "reduce: out must hold exactly `grid` partials");
    const float* a = in.data_ptr<float>(); float* o = out.data_ptr<float>();
    switch (block) {                                            // grid is now a tuning arg, not derived from n
        case 512: reduce_l7<512><<<grid, 512>>>(a, o, n); break;
        case 256: reduce_l7<256><<<grid, 256>>>(a, o, n); break;
        case 128: reduce_l7<128><<<grid, 128>>>(a, o, n); break;
        default:  TORCH_CHECK(false, "reduce_l7: unsupported block size");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


_mod7 = load_inline(
    name="p01_reduce_l7",
    cpp_sources="void reduce_l7_launch(torch::Tensor in, torch::Tensor out, long long grid_ll);",
    cuda_sources=_INCLUDES + _WARP_REDUCE + _L7_KERNEL + _L7_LAUNCHER,
    functions=["reduce_l7_launch"],
    with_cuda=True,
    verbose=False,
)


def make_cascade_reducer(n: int, grid: int, launch=_mod7.reduce_l7_launch, device: str = "cuda"):
    """Two-pass cascade: `grid` blocks → grid partials (pass 1), one block → final (pass 2)."""
    partials = torch.empty(grid, device=device, dtype=torch.float32)
    final = torch.empty(1, device=device, dtype=torch.float32)

    def reduce(x):
        launch(x.contiguous(), partials, grid)   # pass 1: chosen grid, grid-stride covers all of n
        launch(partials, final, 1)               # pass 2: single block reduces the grid partials
        return final
    return reduce


def _level_bw(level: int, path: str = "benchmarks/p01_results.json"):
    data = json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).exists() else []
    return next((d["pct_peak_bw"] for d in data if d.get("level") == level), None)


def sweep_cascade(dev, x, ks=(1, 2, 4, 8, 16, 32, 64, 128), iters=100):
    n, best, rows = x.numel(), None, []
    for k in ks:
        grid = max(1, k * dev.sm_count)
        if grid * 256 > n:                       # more threads than elements: cascade degenerate, skip
            continue
        r = benchmark_reduction(f"reduce_l7_k{k}", make_cascade_reducer(n, grid), x, dev, iters=iters)
        rows.append((k, grid, n / (grid * 256), r))
        if best is None or r.eff_bw_gbps > best[3].eff_bw_gbps:
            best = (k, grid, n / (grid * 256), r)
    return rows, best


if __name__ == "__main__":
    dev = probe_device()
    assert_reduction_correct(make_cascade_reducer(1_000_003, grid=4 * dev.sm_count))
    x = torch.randn(int(dev.l2_bytes * 8 / 4), device="cuda")
    rows, best = sweep_cascade(dev, x)
    print(f"{'k':>4} │ {'grid':>6} │ {'elems/thr':>9} │ {'% peak':>7}")
    print("─" * 36)
    for k, grid, ept, r in rows:
        print(f"{k:>4} │ {grid:>6} │ {ept:>9.0f} │ {r.pct_peak_bw:>6.1f}%")
    bk, bgrid, bept, br = best
    l6 = _level_bw(6)
    tail = f"{br.pct_peak_bw / l6:.2f}× L6" if l6 else "n/a"
    print(f"\nwinner: k={bk} (grid={bgrid}, {bept:.0f} elems/thread) → {br.pct_peak_bw:.1f}% peak  {tail}  ← top of the ladder")
    persist_level(br, level=7)  # persist the swept winner as Level 7
