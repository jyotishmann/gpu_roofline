# src/harness/block_sweep.py — occupancy-driven block-size sweep

import torch
from torch.utils.cpp_extension import load_inline
from harness.timing import benchmark, BenchResult

import json
import pathlib
from dataclasses import asdict

from functools import partial

from harness.device import probe_device


def theoretical_occupancy(block: int, blocks_per_sm: int, dev) -> float:
    """Active-warps / max-warps, given how many blocks the hardware places on an SM."""
    return blocks_per_sm * block / dev.max_threads_sm  # (blocks·threads)/SM ÷ max threads/SM


_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

void vector_add_tuned(torch::Tensor a, torch::Tensor b, torch::Tensor c,
                      long long block, long long grid) {
    TORCH_CHECK(block > 0 && block % 32 == 0, "block must be a positive multiple of the warp size");
    TORCH_CHECK(grid  > 0, "grid must be positive");
    const long long n = a.numel();
    add_kernel<<<(int)grid, (int)block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                          c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int max_active_blocks(long long block) {
    int nb;                                                   // hardware's true blocks/SM for this kernel
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, add_kernel, (int)block, 0);
    return nb;
}

int suggested_block() {
    int min_grid, block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, add_kernel, 0, 0);
    return block;
}
"""

_mod = load_inline(
    name="p00_block_sweep",
    cpp_sources=("void vector_add_tuned(torch::Tensor,torch::Tensor,torch::Tensor,long long,long long);"
                 "int max_active_blocks(long long); int suggested_block();"),
    cuda_sources=_SRC,
    functions=["vector_add_tuned", "max_active_blocks", "suggested_block"],
    with_cuda=True,
    verbose=False,
)


def occupancy_grid(block: int, dev) -> tuple[int, int, float]:
    """Grid that fills every SM to its occupancy limit once; grid-stride covers all N."""
    blocks_per_sm = _mod.max_active_blocks(block)
    grid = max(1, blocks_per_sm * dev.sm_count)   # one occupancy-filling wave, size-independent of N
    return grid, blocks_per_sm, theoretical_occupancy(block, blocks_per_sm, dev)



def sweep_block_sizes(dev, n: int | None = None,
                      blocks=(32, 64, 128, 256, 512, 1024), iters: int = 100):
    """Benchmark vector-add across block sizes at occupancy-filled grids."""
    n = n or int(dev.l2_bytes * 8 / (3 * 4))
    a = torch.randn(n, device="cuda") 
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    rows = []
    for blk in blocks:
        if blk > dev.max_threads_block:
            continue
        grid, bps, occ = occupancy_grid(blk, dev)
        fn = partial(_mod.vector_add_tuned, a, b, c, blk, grid)  # default-arg capture avoids late-binding
        res = benchmark(f"add_blk{blk}", fn, 3 * n * 4, n, dev, n, iters=iters)
        rows.append({"block": blk, "blocks_per_sm": bps, "occupancy": occ, "res": res})
    return n, rows


def print_sweep_table(n: int, rows: list[dict], dev) -> dict:
    """Print the sweep, return the winning row (max measured bandwidth)."""
    print(f"vector-add block-size sweep  (N={n:,}, peak BW {dev.peak_bw_gbps:.0f} GB/s, "
          f"{dev.sm_count} SMs, {dev.max_threads_sm} threads/SM)\n")
    print(f"{'block':>6} │ {'blocks/SM':>9} │ {'occupancy':>9} │ {'eff BW':>10} │ {'% peak':>7}")
    print("─" * 56)
    for r in rows:
        res = r["res"]
        print(f"{r['block']:>6} │ {r['blocks_per_sm']:>9} │ {r['occupancy']*100:>8.0f}% │ "
              f"{res.eff_bw_gbps:>8.1f} GB/s │ {res.pct_peak_bw:>6.1f}%")
    best = max(rows, key=lambda r: r["res"].eff_bw_gbps)
    print(f"\n  measured winner : block={best['block']} "
          f"({best['res'].eff_bw_gbps:.1f} GB/s, {best['res'].pct_peak_bw:.1f}% peak)")
    print(f"  CUDA suggests   : block={_mod.suggested_block()}  (cudaOccupancyMaxPotentialBlockSize)")
    print(f"  our default (256): {'confirmed near-optimal' if 128 <= best['block'] <= 512 else 'OVERRULED — see table'}")
    return best


def _persist_best(best: dict, dev, path: str = "benchmarks/p00_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    rec = {**asdict(best["res"]), "name": "vector_add_tuned",
           "block": best["block"], "occupancy": best["occupancy"]}
    data = [d for d in data if d.get("name") != "vector_add_tuned"] + [rec]  # upsert tuned winner
    p.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    dev = probe_device()
    n, rows = sweep_block_sizes(dev)
    best = print_sweep_table(n, rows, dev)
    _persist_best(best, dev)
    print("\nTuned config persisted to benchmarks/p00_results.json")
