# src/harness/timing.py — the measurement core
import torch
from dataclasses import dataclass
import statistics

import json
import pathlib
from dataclasses import asdict

from harness.device import probe_device
from kernels.vector_add import add


def _time_once(fn) -> float:
    """Time a single kernel call in device-milliseconds using a CUDA event pair."""
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    stop.record()
    torch.cuda.synchronize()          # elapsed_time is only valid once both events have completed
    return start.elapsed_time(stop)   # returns milliseconds of on-device time


def time_kernel(fn, warmup: int = 25, iters: int = 100) -> dict:
    """Warm up, then collect `iters` device-times; return median/min/std (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = [_time_once(fn) for _ in range(iters)]  # median over many samples resists throttle jitter
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "std_ms": statistics.pstdev(samples),
        "iters": iters,
    }


def effective_bandwidth_gbps(bytes_moved: int, ms: float) -> float:
    """Achieved DRAM bandwidth (decimal GB/s) = bytes moved / elapsed seconds."""
    return bytes_moved / (ms * 1e-3) / 1e9  # ms→s, then bytes/s→GB/s (decimal, matches the peak ceiling)


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    return flops / bytes_moved


def gflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e9


@dataclass(frozen=True)
class BenchResult:
    name: str
    n: int
    median_ms: float
    min_ms: float
    eff_bw_gbps: float
    pct_peak_bw: float
    gflops: float
    pct_peak_fp32: float
    arithmetic_intensity: float


def benchmark(name, fn, bytes_per_call, flops_per_call, dev, n, warmup=25, iters=100) -> BenchResult:
    """Kernel-agnostic benchmark: time a launch closure, express results vs device ceilings."""
    t = time_kernel(fn, warmup, iters)
    bw = effective_bandwidth_gbps(bytes_per_call, t["median_ms"])
    gf = gflops(flops_per_call, t["median_ms"])
    return BenchResult(
        name=name, n=n, median_ms=t["median_ms"], min_ms=t["min_ms"],
        eff_bw_gbps=bw, pct_peak_bw=100.0 * bw / dev.peak_bw_gbps,
        gflops=gf, pct_peak_fp32=100.0 * gf / dev.peak_fp32_gflops,
        arithmetic_intensity=arithmetic_intensity(flops_per_call, bytes_per_call),
    )  # % of peak is the headline: bandwidth for memory-bound kernels, FP32 shown to expose imbalance


def _save_result(res: BenchResult, path: str = "benchmarks/p00_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    data = [d for d in data if d["name"] != res.name] + [asdict(res)]  # upsert by kernel name
    p.write_text(json.dumps(data, indent=2))


def run_vector_add_benchmark(dev=None, iters: int = 100) -> BenchResult:
    dev = dev or probe_device()
    n = int(dev.l2_bytes * 8 / (3 * 4))            # working set ≈ 8× L2 → forces HBM traffic, not cache
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    out = torch.empty_like(a)
    res = benchmark("vector_add", lambda: add(a, b, out), 3 * n * 4, n, dev, n, iters=iters)
    assert abs(res.arithmetic_intensity - 1 / 12) < 1e-6, "AI should be exactly 1/12 for fp32 add"
    assert 5.0 < res.pct_peak_bw <= 100.0, "achieved bandwidth is implausible — check sizing/timer"
    return res


if __name__ == "__main__":
    dev = probe_device()
    r = run_vector_add_benchmark(dev)
    print(f"{r.name}  N={r.n:,}  (working set {3 * r.n * 4 / 1e6:.0f} MB)")
    print(f"  time (median/min) : {r.median_ms:.3f} / {r.min_ms:.3f} ms")
    print(f"  effective BW      : {r.eff_bw_gbps:8.1f} GB/s   ({r.pct_peak_bw:.1f}% of peak {dev.peak_bw_gbps:.0f})")
    print(f"  compute           : {r.gflops:8.1f} GFLOP/s ({r.pct_peak_fp32:.2f}% of peak FP32)")
    print(f"  arithmetic int.   : {r.arithmetic_intensity:.3f} FLOP/byte  → bandwidth-bound")
    _save_result(r)
