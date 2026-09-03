# src/harness/transfer.py — PCIe transfer benchmark
"""
Three-ceiling mental model
  (1) Peak DRAM BW   — kernels; 
  (2) Peak PCIe BW   — host↔device transfers; 
  (3) Peak FP32      — compute-bound kernels;
"""
import time
import statistics
import pathlib
import json
from dataclasses import dataclass, asdict
import torch

from gpu_roofline.harness.device import probe_device

_TRANSFER_BYTES_DEFAULT = 256 * 1024 * 1024  # 256 MB — saturates PCIe Gen3/Gen4 reliably


def alloc_transfer_buffers(nbytes: int = _TRANSFER_BYTES_DEFAULT):
    """Return (cpu_pageable, cpu_pinned, gpu_buf) of the same size. Called once; cost excluded."""
    n = nbytes // 4                              # float32: 4 bytes each
    cpu_pageable = torch.zeros(n)                # default host allocation — pages may be swapped
    cpu_pinned   = torch.zeros(n).pin_memory()   # cudaHostAlloc: page-locked, direct DMA eligible
    gpu_buf      = torch.empty(n, device="cuda") # device buffer — reused for all four directions
    return cpu_pageable, cpu_pinned, gpu_buf, nbytes


def _time_transfer_ms(fn, warmup: int = 5, iters: int = 30) -> dict:
    """
    Time a host↔device transfer call with wall clock + explicit synchronize.

    Rationale: CUDA events capture only the on-device DMA; they miss the CPU-side
    staging copy for pageable memory — the dominant cost. Wall clock + sync gives
    total observed transfer latency, which is what a real caller experiences.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()                  # drain each warmup iteration cleanly
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()                  # drain any prior GPU work before starting the clock
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()                  # wait for transfer to fully complete
        samples.append((time.perf_counter() - t0) * 1e3)   # seconds → milliseconds
    return {"median_ms": statistics.median(samples),
            "min_ms":    min(samples),
            "std_ms":    statistics.pstdev(samples)}


@dataclass(frozen=True)
class TransferResult:
    direction: str      # "H2D" or "D2H"
    mem_type:  str      # "pageable" or "pinned"
    nbytes:    int
    median_ms: float
    min_ms:    float
    bw_gbps:   float    # decimal GB/s, matches the p00/01 DRAM ceiling units


def _bw(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def _pcie_gen() -> str:
    """Best-effort PCIe generation read from sysfs — falls back to 'unknown'."""
    try:
        speeds = list(pathlib.Path("/sys/bus/pci/devices").glob("*/current_link_speed"))
        for p in speeds:
            txt = p.read_text().strip()
            if "8" in txt:  
                return "Gen3 (8 GT/s)"   # 8 GT/s = PCIe Gen3
            if "16" in txt: 
                return "Gen4 (16 GT/s)"  # 16 GT/s = PCIe Gen4
            if "32" in txt: 
                return "Gen5 (32 GT/s)"  # 32 GT/s = PCIe Gen5
    except Exception:
        pass
    return "unknown"  # sysfs not visible in this cgroup — report measurements without a ceiling


def run_transfer_benchmarks(nbytes: int = _TRANSFER_BYTES_DEFAULT) -> list[TransferResult]:
    cpu_pg, cpu_pin, gpu_buf, nb = alloc_transfer_buffers(nbytes)
    results = []
    for label, fn_h2d, fn_d2h in [
        ("pageable", lambda: gpu_buf.copy_(cpu_pg),  lambda: cpu_pg.copy_(gpu_buf)),
        ("pinned",   lambda: gpu_buf.copy_(cpu_pin), lambda: cpu_pin.copy_(gpu_buf)),
    ]:
        for direction, fn in [("H2D", fn_h2d), ("D2H", fn_d2h)]:
            t = _time_transfer_ms(fn)
            results.append(TransferResult(
                direction=direction, mem_type=label, nbytes=nb,
                median_ms=t["median_ms"], min_ms=t["min_ms"],
                bw_gbps=_bw(nb, t["median_ms"]),
            ))
    return results


def print_transfer_table(results: list[TransferResult], dram_bw_gbps: float) -> None:
    gen = _pcie_gen()
    print(f"PCIe link: {gen}   │   peak DRAM BW: {dram_bw_gbps:.0f} GB/s\n")
    print(f"{'direction':<10}│ {'mem type':<10}│ {'BW (GB/s)':>10} │ {'note'}")
    print("─" * 56)
    for r in results:
        ratio = f"{r.bw_gbps / dram_bw_gbps * 100:.1f}% of DRAM"
        print(f"{r.direction:<10}│ {r.mem_type:<10}│ {r.bw_gbps:>10.1f} │ {ratio}")
    best = max(r.bw_gbps for r in results)
    print(f"\n  peak DRAM is {dram_bw_gbps / best:.0f}× faster than the best PCIe transfer measured")


def _upsert_transfer(res: TransferResult,
                     path: str = "benchmarks/p00_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    key = f"transfer_{res.direction}_{res.mem_type}"
    rec = {**asdict(res), "name": key}
    data = [d for d in data if d.get("name") != key] + [rec]  # upsert by composite key
    p.write_text(json.dumps(data, indent=2))


def _roundtrip_sanity(nbytes: int = 64 * 1024 * 1024) -> None:
    """H2D then D2H must recover the original values — confirms copy_() is lossless."""
    n   = nbytes // 4
    src = torch.randn(n)
    gpu = src.cuda()                       # H2D
    dst = gpu.cpu()                        # D2H
    assert torch.equal(src, dst), "round-trip H2D→D2H changed values — transfer is broken"
    print("[ok] round-trip H2D→D2H exact-matches source")


if __name__ == "__main__":
    dev = probe_device()
    _roundtrip_sanity()
    results = run_transfer_benchmarks()
    print_transfer_table(results, dev.peak_bw_gbps)
    for r in results:
        _upsert_transfer(r)
    print("\nResults appended to benchmarks/p00_results.json")
