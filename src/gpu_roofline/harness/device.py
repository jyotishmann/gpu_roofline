# src/harness/device.py — device-probe module
import torch

from dataclasses import dataclass, asdict
from torch.utils.cpp_extension import load_inline

import json
import pathlib

def assert_cuda() -> None:
    """Refuse to run the harness without a usable CUDA device — fail loud, fail early."""
    if not torch.cuda.is_available():  # a False here means driver/runtime/device didn't all agree
        raise RuntimeError(
            "No CUDA device visible. In Colab: Runtime ▸ Change runtime type ▸ Hardware "
            "accelerator ▸ GPU, then re-run. (torch.cuda.is_available() returned False.)"
        )

def torch_device_record(index: int = 0) -> dict:
    """The device facts PyTorch surfaces directly: identity, arch, SM count, capacity."""
    p = torch.cuda.get_device_properties(index)
    return {
        "name": p.name,
        "cc": (p.major, p.minor),          # compute capability: gates features AND cores/SM (see 1.6)
        "sm_count": p.multi_processor_count,
        "total_mem_bytes": p.total_memory,
    }


_PROBE_CPP = r"""
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <map>
#include <string>

std::map<std::string, long long> probe(long long dev) {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, (int)dev);  // the runtime struct carries what torch's Python object omits
    return {
        {"mem_clock_khz",     (long long)p.memoryClockRate},
        {"bus_width_bits",    (long long)p.memoryBusWidth},
        {"core_clock_khz",    (long long)p.clockRate},
        {"l2_bytes",          (long long)p.l2CacheSize},
        {"max_threads_block", (long long)p.maxThreadsPerBlock},
        {"max_threads_sm",    (long long)p.maxThreadsPerMultiProcessor},
        {"warp_size",         (long long)p.warpSize},
        {"smem_block_bytes",  (long long)p.sharedMemPerBlock},
        {"smem_sm_bytes",     (long long)p.sharedMemPerMultiprocessor},
        {"regs_block",        (long long)p.regsPerBlock},
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &probe); }
"""

_probe_mod = load_inline(
    name="p00_device_probe",
    cpp_sources=_PROBE_CPP,
    functions=None,                 # we supply our own PYBIND11_MODULE, so don't auto-generate one
    with_cuda=True,
    extra_ldflags=["-lcudart"],
    verbose=False,
)


def runtime_device_record(index: int = 0) -> dict:
    """The performance-critical fields we must read straight from the CUDA runtime."""
    return dict(_probe_mod.probe(index))

def peak_bandwidth_gbps(mem_clock_khz: int, bus_width_bits: int) -> float:
    """Theoretical peak DRAM bandwidth (decimal GB/s) from memory clock and bus width."""
    bytes_per_transfer = bus_width_bits / 8
    transfers_per_sec = 2 * (mem_clock_khz * 1_000)   # ×2 is DDR: data on both clock edges
    return transfers_per_sec * bytes_per_transfer / 1e9

_KNOWN_PEAK_GBPS = {"T4": 320.0, "A100": 1555.0, "V100": 900.0, "L4": 300.0}


def sanity_check_bandwidth(name: str, computed_gbps: float, tol: float = 0.08) -> None:
    """Smoke alarm: warn (don't fail) if computed peak drifts from a known card's datasheet."""
    for key, ref in _KNOWN_PEAK_GBPS.items():
        if key in name and abs(computed_gbps - ref) / ref > tol:  # datasheet is ground truth for the check
            print(f"[warn] {name}: computed {computed_gbps:.0f} GB/s vs datasheet ~{ref:.0f} GB/s "
                  f"(>{tol:.0%} off — check the queried fields)")

_CORES_PER_SM = {  # provenance: CUDA-samples helper_cuda.h :: _ConvertSMVer2Cores (keyed by CC)
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0):  64, (6, 1): 128, (6, 2): 128,
    (7, 0):  64, (7, 2):  64, (7, 5):  64,      # 7.5 = Turing (T4)
    (8, 0):  64, (8, 6): 128, (8, 7): 128, (8, 9): 128,   # 8.0 = Ampere (A100)
    (9, 0): 128, (10, 0): 128, (12, 0): 128,
}


def peak_fp32_gflops(cc: tuple, sm_count: int, core_clock_khz: int) -> float:
    """Theoretical peak FP32 (GFLOP/s); depends on a maintained arch cores/SM table."""
    cores = _CORES_PER_SM[cc] * sm_count
    return cores * 2 * (core_clock_khz * 1_000) / 1e9   # ×2 because one FMA = two FLOPs


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    cc: tuple
    sm_count: int
    total_mem_bytes: int
    mem_clock_khz: int
    bus_width_bits: int
    core_clock_khz: int
    l2_bytes: int
    max_threads_block: int
    max_threads_sm: int
    warp_size: int
    peak_bw_gbps: float
    peak_fp32_gflops: float
    ridge_flop_per_byte: float


def probe_device(index: int = 0) -> DeviceInfo:
    """One call returning every ceiling the roofline harness will need, read from the device."""
    assert_cuda()
    t = torch_device_record(index)
    r = runtime_device_record(index)
    bw = peak_bandwidth_gbps(r["mem_clock_khz"], r["bus_width_bits"])
    fp32 = peak_fp32_gflops(t["cc"], t["sm_count"], r["core_clock_khz"])
    sanity_check_bandwidth(t["name"], bw)
    return DeviceInfo(
        name=t["name"], cc=t["cc"], sm_count=t["sm_count"],
        total_mem_bytes=t["total_mem_bytes"],
        mem_clock_khz=r["mem_clock_khz"], bus_width_bits=r["bus_width_bits"],
        core_clock_khz=r["core_clock_khz"], l2_bytes=r["l2_bytes"],
        max_threads_block=r["max_threads_block"], max_threads_sm=r["max_threads_sm"],
        warp_size=r["warp_size"], peak_bw_gbps=bw, peak_fp32_gflops=fp32,
        ridge_flop_per_byte=fp32 / bw,   # roofline corner: kernels left of this are bandwidth-bound
    )


def save_device_info(info: DeviceInfo, path: str = "benchmarks/device_info.json") -> None:
    """Pin the ceilings to disk so every later report is reproducible against them."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**asdict(info), "cc": list(info.cc)}, indent=2))  # tuple→list: JSON has no tuples


if __name__ == "__main__":
    info = probe_device()
    print(f"{info.name}  (sm_{info.cc[0]}{info.cc[1]}, {info.sm_count} SMs)")
    print(f"  peak bandwidth : {info.peak_bw_gbps:8.1f} GB/s")
    print(f"  peak FP32      : {info.peak_fp32_gflops / 1000:8.2f} TFLOP/s")
    print(f"  ridge point    : {info.ridge_flop_per_byte:8.1f} FLOP/byte")
    save_device_info(info)
