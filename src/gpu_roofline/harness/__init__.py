# src/harness/__init__.py — the harness package's public surface
from .device import probe_device, DeviceInfo  # re-export so callers import from `harness`, not `harness.device`
from .timing import time_kernel, benchmark, BenchResult, effective_bandwidth_gbps
