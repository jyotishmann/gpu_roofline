# src/harness/__init__.py — the harness package's public surface
from .device import probe_device as probe_device, DeviceInfo as DeviceInfo
from .timing import time_kernel as time_kernel, benchmark as benchmark, BenchResult as BenchResult, effective_bandwidth_gbps as effective_bandwidth_gbps
