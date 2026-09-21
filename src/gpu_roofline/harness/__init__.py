# src/gpu_roofline/harness/__init__.py — the harness package's public surface
from gpu_roofline.harness.device import probe_device as probe_device, DeviceInfo as DeviceInfo
from gpu_roofline.harness.timing import time_kernel as time_kernel, benchmark as benchmark, BenchResult as BenchResult, effective_bandwidth_gbps as effective_bandwidth_gbps
from gpu_roofline.harness.transfer import run_transfer_benchmarks as run_transfer_benchmarks, print_transfer_table as print_transfer_table, TransferResult as TransferResult
from gpu_roofline.harness.block_sweep import sweep_block_sizes as sweep_block_sizes, occupancy_grid as occupancy_grid, theoretical_occupancy as theoretical_occupancy
from gpu_roofline.harness.roofline import build_project00_report as build_project00_report, attainable_gflops as attainable_gflops
from gpu_roofline.harness.climb import build_project01_report
from gpu_roofline.harness.parity import build_project02_report
from gpu_roofline.harness.fa_parity import build_project03_report
from gpu_roofline.harness.tier2_closure import build_tier2_report
from gpu_roofline.harness.ar_parity import build_project05_report
from gpu_roofline.harness.tp_parity import build_project06_report
from gpu_roofline.harness.kv_parity import build_tier3_closure
