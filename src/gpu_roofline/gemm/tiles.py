# src/gemm_lib/tiles.py — tile algebra and constraint checking
from __future__ import annotations
from dataclasses import dataclass
import sys
from gpu_roofline.gemm.api import DeviceSpec, KernelConfig
import torch


@dataclass(frozen=True)
class TileLayout:
    """Minimal tile description: shape + stride pair (CuTe-inspired vocabulary type).
    The stride determines memory access pattern; (BK, 1) is row-major for a (BM, BK) tile.
    """
    shape:  tuple[int, int]
    stride: tuple[int, int]   # (row_stride, col_stride) in units of elements

    @property
    def is_row_major(self) -> bool:
        return self.stride[1] == 1 and self.stride[0] == self.shape[1]

    @property
    def is_col_major(self) -> bool:
        return self.stride[0] == 1 and self.stride[1] == self.shape[0]


def DeviceSpec_from_probe() -> DeviceSpec:
    """Read hardware ceilings from the Project-00 harness (ADR-007: compute, don't hardcode)."""
    from gpu_roofline.harness.device import probe_device
    dev = probe_device()
    return DeviceSpec(
        name=dev.name, peak_fp32_gflops=dev.peak_fp32_gflops,
        peak_bw_gbps=dev.peak_bw_gbps, ridge_flop_per_byte=dev.ridge_flop_per_byte,
        smem_per_sm_bytes=dev.smem_sm_bytes, regs_per_sm=dev.regs_block,
        max_threads_sm=dev.max_threads_sm, cc=dev.cc,
    )


# Monkey-patch the classmethod (implementation separated from stub file)
DeviceSpec.from_probe = classmethod(lambda cls: DeviceSpec_from_probe())


def is_feasible(cfg: KernelConfig, dev: DeviceSpec, K: int,
                smem_limit: int | None = None) -> tuple[bool, str]:
    """Return (feasible, reason). Reason is '' if feasible, else a short explanation."""
    limit = smem_limit or dev.smem_per_sm_bytes
    if cfg.smem_bytes > limit:
        return False, f"smem {cfg.smem_bytes//1024}KB > limit {limit//1024}KB"
    if cfg.threads_per_block > 1024:
        return False, f"threads/block {cfg.threads_per_block} > 1024"
    if K % cfg.BK != 0:
        return False, f"K={K} not divisible by BK={cfg.BK}"
    regs_per_thread = cfg.WM * cfg.WN + cfg.WM + cfg.WN + 8  # accumulators + loads + overhead
    budget = dev.regs_per_sm // cfg.threads_per_block
    if regs_per_thread > budget:
        return False, f"~{regs_per_thread} regs/thread > budget {budget} (may spill)"
    if cfg.BM % (cfg.WM * 32) != 0 or cfg.BN % (cfg.WN * 32) != 0:
        return False, "BM/WM or BN/WN not divisible by 32 (warp size)"
    return True, ""


def enumerate_candidate_configs(K: int, dev: DeviceSpec) -> list[KernelConfig]:
    """Generate all feasible KernelConfig objects for the given K and device."""
    candidates = []
    for WM in [2, 4, 8]:
        for WN in [2, 4, 8]:
            for BK in [8, 16, 32]:
                for stages in [1, 2, 3] if dev.cc >= (8, 0) else [1]:
                    BM = WM * 32; BN = WN * 32   # fixed: block is always 32×32
                    cfg = KernelConfig(BM=BM, BN=BN, BK=BK, WM=WM, WN=WN, num_stages=stages)
                    ok, reason = is_feasible(cfg, dev, K)
                    if ok:
                        candidates.append(cfg)
    return candidates


if __name__ == "__main__":
    dev = DeviceSpec.from_probe()
    K = 2048
    configs = enumerate_candidate_configs(K, dev)
    print(f"\nFeasible configurations for {dev.name} (cc={dev.cc}), K={K}:")
    print(f"  {'BM':>4}  {'BN':>4}  {'BK':>4}  {'WM':>3}  {'WN':>3}  "
          f"{'stages':>6}  {'AI':>7}  {'smem KB':>8}")
    print("─" * 56)
    for c in sorted(configs, key=lambda x: -x.arithmetic_intensity):
        print(f"  {c.BM:>4}  {c.BN:>4}  {c.BK:>4}  {c.WM:>3}  {c.WN:>3}  "
              f"{c.num_stages:>6}  {c.arithmetic_intensity:>7.0f}  {c.smem_bytes//1024:>6} KB")
    print(f"\n  Ridge: {dev.ridge_flop_per_byte:.1f} FLOP/byte — "
          f"configs above this AI are compute-bound")
