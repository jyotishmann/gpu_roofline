# src/gemm/api.py — public API contract (stubs)
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass(frozen=True)
class DeviceSpec:
    """Hardware ceilings and capacities extracted from the device probe.
    The performance model uses these; callers obtain one via `DeviceSpec.from_probe()`.
    """
    name:           str
    peak_fp32_gflops: float   # GFLOP/s (FP32 CUDA cores, no tensor cores)
    peak_bw_gbps:   float     # GB/s (DRAM bandwidth, decimal)
    ridge_flop_per_byte: float  # peak_fp32_gflops / peak_bw_gbps
    smem_per_sm_bytes: int    # shared memory bytes per SM (configurable upper limit)
    regs_per_sm:    int       # total register file words per SM
    max_threads_sm: int       # max resident threads per SM (limits occupancy)
    cc:             tuple     # compute capability, e.g. (8, 0) for A100

    @classmethod
    def from_probe(cls) -> DeviceSpec:
        """Build from the initial harness (no hardcoded values)."""


@dataclass(frozen=True)
class KernelConfig:
    """A complete, hashable GEMM tile configuration — the unit the autotuner reasons over.
    Frozen so it can be used as a dict key in the results cache.
    """
    BM:         int   # output tile rows (must be multiple of WM * 32)
    BN:         int   # output tile columns (must be multiple of WN * 32)
    BK:         int   # k-tile depth
    WM:         int   # per-thread register rows
    WN:         int   # per-thread register columns
    num_stages: int   # software pipeline depth (1 = synchronous; >1 requires cp.async)

    @property
    def smem_bytes(self) -> int:
        """Shared memory needed per block: As[BM][BK] + Bs[BK][BN], float32."""
        return (self.BM * self.BK + self.BK * self.BN) * 4

    @property
    def threads_per_block(self) -> int:
        """Block is (BN//WN, BM//WM) threads (strided ownership model)."""
        return (self.BN // self.WN) * (self.BM // self.WM)

    @property
    def arithmetic_intensity(self) -> float:
        """Tile-level AI: BM·BN / (2·(BM+BN)) FLOP/byte."""
        return self.BM * self.BN / (2 * (self.BM + self.BN))


def predict_configs(dev: DeviceSpec, M: int, N: int, K: int,
                    top_k: int = 5) -> list[KernelConfig]:
    """
    Return the top-k `KernelConfig` objects ranked by predicted throughput,
    derived analytically from `dev`'s hardware ceilings (no kernel launches).

    The model uses the roofline formula corrected for register
    pressure and shared memory occupancy. Its ranking is correct for ~80% of
    (M, N, K, dev) combinations in practice; the autotuner verifies the top-k
    by measurement.

    Args:
        dev:   Hardware spec (from DeviceSpec.from_probe() or a known-card dict).
        M, N, K: Problem dimensions.
        top_k: How many candidates to return (typically 3–10).

    Returns:
        Ordered list of KernelConfig, best prediction first.
        Every config is feasible (smem ≤ dev limit, regs ≤ budget, K % BK == 0).

    Raises:
        ValueError: if no feasible configs exist for (dev, M, N, K).
    """
    raise NotImplementedError("implemented in p08/03")


def autotune(dev: DeviceSpec, M: int, N: int, K: int,
             warmup: int = 10, iters: int = 50) -> KernelConfig:
    """
    Run `predict_configs`, benchmark each candidate, return the fastest.

    The function is *idempotent*: calling it twice with the same (dev, M, N, K)
    returns the same config (the result is cached keyed by (dev.name, M, N, K)).

    Args:
        dev, M, N, K: as in predict_configs.
        warmup, iters: passed to the benchmark harness.

    Returns:
        The KernelConfig that achieved the highest measured GFLOP/s.
    """
    raise NotImplementedError("implemented in p08/04")


def run(A: torch.Tensor, B: torch.Tensor,
        config: KernelConfig | None = None,
        dev: DeviceSpec | None = None) -> torch.Tensor:
    """
    Compute C = A @ B using the given config (or autotune if config is None).

    If `config` is None and `dev` is None, DeviceSpec.from_probe() is called.
    If `config` is None, autotune(dev, *A.shape, B.shape[1]) is called and cached.

    Args:
        A: [M, K] float32 on CUDA.
        B: [K, N] float32 on CUDA.
        config: Optional pre-tuned KernelConfig.
        dev: Optional DeviceSpec; probed if None.

    Returns:
        C: [M, N] float32 on CUDA.

    Raises:
        RuntimeError: if A, B are not CUDA float32 tensors.
        ValueError:   if A.shape[1] != B.shape[0].
    """
    raise NotImplementedError("implemented in later segment")
