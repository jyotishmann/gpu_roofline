# src/gemm/perf_model.py — performance model for the autotuning library
import math
import json, pathlib
from gpu_roofline.gemm.api import DeviceSpec, KernelConfig
from gpu_roofline.gemm.tiles import enumerate_candidate_configs, is_feasible


def predicted_throughput_gflops(cfg: KernelConfig, dev: DeviceSpec) -> float:
    """
    Predicted GFLOP/s = min(peak_FP32, AI × peak_BW) × η_occ × η_pipe
    This is the roofline ceiling corrected for occupancy and pipeline efficiency.
    This is a ranker — absolute accuracy is not required, ranking is.
    """
    # Roofline ceiling
    T_roof = min(dev.peak_fp32_gflops,
                 cfg.arithmetic_intensity * dev.peak_bw_gbps)

    # Occupancy: how many blocks fit per SM (limited by smem and threads)
    blocks_by_smem    = dev.smem_per_sm_bytes // max(cfg.smem_bytes, 1)
    blocks_by_threads = dev.max_threads_sm // max(cfg.threads_per_block, 1)
    blocks_per_sm     = min(blocks_by_smem, blocks_by_threads, 32)  # hw max
    warps_resident    = blocks_per_sm * (cfg.threads_per_block // 32)
    warps_max         = dev.max_threads_sm // 32
    eta_occ           = warps_resident / max(warps_max, 1)

    # Pipeline efficiency: cp.async is only effective on sm_80+
    if cfg.num_stages > 1 and dev.cc >= (8, 0):
        eta_pipe = min(0.95, 0.7 + 0.1 * cfg.num_stages)   # empirical: stages help up to ~3
    else:
        eta_pipe = 0.70 if cfg.num_stages == 1 else 0.70    # no-op on sm_75

    return T_roof * eta_occ * eta_pipe


def predict_configs(dev: DeviceSpec, M: int, N: int, K: int,
                    top_k: int = 5) -> list[KernelConfig]:
    """Rank feasible configs by predicted throughput; return top-k."""
    candidates = enumerate_candidate_configs(K, dev)
    if not candidates:
        raise ValueError(f"No feasible configs for K={K} on {dev.name}")
    scored = [(predicted_throughput_gflops(c, dev), c) for c in candidates]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


def validate_model_ranking(dev: DeviceSpec, M: int = 2048, N: int = 2048,
                            K: int = 2048) -> dict:
    """Compare model's top-5 prediction against the p02/06 measured winner."""
    predicted = predict_configs(dev, M, N, K, top_k=5)
    # Load the p02/06 measured winner if available
    p02_path = pathlib.Path("../../tier1-make-it-fast/p02-sgemm/benchmarks/p02_best_config.json")
    measured_winner = None
    if p02_path.exists():
        d = json.loads(p02_path.read_text())
        measured_winner = KernelConfig(BM=d["wm"]*32, BN=d["wn"]*32, BK=32,
                                       WM=d["wm"], WN=d["wn"], num_stages=1)
    found_in_top_k = measured_winner in predicted if measured_winner else None
    print(f"\nModel validation  (M=N=K={M}, {dev.name})")
    print(f"  Top-5 predicted configs (ranked by predicted GFLOP/s):")
    for rank, cfg in enumerate(predicted, 1):
        marker = " ← measured winner" if cfg == measured_winner else ""
        print(f"    {rank}. BM={cfg.BM} BN={cfg.BN} BK={cfg.BK} WM={cfg.WM} WN={cfg.WN} "
              f"stages={cfg.num_stages}  AI={cfg.arithmetic_intensity:.0f}{marker}")
    if measured_winner:
        print(f"  Measured winner in top-5: {found_in_top_k}")
    return {"predicted": predicted, "measured_winner": measured_winner,
            "found_in_top_k": found_in_top_k}


if __name__ == "__main__":
    import sys
    dev = DeviceSpec.from_probe()
    result = validate_model_ranking(dev)
    print(f"\nModel produces {len(result['predicted'])} candidate configs "
          f"from {len(enumerate_candidate_configs(2048, dev))} feasible.")
    print("Next: benchmarking the top-5 to find the measured winner.")
