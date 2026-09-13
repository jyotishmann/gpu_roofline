# src/gemm/blackwell_analysis.py — Blackwell co-design analysis
_BLACKWELL_ANALYSIS = """
# Blackwell Co-Design Analysis
*Applying the gpu-mastery performance models to NVIDIA B200 (Blackwell, 2024)*

## 1. B200 Hardware Specifications

| Parameter | T4 (sm_75) | A100 (sm_80) | B200 (sm_90) |
|---|---|---|---|
| Peak FP32 (CUDA cores) | 8.1 TFLOP/s | 19.5 TFLOP/s | ~80 TFLOP/s |
| Peak BF16 (tensor cores) | — | ~77 TFLOP/s | ~1,125 TFLOP/s |
| Peak FP8 (tensor cores) | — | — | ~2,250 TFLOP/s |
| DRAM bandwidth | 320 GB/s | 2,000 GB/s | ~8,000 GB/s |
| NVLink bandwidth (per GPU) | — | 600 GB/s | 900 GB/s |
| SMEM per SM | 64 KB | 228 KB | 228 KB |

## 2. Ridge Points (derived from §3 of the p02 master doc)

ridge = peak_compute / peak_bw (FLOP/byte)

| Card | FP32 ridge | BF16 ridge | FP8 ridge |
|---|---|---|---|
| T4 | 25 | — | — |
| A100 | ~10 | ~39 | ~70 |
| B200 | ~10 | **141** | **281** |

Key observation: B200's BF16 ridge (141 FLOP/byte) is 3.6× higher than A100's BF16 ridge
(39 FLOP/byte). Tile configurations that were compute-bound on A100 BF16 are bandwidth-bound
on B200 BF16.

## 3. Tile Size Implications for GEMM (from §3.2 of the p02 master doc)

AI(BM, BN) = BM × BN / (2 × (BM + BN))

For BM = BN = T (square tiles): AI = T / 4

Break-even condition (AI > ridge → compute-bound):
  T > 4 × ridge

| Card | FP32 break-even T | BF16 break-even T |
|---|---|---|
| T4 | T > 100 | — |
| A100 | T > 40 | T > 156 |
| B200 | T > 40 | **T > 564** |

Implication: the 128×128 tiles that reach compute-bound on A100 BF16 are bandwidth-bound on
B200 BF16 (128/4 = 32 < 141). Standard CTA-level tiles are insufficient.

Solution: Blackwell's WGMMA (Warpgroup Matrix Multiply Accumulate) enables multi-CTA clusters
that cooperate on a shared tile. A cluster of 4 CTAs effectively produces a 256×256 logical
tile (AI = 64) or an 8-CTA cluster produces a 512×512 tile (AI = 128 ≈ break-even). This
is why Blackwell introduces the "cluster" programming model: it is the only way to achieve
compute-bound BF16 GEMM.

**Structural change required:** the p08 kernel factory's tile sizes need a new dimension for
B200 BF16: the cluster size (how many CTAs cooperate). This is not a retune — it is a
redesign. The existing library handles sm_80 and below; a B200 extension would add a
`ClusteredKernelConfig` with a `cluster_shape: (int, int)` field.

## 4. All-Reduce Impact (from p05 master doc §2.4)

Break-even hidden dimension: h_breakeven = 2(N-1) × ridge

For BF16 TP on B200 (ridge_BF16 = 141):
  h_breakeven(N=4) = 2 × 3 × 141 = 846

For BF16 TP on A100 (ridge_BF16 = 39):
  h_breakeven(N=4) = 2 × 3 × 39 = 234

Both are below typical transformer hidden dimensions (h ≥ 1024), so TP remains efficient on
B200. However, the ratio T_comm/T_compute drops dramatically on B200 because compute is 8×
faster (BF16 tensor cores) while NVLink is only 1.5× faster. TP efficiency η approaches 1
even faster, and the overhead of the all-reduce shrinks as a fraction of total step time.

**No structural change required for p05 (all-reduce):** the ring algorithm is optimal
regardless of the bandwidth ratio. The p05 library correctly re-tunes when given B200's
`peak_bw_gbps` (from NVLink 5.0, not DRAM).

## 5. FlashAttention Impact (from p03 master doc §5)

Per-tile AI for paged FlashAttention: AI ≈ Bq × Bkv / (Bq + 2·Bkv)

For Bq=Bkv=64: AI ≈ 21 FLOP/byte.

On B200 FP8 (ridge = 281): AI = 21 ≪ 281 — bandwidth-bound.

Required tile for B200 FP8 compute-bound:
  Bq × Bkv / (Bq + 2·Bkv) > 281  →  for square tiles: Bq/4 > 281  →  Bq > 1124

This is impossible for a single CTA (max 1024 threads).

**Structural change required for FP8 FlashAttention on B200:**
1. Larger effective tiles via cluster (same as GEMM).
2. FP8 quantisation of K and V before the attention softmax (acceptable for inference;
   requires careful scale management in the online softmax).
3. The p03/02 online softmax recurrence stays valid; the numerics change because FP8 has
   a 5-bit mantissa, requiring explicit per-head scale factors to avoid overflow.

## 6. Summary: what changes structurally vs what just retunes

| Project | B200 change needed |
|---|---|
| p02 SGEMM (FP32) | Retune only (FP32 ridge unchanged at ~10) |
| p02 SGEMM (BF16) | **Structural:** cluster tiles for Bq > 564 |
| p03 FlashAttention (FP32) | Retune only |
| p03 FlashAttention (FP8) | **Structural:** cluster tiles + FP8 scale management |
| p05 All-reduce | Retune only (ring algorithm stays; bw_gbps changes) |
| p06 Tensor Parallelism | Retune only (η formula unchanged; h_breakeven drops) |
| p07 Paged KV Cache | Retune only (page size, pool geometry) |
| p08 GEMM library | **Structural:** add ClusteredKernelConfig for BF16/FP8 |

The pattern: FP32 kernels just retune. BF16 and FP8 kernels on Blackwell require new cluster-
level tile abstractions because the ridge point has moved beyond what single-CTA tiles can
reach. This is the "hardware-roadmap literacy" test: reading a new GPU's spec sheet and
predicting which of your existing kernels need structural redesign vs which just need new
autotuning configs.
"""


# Blackwell specs (from NVIDIA B200 datasheet, 2024)
B200_PEAK_FP32_GFLOPS   =    80_000    # ~80 TFLOP/s CUDA cores
B200_PEAK_BF16_GFLOPS   = 1_125_000   # ~1.125 PFLOP/s tensor cores
B200_PEAK_FP8_GFLOPS    = 2_250_000   # ~2.25 PFLOP/s tensor cores
B200_PEAK_BW_GBPS       =   8_000     # HBM3e
B200_NVLINK5_GBPS       =     900     # per GPU, bidirectional

def b200_ridge(compute_gflops: float, bw_gbps: float = B200_PEAK_BW_GBPS) -> float:
    return compute_gflops / bw_gbps

def b200_tile_breakeven(ridge_flop_per_byte: float) -> int:
    """Minimum square tile side length T for AI(T,T) > ridge."""
    return int(4 * ridge_flop_per_byte) + 1

assert abs(b200_ridge(B200_PEAK_BF16_GFLOPS) - 140.625) < 1.0
assert b200_tile_breakeven(b200_ridge(B200_PEAK_BF16_GFLOPS)) > 560
print("[ok] B200 ridge (BF16): {:.0f} FLOP/byte".format(b200_ridge(B200_PEAK_BF16_GFLOPS)))
print("[ok] B200 tile break-even (BF16): T > {}".format(
      b200_tile_breakeven(b200_ridge(B200_PEAK_BF16_GFLOPS))))


import pathlib

if __name__ == "__main__":
    p = pathlib.Path("benchmarks/blackwell_analysis.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_BLACKWELL_ANALYSIS)
    print(f"[ok] Blackwell co-design analysis → {p}")
    print(f"     BF16 ridge on B200: {b200_ridge(B200_PEAK_BF16_GFLOPS):.0f} FLOP/byte")
    print(f"     Break-even tile (BF16): T > {b200_tile_breakeven(b200_ridge(B200_PEAK_BF16_GFLOPS))}")
    print(f"     Conclusion: single-CTA 128×128 tiles are bandwidth-bound on B200 BF16.")
    print(f"     Structural change required: WGMMA cluster tiles (new abstraction in p08).")
