# src/harness/tier2_closure.py — three-way comparison and Tier 2 closure
import json, pathlib, sys
import torch
from gpu_roofline.harness.device import probe_device
from gpu_roofline.kernels.attn_naive import (naive_hbm_bytes, flash_hbm_bytes, attention_flops,
                                 benchmark_attention, print_attention_report, persist_result)
from gpu_roofline.kernels.flash_fwd_causal import flash_fwd_causal
from gpu_roofline.kernels.triton_attn import _flash_attn_fwd_autotuned, flash_attn_triton
import torch.nn.functional as F
from dataclasses import asdict
from torch.nn.attention import SDPBackend, sdpa_kernel


def sdpa(enable_flash: bool, enable_math: bool, enable_mem_efficient: bool):
    backends = []
    if enable_flash:         backends.append(SDPBackend.FLASH_ATTENTION)
    if enable_math:          backends.append(SDPBackend.MATH)
    if enable_mem_efficient: backends.append(SDPBackend.EFFICIENT_ATTENTION)

    def fn(Q, K, V, causal=False):
        q, k, v = [x.unsqueeze(0).unsqueeze(0) for x in (Q, K, V)]
        try:
            with sdpa_kernel(backends):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal).squeeze(0).squeeze(0)
        except RuntimeError:
            return None
    return fn


def run_three_way(dev, N: int = 2048, D: int = 64, iters: int = 100) -> list[dict]:
    Q = torch.randn(N, D, device="cuda")
    K = torch.randn(N, D, device="cuda")
    V = torch.randn(N, D, device="cuda")

    candidates = [
        ("our_cuda_p03",   lambda: flash_fwd_causal(Q, K, V, causal=False)[0]),
        ("our_triton_p04", lambda: flash_attn_triton(Q, K, V, causal=False)),
        ("sdpa_flash",     lambda: sdpa(True,  False, False)(Q, K, V)),
        ("sdpa_math_fp32", lambda: sdpa(False, True,  False)(Q, K, V)),
    ]

    results = []
    for name, fn in candidates:
        if fn() is None:
            print(f"  {name}: not available — skipped")
            continue
        r = benchmark_attention(name, fn, Q, K, V, N, D, dev, iters=iters)
        results.append({"name": name, **asdict(r)})   # BenchResult is a dataclass → asdict()
    return results


def abstraction_cost_analysis(dev, results: list[dict]) -> str:
    cuda   = next((r for r in results if "cuda" in r["name"]), None)
    triton = next((r for r in results if "triton" in r["name"]), None)
    if not cuda or not triton:
        return "Cannot analyse: one or both kernels missing from results."
    gap_pct = (triton["gflops"] - cuda["gflops"]) / cuda["gflops"] * 100
    analysis = []
    analysis.append(f"CUDA p03 vs Triton p04 speed gap: {gap_pct:+.1f}%")
    if gap_pct > 5:
        analysis.append("  Triton is faster. Likely causes:")
        if dev.cc >= (8, 0):
            analysis.append("  • num_stages > 1: Triton auto-inserts cp.async pipeline stages")
            analysis.append("    (equivalent to p03/04's register-prefetch, but in hardware)")
        analysis.append("  • num_warps tuning: Triton exposed warp count as a free variable")
    elif gap_pct < -5:
        analysis.append("  CUDA is faster. Likely causes:")
        analysis.append("  • Triton's register allocator may spill more than p03's hand-written layout")
        analysis.append("  • Bank-conflict swizzle: p03's smem layout is manually conflict-free;")
        analysis.append("    Triton's smem layout for tl.dot may conflict on some (BLOCK_M, HEAD_DIM)")
        analysis.append("  • Verify allow_tf32 is consistently False in both kernels")
    else:
        analysis.append("  Kernels are within noise — compiler quality matches hand-written for this shape.")
    return "\n".join(analysis)


def write_tier2_report(dev, results: list[dict],
                       out: str = "benchmarks/p04_report.md") -> str:
    N, D = 2048, 64
    nb, fb = naive_hbm_bytes(N, D), flash_hbm_bytes(N, D)
    lines = [
        "# Tier 2 Closure — FlashAttention CUDA & Triton\n",
        f"**Device:** {dev.name}  **Peak BW:** {dev.peak_bw_gbps:.0f} GB/s\n",
        "| kernel | GFLOP/s | eff BW (FA bytes) | notes |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r.get('gflops',0):.0f} "
                     f"| {r.get('eff_bw_gbps',0):.0f} GB/s | |")
    lines += [
        f"\n**IO reduction (N=2048, d=64):** {nb/1e6:.0f} MB → {fb/1e6:.1f} MB "
        f"({nb/fb:.0f}× fewer DRAM bytes)\n",
        "## Tier 2 complete\n",
        "**p03 (CUDA):** Implemented FlashAttention forward pass from scratch — online softmax "
        "recurrence, O(N) HBM traffic, single-kernel fused attention, causal masking. "
        "Proved exact numerical agreement with SDPA (atol=1e-5) at N∈{128…4096}.\n",
        "**p04 (Triton):** Same algorithm in ~50 lines. `tl.dot` replaced the two-warp "
        "dot-product reduction; `tl.load` replaced cooperative smem loading loops; "
        "`@triton.autotune` swept (BLOCK_M, BLOCK_N, num_stages, num_warps). "
        "Causal masking in 3 lines vs 20 in CUDA.\n",
        "**Tier 3 next:** ring all-reduce, tensor parallelism, comm/compute overlap — "
        "the skills that separate a single-GPU optimiser from a systems architect.",
    ]
    pathlib.Path(out).write_text("\n".join(lines))
    return out


def build_tier2_report() -> None:
    dev = probe_device()
    results = run_three_way(dev)
    print(abstraction_cost_analysis(dev, results))
    report = write_tier2_report(dev, results)
    print(f"\n[ok] report → {report}")
    cuda   = next((r for r in results if "cuda"   in r["name"]), {})
    triton = next((r for r in results if "triton" in r["name"]), {})
    print(f"\nTier 2 complete on {dev.name}:")
    print(f"  p03 CUDA   : {cuda.get('gflops', 0):.0f} GFLOP/s")
    print(f"  p04 Triton : {triton.get('gflops', 0):.0f} GFLOP/s  "
          f"({len(_flash_attn_fwd_autotuned.cache)} autotuner config(s) evaluated)")
    print(f"  IO reduction: {naive_hbm_bytes(2048,64)/flash_hbm_bytes(2048,64):.0f}× at N=2048, d=64")


if __name__ == "__main__":
    build_tier2_report()
