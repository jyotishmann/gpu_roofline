# src/kernels/attn_naive.py — naïve attention baseline
import contextlib
import torch
import torch.nn.functional as F

import json
import pathlib
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.harness.timing import time_kernel
import statistics


@contextlib.contextmanager
def _tf32_off():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def reference_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                   causal: bool = False) -> torch.Tensor:
    """SDPA math backend — honest FP32, deterministic, no Flash/mem-efficient routing."""
    q = Q.unsqueeze(0).unsqueeze(0)  # [1, 1, N, d] — SDPA expects 4D
    k = K.unsqueeze(0).unsqueeze(0)
    v = V.unsqueeze(0).unsqueeze(0)
    with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.squeeze(0).squeeze(0)   # back to [N, d]


def assert_attention_correct(attn_fn, causal: bool = False,
                              atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Correctness gate: allclose vs SDPA math backend with TF32 off."""
    torch.manual_seed(0)
    cases = [
        (128,  64,  "small  (N=128,  d=64)"),
        (512,  64,  "medium (N=512,  d=64)"),
        (2048, 64,  "large  (N=2048, d=64)"),
        (128,  128, "wide   (N=128,  d=128)"),
    ]
    with _tf32_off():
        for N, d, label in cases:
            Q = torch.randn(N, d, device="cuda")
            K = torch.randn(N, d, device="cuda")
            V = torch.randn(N, d, device="cuda")
            got = attn_fn(Q, K, V, causal=causal)
            ref = reference_sdpa(Q, K, V, causal=causal)
            if not torch.allclose(got, ref, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"attention wrong on '{label}': max err "
                    f"{(got - ref).abs().max().item():.2e}  (atol={atol})")
            print(f"[ok] {label}")


def naive_hbm_bytes(N: int, d: int) -> int:
    """Bytes read+written by the three-pass naïve attention (x4 for float32)."""
    return 4 * (3 * N * N + 4 * N * d)   # 3N² for the N×N intermediates, 4Nd for Q/K/V/O


def flash_hbm_bytes(N: int, d: int) -> int:
    """Theoretical HBM minimum: read Q,K,V once + write O once (FlashAttention)."""
    return 4 * 4 * N * d                  # 4 arrays × N × d × 4 bytes


def io_ratio(N: int, d: int) -> float:
    """How many x more DRAM bytes naïve uses vs the FlashAttention minimum."""
    return naive_hbm_bytes(N, d) / flash_hbm_bytes(N, d)


def attention_flops(N: int, d: int) -> int:
    """FLOPs for the two GEMMs (QKᵀ and PV); softmax ops are O(N²) but small."""
    return 4 * N * N * d                  # 2×N²×d for QKᵀ + 2×N²×d for PV


def print_io_model(N: int, d: int) -> None:
    nb, fb = naive_hbm_bytes(N, d), flash_hbm_bytes(N, d)
    print(f"N={N}, d={d}:")
    print(f"  naïve HBM traffic  : {nb/1e6:.1f} MB  (3N²+4Nd x 4 bytes)")
    print(f"  FlashAttention min : {fb/1e6:.1f} MB  (4Nd x 4 bytes)")
    print(f"  IO ratio           : {io_ratio(N,d):.1f}x  "
          f"(≈ 3N/4d = {3*N/(4*d):.1f} for large N)")


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = False,
                    scale: float | None = None) -> torch.Tensor:
    """
    Three-pass naïve attention. Materialises S[N,N] and P[N,N] in DRAM.
    Primary educational purpose: making the O(N²) DRAM cost visible and measurable.
    """
    N, d = Q.shape
    scale = scale or (d ** -0.5)
    S = torch.mm(Q * scale, K.t())          # [N, N] written to DRAM — cost: N² floats
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)             # [N, N] written to DRAM — cost: N² floats
    O = torch.mm(P, V)                       # [N, d] — uses both P and V from DRAM
    return O


def benchmark_attention(name: str, attn_fn, Q, K, V,
                        N: int, d: int, dev,
                        warmup: int = 10, iters: int = 50) -> dict:
    """Benchmark an attention implementation; report both HBM-traffic and GFLOP/s metrics."""
    samples = []
    for _ in range(warmup):
        attn_fn(Q, K, V)
    torch.cuda.synchronize()
    for _ in range(iters):
        t = time_kernel(lambda: attn_fn(Q, K, V))
        samples.append(t)
    med_ms = statistics.median(samples)
    nb    = naive_hbm_bytes(N, d)
    fb    = flash_hbm_bytes(N, d)
    flops = attention_flops(N, d)
    return {
        "name":          name,
        "N": N, "d": d,
        "median_ms":     med_ms,
        "naive_hbm_mb":  nb / 1e6,
        "flash_hbm_mb":  fb / 1e6,
        "io_ratio":      io_ratio(N, d),
        "naive_bw_gbps": nb / (med_ms * 1e-3) / 1e9,     # GB/s against naïve bytes
        "pct_peak_bw":   nb / (med_ms * 1e-3) / 1e9 / dev.peak_bw_gbps * 100,
        "gflops":        flops / (med_ms * 1e-3) / 1e9,
        "pct_peak_fp32": flops / (med_ms * 1e-3) / 1e9 / dev.peak_fp32_gflops * 100,
    }


def print_attention_report(r: dict, dev) -> None:
    print(f"\n{r['name']}  (N={r['N']}, d={r['d']})")
    print(f"  time (median)    : {r['median_ms']:.2f} ms")
    print(f"  GFLOP/s          : {r['gflops']:.0f}  ({r['pct_peak_fp32']:.1f}% peak FP32)")
    print(f"  naïve HBM bytes  : {r['naive_hbm_mb']:.1f} MB"
          f"  @ {r['naive_bw_gbps']:.0f} GB/s  ({r['pct_peak_bw']:.0f}% peak BW)")
    print(f"  FA minimum       : {r['flash_hbm_mb']:.1f} MB  "
          f"→ IO ratio {r['io_ratio']:.0f}x  ← the multiplier FlashAttention removes")
    print(f"  peak BW (device) : {dev.peak_bw_gbps:.0f} GB/s  "
          f"(every byte above {r['flash_hbm_mb']:.0f} MB is algorithmic waste)")


def persist_result(r: dict, path: str = "benchmarks/p03_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    data = [d for d in data if d.get("name") != r["name"]] + [r]
    p.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    dev = probe_device()

    # Show IO model growth with N
    print("IO complexity model  (d=64, float32 x 4 bytes):")
    print(f"{'N':>6}  {'naïve MB':>10}  {'FA MB':>7}  {'ratio':>7}")
    print("─" * 38)
    for N in [128, 512, 1024, 2048, 4096]:
        nb, fb = naive_hbm_bytes(N, 64), flash_hbm_bytes(N, 64)
        print(f"{N:>6}  {nb/1e6:>10.1f}  {fb/1e6:>7.2f}  {io_ratio(N,64):>7.1f}×")

    # Correctness + floor benchmark
    assert_attention_correct(naive_attention, causal=False)
    assert_attention_correct(naive_attention, causal=True)

    N, d = 2048, 64
    Q = torch.randn(N, d, device="cuda")
    K = torch.randn(N, d, device="cuda")
    V = torch.randn(N, d, device="cuda")
    r = benchmark_attention("naive_attention", naive_attention, Q, K, V, N, d, dev)
    print_attention_report(r, dev)
    persist_result(r)
    print(f"\nFloor established. FlashAttention (p03/03) targets <{r['flash_hbm_mb']:.0f} MB HBM traffic.")
