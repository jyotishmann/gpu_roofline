# src/comms/naive_allreduce.py — naïve all-reduce + bandwidth model
import time, statistics, math
import torch
import sys
from gpu_roofline.harness.device import probe_device
import json, pathlib


def make_workers(N: int, P: int, device: str = "cuda") -> torch.Tensor:
    """
    N workers x P parameters. state[i] is worker i's gradient buffer.
    In simulation, different workers share the same DRAM; 'sending' is a tensor copy.
    """
    torch.manual_seed(0)
    return torch.randn(N, P, device=device)   # state[i] = gradient of worker i


def allreduce_bytes_naive(N: int, P: int) -> dict:
    """
    Bytes transferred ON EACH WORKER'S LINK during the naïve gather-broadcast.
    Worker 0 (root): receives (N-1)xPx4 bytes, sends (N-1)xPx4 bytes.
    Other workers   : send Px4 bytes (reduce), receive Px4 bytes (broadcast).
    Returns a dict of {worker_id: bytes_on_link} for the link-utilisation analysis.
    """
    per_worker = {0: 2 * (N - 1) * P * 4}        # root: both directions fully loaded
    for i in range(1, N):
        per_worker[i] = 2 * P * 4                  # leaf: send once, receive once
    return per_worker


def alpha_beta_time(S: int, alpha_us: float, bw_gbps: float) -> float:
    """Predicted transfer time (seconds) for S bytes: T = alpha + S/bandwidth."""
    return alpha_us * 1e-6 + S / (bw_gbps * 1e9)


def naive_allreduce(state: torch.Tensor) -> torch.Tensor:
    """
    In-place all-reduce: gather to worker 0, sum, broadcast.
    state[i] is worker i's gradient buffer (modified in-place).
    Returns state (all rows now equal to the per-element sum divided by N).
    """
    N, P = state.shape
    # Phase 1 — reduce to root (worker 0)
    for i in range(1, N):
        state[0] += state[i]          # worker i "sends" to root; root accumulates
    state[0] /= N                     # normalise (sum → mean)
    # Phase 2 — broadcast from root to all
    for i in range(1, N):
        state[i].copy_(state[0])      # root "sends" reduced gradient back to worker i
    return state


def assert_allreduce_correct(allreduce_fn, N: int = 4, P: int = 1024) -> None:
    """Correctness: all workers should hold the same tensor after all-reduce."""
    torch.manual_seed(42)
    originals = torch.randn(N, P, device="cuda")
    state = originals.clone()
    allreduce_fn(state)
    expected = originals.mean(dim=0)            # per-element mean across workers
    for i in range(N):
        assert torch.allclose(state[i], expected, atol=1e-5), \
            f"worker {i} disagrees: max err {(state[i]-expected).abs().max():.2e}"
    print(f"[ok] all {N} workers agree after allreduce  (P={P})")


def benchmark_allreduce(fn, dev, N: int, P: int, warmup: int = 5, iters: int = 50) -> dict:
    """Time an all-reduce operation; return per-worker bandwidth utilisation."""
    state_template = make_workers(N, P)
    for _ in range(warmup):
        state = state_template.clone()
        fn(state)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        state = state_template.clone()          # reset to original values
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(state)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med_s = statistics.median(times)
    bw_per_worker = allreduce_bytes_naive(N, P)   # bytes per worker link
    return {
        "median_s": med_s,
        "root_bw_gbps":   bw_per_worker[0] / med_s / 1e9,
        "leaf_bw_gbps":   bw_per_worker[1] / med_s / 1e9,
        "root_pct_peak":  bw_per_worker[0] / med_s / 1e9 / dev.peak_bw_gbps * 100,
        "leaf_pct_peak":  bw_per_worker[1] / med_s / 1e9 / dev.peak_bw_gbps * 100,
        "N": N, "P": P,
    }


def print_bandwidth_table(dev, Ns=(2, 4, 8), P: int = 1 << 22) -> None:
    """Print per-worker link utilisation for naive all-reduce at several N."""
    print(f"\nNaïve all-reduce bandwidth utilisation  (P={P//1024}K floats, peak={dev.peak_bw_gbps:.0f} GB/s)")
    print(f"{'N':>4}  {'root link':>12}  {'leaf link':>12}  {'root %':>8}  {'leaf %':>8}")
    print("─" * 54)
    for N in Ns:
        r = benchmark_allreduce(naive_allreduce, N, P, dev)
        print(f"{N:>4}  {r['root_bw_gbps']:>10.1f} GB/s  {r['leaf_bw_gbps']:>10.1f} GB/s  "
              f"{r['root_pct_peak']:>7.1f}%  {r['leaf_pct_peak']:>7.1f}%")
    print("\nNote: root link is saturated; leaf links are ~1/N of that. "
          "Ring (p05/03) equalises all workers.")


def persist_ar_result(r: dict, path: str = "benchmarks/p05_results.json") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else []
    data = [d for d in data if d.get("name") != r.get("name")] + [r]
    p.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    dev = probe_device()
    assert_allreduce_correct(naive_allreduce, N=4, P=1024)
    print_bandwidth_table(dev, Ns=[2, 4, 8], P=1 << 22)
    r = benchmark_allreduce(naive_allreduce, dev, N=4, P=1 << 22)
    r["name"] = "naive_allreduce"
    persist_ar_result(r)
    print(f"\nNaïve all-reduce baseline: {r['root_bw_gbps']:.0f} GB/s root, "
          f"{r['leaf_bw_gbps']:.0f} GB/s leaves.")
    print("Ring (p05/02-03) target: equalise ALL worker links near peak.")
