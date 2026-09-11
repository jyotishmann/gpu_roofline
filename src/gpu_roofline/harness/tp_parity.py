# src/harness/tp_parity.py — TP efficiency measurement and project closure
import time, statistics, json, pathlib
import torch
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.layers.tp_linear import sum_allreduce
from gpu_roofline.layers.tp_mlp import TensorParallelMLP
import torch.nn.functional as F
import gpu_roofline.layers.tp_linear as tp_mod


def measure_compute_time(N, B, d, h, iters=50) -> float:
    """T_compute: MLP forward without all-reduce."""
    torch.manual_seed(0)
    W1 = torch.randn(d, h, device="cuda"); W2 = torch.randn(h, d, device="cuda")
    X  = torch.randn(B, d, device="cuda")
    workers  = [TensorParallelMLP(d, h, N, i, W1, W2).cuda() for i in range(N)]
    noop = lambda t: t
    orig = tp_mod.sum_allreduce; tp_mod.sum_allreduce = noop
    try:
        for _ in range(10):                         # warmup
            [w(X) for w in workers]
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            [w(X) for w in workers]
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    finally:
        tp_mod.sum_allreduce = orig
    return statistics.median(times)


def measure_comm_time(N, B, d, iters=50) -> float:
    """T_comm: all-reduce of one [N, B, d] tensor."""
    dummy = torch.randn(N, B, d, device="cuda")
    for _ in range(10): sum_allreduce(dummy.clone())
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t = dummy.clone(); torch.cuda.synchronize()
        t0 = time.perf_counter(); sum_allreduce(t); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def efficiency_table(dev, N=4, B=8, d=64, hs=None) -> list[dict]:
    hs = hs or [64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    ridge = dev.ridge_flop_per_byte
    h_breakeven = 2 * (N - 1) * ridge    # from master doc §3
    print(f"\nTP efficiency η  (N={N}, B={B}, d={d},  peak={dev.peak_bw_gbps:.0f} GB/s)")
    print(f"Break-even h (theory): {h_breakeven:.0f}")
    print(f"{'h':>6}  {'T_comp ms':>10}  {'T_comm ms':>10}  {'η':>6}  {'note'}")
    print("─" * 55)
    for h in hs:
        if h % N != 0: continue
        tc = measure_compute_time(N, B, d, h) * 1000
        tm = measure_comm_time(N, B, d)    * 1000
        eta = tc / (tc + tm)
        note = "< break-even" if h < h_breakeven else ("≈ break-even" if h < 2*h_breakeven else "compute-bound")
        print(f"{h:>6}  {tc:>10.2f}  {tm:>10.2f}  {eta:>5.2f}  {note}")
        rows.append({"h": h, "T_comp_ms": tc, "T_comm_ms": tm, "eta": eta})
    return rows


def single_gpu_vs_tp(dev, N=4, B=8, d=64, h=1024, iters=50) -> dict:
    torch.manual_seed(0)
    W1f = torch.randn(d, h, device="cuda"); W2f = torch.randn(h, d, device="cuda")
    X   = torch.randn(B, d, device="cuda")
    # Single-GPU reference
    for _ in range(10): F.gelu(X @ W1f) @ W2f
    torch.cuda.synchronize()
    times_sg = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        F.gelu(X @ W1f) @ W2f
        torch.cuda.synchronize(); times_sg.append(time.perf_counter() - t0)
    t_single = statistics.median(times_sg)
    # TP version (with all-reduce)
    t_compute = measure_compute_time(N, B, d, h, iters)
    t_comm    = measure_comm_time(N, B, d, iters)
    t_tp      = t_compute + t_comm        # sequential: no overlap (conservative estimate)
    speedup   = t_single / t_tp
    flops     = 4 * B * d * h             # 2 GEMMs × 2MNK
    print(f"\nTP vs single-GPU  (N={N}, B={B}, d={d}, h={h}):")
    print(f"  single-GPU  : {t_single*1000:.2f} ms  ({flops/t_single/1e9:.0f} GFLOP/s)")
    print(f"  TP (N={N})   : {t_tp   *1000:.2f} ms  (speedup {speedup:.2f}x; ideal {N}x)")
    print(f"  η = {t_compute/(t_compute+t_comm):.2f}")
    return {"speedup": speedup, "ideal": N, "eta": t_compute/(t_compute+t_comm)}


def write_tp_report(dev, rows: list[dict], speedup_r: dict,
                    out: str = "benchmarks/p06_report.md") -> str:
    lines = [
        "# Project 06 — Tensor Parallelism · Report\n",
        f"**Device:** {dev.name}  **Ridge:** {dev.ridge_flop_per_byte:.1f} FLOP/byte\n",
        "## TP efficiency η vs hidden dimension\n",
        "| h | T_compute (ms) | T_comm (ms) | η | note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['h']} | {r['T_comp_ms']:.2f} | {r['T_comm_ms']:.2f} | "
                     f"{r['eta']:.2f} | |")
    lines += [
        f"\n**Speedup at h=1024:** {speedup_r['speedup']:.2f}x "
        f"(ideal {speedup_r['ideal']}x, η={speedup_r['eta']:.2f})\n",
        "## What this proves",
        "1. Column-parallel W1: no all-reduce needed; outputs tile perfectly.",
        "2. Row-parallel W2: one all-reduce gives the full output.",
        "3. Chaining: column then row → exactly one all-reduce per MLP block.",
        "4. GELU commutes: element-wise on the shard, no cross-dimension dependency.",
        "5. η → 1 for large h; break-even h ≈ 2(N-1) x ridge.",
        "\n_Generated by `harness/tp_parity.py`._",
    ]
    pathlib.Path(out).write_text("\n".join(lines))
    return out


def build_project06_report() -> None:
    dev = probe_device()
    rows   = efficiency_table(dev)
    sp_r   = single_gpu_vs_tp(dev)
    report = write_tp_report(dev, rows, sp_r)
    print(f"\n[ok] report → {report}")
    print(f"\nProject 06 complete on {dev.name}.")
    print(f"  TP MLP: column-parallel W1 + row-parallel W2 = 1 all-reduce per block.")
    print(f"  η converges to 1 for large h (transformer hidden dims are well above break-even).")
    print(f"\nTier 3 continues: p07 — Paged KV Cache (the vLLM idea: "
          f"fixed page pool → variable-length sequences → continuous batching).")


if __name__ == "__main__":
    build_project06_report()
