# src/harness/roofline.py — assemble the roofline + report.

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend: write the PNG without needing a display
import matplotlib.pyplot as plt

import json
import pathlib
from types import SimpleNamespace

def attainable_gflops(ai: float, peak_bw_gbps: float, peak_fp32_gflops: float) -> float:
    """The roofline ceiling at a given arithmetic intensity (GFLOP/s)."""
    return min(peak_fp32_gflops, ai * peak_bw_gbps)  # min(compute roof, memory roof) — the two-roof model


def load_persisted(bench_dir: str = "benchmarks"):
    """Read device ceilings + kernel/transfer results from disk; no GPU or re-run needed."""
    d = pathlib.Path(bench_dir)
    dev_path, res_path = d / "device_info.json", d / "p00_results.json"
    if not dev_path.exists():
        raise FileNotFoundError("benchmarks/device_info.json missing — run p00/01 (harness.device) first")
    if not res_path.exists():
        raise FileNotFoundError("benchmarks/p00_results.json missing — run p00/03–06 benchmarks first")
    dev_d = json.loads(dev_path.read_text())
    dev_d["cc"] = tuple(dev_d["cc"])                   # JSON list → tuple, inverse of the 1.8 serialisation
    return SimpleNamespace(**dev_d), json.loads(res_path.read_text())


def setup_roofline(dev):
    ai = np.logspace(-2, 3, 500)
    roof = np.minimum(ai * dev.peak_bw_gbps, dev.peak_fp32_gflops)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.loglog(ai, roof, "k-", lw=2.2,
              label=f"theoretical roof · {dev.peak_bw_gbps:.0f} GB/s · {dev.peak_fp32_gflops/1e3:.1f} TFLOP/s")
    ax.axvline(dev.ridge_flop_per_byte, color="gray", ls=":", lw=1.2,
               label=f"ridge · {dev.ridge_flop_per_byte:.1f} FLOP/byte")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("attainable performance (GFLOP/s)")
    ax.set_title(f"Roofline — {dev.name}")
    ax.grid(True, which="both", alpha=0.25)
    return fig, ax


def plot_kernels(fig, ax, dev, results, out: str = "benchmarks/roofline.png"):
    by_name = {r["name"]: r for r in results}
    copy = by_name.get("copy_buf")
    if copy:                                           # copy (AI=0) can't be a point — it sets the empirical roof
        ai = np.logspace(-2, 3, 500)
        ax.loglog(ai, ai * copy["eff_bw_gbps"], "k--", lw=1.3,
                  label=f"achieved-BW roof · {copy['eff_bw_gbps']:.0f} GB/s "
                        f"({copy['eff_bw_gbps']/dev.peak_bw_gbps*100:.0f}% of peak)")
    for name in ("vector_add", "saxpy", "vector_add_tuned"):
        r = by_name.get(name)
        if r and r.get("arithmetic_intensity", 0) > 0:
            ax.plot(r["arithmetic_intensity"], r["gflops"], "o", ms=8)
            ax.annotate(f"{name}\n{r['pct_peak_bw']:.0f}% BW",
                        (r["arithmetic_intensity"], r["gflops"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    return out


def write_report(dev, results, png_path: str, out: str = "benchmarks/p00_report.md") -> str:
    by = {r["name"]: r for r in results}
    def bw(n): return by[n]["pct_peak_bw"] if n in by else float("nan")
    transfers = [r for r in results if r["name"].startswith("transfer_")]
    best_pcie = max((r["bw_gbps"] for r in transfers), default=float("nan"))
    lines = [                                          # report is pure string-assembly from persisted data
        "# Project 00 — Roofline Report\n",
        f"**Device:** {dev.name} (sm_{dev.cc[0]}{dev.cc[1]}, {dev.sm_count} SMs)  ",
        f"**Peak bandwidth:** {dev.peak_bw_gbps:.0f} GB/s **· Peak FP32:** "
        f"{dev.peak_fp32_gflops/1e3:.1f} TFLOP/s **· Ridge:** {dev.ridge_flop_per_byte:.1f} FLOP/byte\n",
        f"![roofline]({pathlib.Path(png_path).name})\n",
        "## 1. Every kernel is bandwidth-bound",
        f"Arithmetic intensities (copy 0, vector-add 0.083, SAXPY 0.167) sit far left of the "
        f"ridge at {dev.ridge_flop_per_byte:.1f} FLOP/byte. None can use more than a sliver of the "
        f"FP32 units — the correct, expected result for elementwise kernels.\n",
        "## 2. Measured vs theoretical bandwidth",
        f"vector-add reached {bw('vector_add'):.0f}% of theoretical peak; SAXPY {bw('saxpy'):.0f}%. "
        f"The ~10–25% shortfall is the achievable-bandwidth wall: DRAM refresh, row-activation timing, "
        f"ECC, and read/write turnaround — not measurement error. ~80% of theoretical is success here.\n",
        "## 3. The ridge is Tier 1's mission",
        "To become compute-bound, a kernel must raise arithmetic intensity past the ridge, which "
        "requires data reuse — reading each byte once and doing many FLOPs on it. That is exactly "
        "what tiling in SGEMM achieves, and why Tier 1 exists.\n",
        "## 4. PCIe is the hidden ceiling",
        f"The fastest host↔device transfer measured was {best_pcie:.0f} GB/s — about "
        f"{dev.peak_bw_gbps/best_pcie:.0f}× below DRAM. A full training step is data-movement-bound "
        f"unless transfers overlap with compute (the seed of Tier 3).\n",
    ]
    pathlib.Path(out).write_text("\n".join(lines))
    return out

def build_project00_report(bench_dir: str = "benchmarks") -> None:
    """Regenerate the full Project-00 deliverable (plot + report) from persisted data."""
    dev, results = load_persisted(bench_dir)
    fig, ax = setup_roofline(dev)
    png = plot_kernels(fig, ax, dev, results)
    report = write_report(dev, results, png)
    print(f"[ok] roofline → {png}")
    print(f"[ok] report   → {report}")
    print(f"\nProject 00 complete on {dev.name}: all primitives bandwidth-bound, "
          f"ridge at {dev.ridge_flop_per_byte:.1f} FLOP/byte. Tier 1 next → raise arithmetic intensity.")


if __name__ == "__main__":
    build_project00_report()
