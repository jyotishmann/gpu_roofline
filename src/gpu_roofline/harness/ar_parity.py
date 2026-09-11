# src/harness/ar_parity.py — comparison runner and project closure
import sys
from gpu_roofline.harness.device import probe_device
from gpu_roofline.comms.overlap import overlap_demo, compare_to_nccl, write_ar_report


def build_project05_report() -> None:
    dev = probe_device()
    overlap_demo()
    nccl = compare_to_nccl()
    if nccl:
        from gpu_roofline.comms.naive_allreduce import persist_ar_result
        persist_ar_result(nccl)
    report = write_ar_report(dev)
    print(f"\n[ok] report → {report}")
    print(f"\nProject 05 complete on {dev.name}.")
    print(f"  Ring all-reduce: bandwidth-optimal (2(N-1)/N × data, → 2× for large N).")
    print(f"  Comm/compute overlap: CUDA streams enable DDP-style bucketing.")
    print(f"  NCCL gap: {'measured (see report)' if nccl else 'requires multi-GPU (noted in report)'}.")
    print(f"\nTier 3 continues: p06 — Tensor Parallelism "
          f"(column+row parallel linear layer using the ring all-reduce).")


if __name__ == "__main__":
    build_project05_report()
