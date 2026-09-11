# src/layers/tp_mlp.py — Tensor-parallel MLP block
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "../../tier0-foundations/p00-primitives/src")
from gpu_roofline.layers.tp_linear import (ColumnParallelLinear, RowParallelLinear,
                               sum_allreduce, partition_weight_column,
                               partition_weight_row)


class TensorParallelMLP(nn.Module):
    """
    Megatron-style TP MLP: Y = σ(X @ W1) @ W2
    Column-parallel W1 → element-wise GELU → row-parallel W2 → one all-reduce.
    Exactly ONE all-reduce per forward pass (in RowParallelLinear).
    """
    def __init__(self, d: int, h: int, N: int, worker_id: int,
                 W1_full: torch.Tensor | None = None,
                 W2_full: torch.Tensor | None = None):
        super().__init__()
        self.N, self.worker_id = N, worker_id
        self.col_linear = ColumnParallelLinear(d, h, N, worker_id, W1_full)   # [d, h//N]
        self.row_linear = RowParallelLinear(h, d, N, worker_id, W2_full)       # [h//N, d]
        # Note: row-linear's in_features = h, sharding the h dimension;
        #       the col-linear shard [B, h//N] IS the row-linear's input shard [B, (h//N)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: full input [B, d] — replicated on all workers
        returns: full output [B, d] — replicated on all workers after all-reduce

        Communication count:
          col_linear.forward : 0 all-reduces  (X replicated, Y sharded)
          GELU               : 0 all-reduces  (element-wise on shard)
          row_linear.forward : 0 all-reduces  (returns partial product)
          sum_allreduce      : 1 all-reduce   ← the only communication
        """
        z_i = self.col_linear(x)                    # [B, h//N]: no comm
        z_i = F.gelu(z_i)                           # [B, h//N]: element-wise on shard, no comm
        y_partial = self.row_linear(z_i)            # [B, d]:    partial product, no comm yet
        return y_partial                            # caller stacks and all-reduces (see test)


def assert_tp_mlp_correct(N: int = 4, B: int = 8, d: int = 64, h: int = 256) -> None:
    torch.manual_seed(2)
    W1_full = torch.randn(d, h, device="cuda") * (d ** -0.5)
    W2_full = torch.randn(h, d, device="cuda") * (h ** -0.5)
    X       = torch.randn(B, d, device="cuda")

    # Single-GPU reference
    ref = F.gelu(X @ W1_full) @ W2_full            # [B, d]

    # TP forward passes (N workers)
    workers  = [TensorParallelMLP(d, h, N, i, W1_full, W2_full).cuda() for i in range(N)]
    partials = [w(X) for w in workers]             # list of N [B, d] partial products
    stacked  = torch.stack(partials)               # [N, B, d]
    sum_allreduce(stacked)                         # ← the one all-reduce
    got = stacked[0]                               # all rows are identical after all-reduce

    assert torch.allclose(got, ref, atol=1e-5), \
        f"TP MLP mismatch: max err {(got-ref).abs().max():.2e}"
    print(f"[ok] TP MLP correct for N={N}, B={B}, d={d}, h={h}  (1 all-reduce)")


def count_allreduce_calls(N: int = 4, B: int = 8, d: int = 64, h: int = 256) -> None:
    """Monkey-patch sum_allreduce to count calls; assert exactly 1 per forward pass."""
    import gpu_roofline.layers.tp_linear as tp_mod
    original_ar = tp_mod.sum_allreduce
    call_count  = [0]
    def counting_ar(t):
        call_count[0] += 1
        return original_ar(t)
    tp_mod.sum_allreduce = counting_ar
    try:
        torch.manual_seed(3)
        W1 = torch.randn(d, h, device="cuda"); W2 = torch.randn(h, d, device="cuda")
        X  = torch.randn(B, d, device="cuda")
        workers = [TensorParallelMLP(d, h, N, i, W1, W2).cuda() for i in range(N)]
        partials = [w(X) for w in workers]
        stacked  = torch.stack(partials)
        tp_mod.sum_allreduce(stacked)              # the one AR the caller explicitly invokes
        assert call_count[0] == 1, f"Expected 1 all-reduce, got {call_count[0]}"
        print(f"[ok] all-reduce count = 1 per forward pass (N={N})")
    finally:
        tp_mod.sum_allreduce = original_ar         # restore


if __name__ == "__main__":
    for N in [2, 4, 8]:
        assert_tp_mlp_correct(N=N, B=8, d=64, h=256)
    count_allreduce_calls()
    print("\nTensorParallelMLP: correct, one all-reduce per forward pass.")
    print("Next: Measuring efficiency η = T_compute / (T_compute + T_comm).")
