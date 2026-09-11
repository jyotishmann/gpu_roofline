# src/layers/tp_linear.py — tensor-parallel linear layers
import torch
import torch.nn as nn
import sys



def partition_weight_column(W: torch.Tensor, N: int) -> list[torch.Tensor]:
    """Split W [d, h] into N column-shards [d, h//N]."""
    d, h = W.shape
    assert h % N == 0, f"out_features {h} must be divisible by N={N}"
    chunk = h // N
    return [W[:, i*chunk:(i+1)*chunk].clone() for i in range(N)]


class ColumnParallelLinear(nn.Module):
    """
    Worker i computes Y_i = X @ W_col_i. No all-reduce needed (outputs tile).
    X is the FULL input (replicated on all workers).
    Y_i is a SHARD of the output: [B, h//N].
    """
    def __init__(self, in_features: int, out_features: int,
                 N: int, worker_id: int, W_full: torch.Tensor | None = None):
        super().__init__()
        assert out_features % N == 0
        chunk = out_features // N
        if W_full is not None:
            shard = partition_weight_column(W_full, N)[worker_id]
        else:
            shard = torch.randn(in_features, chunk) * (in_features ** -0.5)
        self.weight = nn.Parameter(shard)          # [d, h//N] — this worker's column shard
        self.N, self.worker_id, self.chunk = N, worker_id, chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight                      # [B, d] @ [d, h//N] → [B, h//N]; no comm


def assert_column_parallel_correct(N: int = 4, B: int = 8,
                                    d: int = 64, h: int = 256) -> None:
    torch.manual_seed(0)
    W_full = torch.randn(d, h, device="cuda")
    X      = torch.randn(B, d, device="cuda")
    ref    = X @ W_full                           # single-GPU reference [B, h]

    workers = [ColumnParallelLinear(d, h, N, i, W_full).cuda() for i in range(N)]
    shards  = [w(X) for w in workers]             # [B, h//N] for each worker
    got     = torch.cat(shards, dim=-1)            # [B, h]

    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5), \
        f"column-parallel mismatch: max err {(got-ref).abs().max():.2e}"
    print(f"[ok] column-parallel: cat(Y_0..Y_{N-1}) == X@W  (N={N}, B={B}, d={d}, h={h})")
    print(f"     No all-reduce issued — outputs tile trivially.")


if __name__ == "__main__":
    assert_column_parallel_correct(N=2, B=8, d=64, h=128)
    assert_column_parallel_correct(N=4, B=8, d=64, h=256)
    assert_column_parallel_correct(N=8, B=4, d=128, h=512)
    print("\nColumn-parallel correctness verified for N=2,4,8.")
    print("Next: row-parallel (p06/02) needs an all-reduce to combine partial products.")
