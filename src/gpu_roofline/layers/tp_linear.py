# src/layers/tp_linear.py — tensor-parallel linear layers
import torch
import torch.nn as nn
import sys
from gpu_roofline.comms.ring_allreduce import reduce_scatter_ring, all_gather_ring


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


def sum_allreduce(partial_outputs: torch.Tensor) -> torch.Tensor:
    """
    All-reduce without dividing by N: returns the SUM, not the mean.
    partial_outputs: [N, B, h] — worker i's partial product in row i.
    After call: every row holds the correct sum (not mean).
    """
    N = partial_outputs.shape[0]
    streams = [torch.cuda.Stream() for _ in range(N)]
    # Flatten to [N, B*h] for the ring implementation, which expects [N, P]
    flat = partial_outputs.view(N, -1).contiguous()
    reduce_scatter_ring(flat, streams)              # phase 1: accumulate per-chunk
    # Note: reduce_scatter does NOT divide; ring_allreduce's /= N is in ring_allreduce(),
    # not in the phases separately. So here we call phases directly and skip /= N.
    all_gather_ring(flat, streams)                  # phase 2: distribute reduced chunks
    return flat.view_as(partial_outputs)            # [N, B, h] — all rows equal the sum


def partition_weight_row(W: torch.Tensor, N: int) -> list[torch.Tensor]:
    """Split W [d, h] into N row-shards [d//N, h]."""
    d, h = W.shape
    assert d % N == 0
    chunk = d // N
    return [W[i*chunk:(i+1)*chunk, :].clone() for i in range(N)]


class RowParallelLinear(nn.Module):
    """
    Worker i computes Y_partial_i = X_i @ W_row_i, then all-reduce gives full Y.
    X_i is a SHARD of input: [B, d//N].  W_row_i is [d//N, h].
    After all-reduce, every worker holds the full Y: [B, h].
    """
    def __init__(self, in_features: int, out_features: int,
                 N: int, worker_id: int, W_full: torch.Tensor | None = None):
        super().__init__()
        assert in_features % N == 0
        chunk = in_features // N
        if W_full is not None:
            shard = partition_weight_row(W_full, N)[worker_id]
        else:
            shard = torch.randn(chunk, out_features) * (in_features ** -0.5)
        self.weight = nn.Parameter(shard)           # [d//N, h]
        self.N, self.worker_id, self.chunk = N, worker_id, chunk

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        """x_i: [B, d//N] — this worker's input shard. Returns full Y: [B, h]."""
        return x_i @ self.weight                    # [B, d//N] @ [d//N, h] → [B, h] partial


def assert_row_parallel_correct(N: int = 4, B: int = 8,
                                 d: int = 64, h: int = 128) -> None:
    torch.manual_seed(1)
    W_full = torch.randn(d, h, device="cuda")
    X      = torch.randn(B, d, device="cuda")
    ref    = X @ W_full                             # [B, h]

    workers    = [RowParallelLinear(d, h, N, i, W_full).cuda() for i in range(N)]
    chunk      = d // N
    partials   = [workers[i](X[:, i*chunk:(i+1)*chunk]) for i in range(N)]
    stacked    = torch.stack(partials)              # [N, B, h]
    sum_allreduce(stacked)                          # in-place: every row → sum
    for i in range(N):
        assert torch.allclose(stacked[i], ref, atol=1e-5), \
            f"worker {i} wrong after AR: max err {(stacked[i]-ref).abs().max():.2e}"
    print(f"[ok] row-parallel: all {N} workers hold full Y after 1 all-reduce  "
          f"(N={N}, B={B}, d={d}, h={h})")

if __name__ == "__main__":
    assert_column_parallel_correct(N=2, B=8, d=64, h=128)
    assert_column_parallel_correct(N=4, B=8, d=64, h=256)
    assert_column_parallel_correct(N=8, B=4, d=128, h=512)
    print("\nColumn-parallel correctness verified for N=2,4,8.")
    print("Next: row-parallel needs an all-reduce to combine partial products.")
    assert_column_parallel_correct()   # regression
    assert_row_parallel_correct(N=2, B=8, d=64, h=128)
    assert_row_parallel_correct(N=4, B=8, d=64, h=256)
    assert_row_parallel_correct(N=8, B=4, d=128, h=512)
    print("\nBoth layer types verified. Next: chain them for the full MLP block.")
