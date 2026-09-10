# src/comms/ring_allreduce.py — ring all-reduce (reduce-scatter + all-gather)
import torch


def reduce_scatter_ring(state: torch.Tensor, streams: list) -> torch.Tensor:
    """
    Phase 1 of ring all-reduce: N-1 steps, each worker accumulates one received chunk.
    After this, state[i][chunk_i] holds the correct sum of chunk i across all workers.
    state: [N, P] — worker i's gradient in state[i]
    streams: list of N CUDA streams (one per worker; allows concurrent copies)
    """
    N, P = state.shape
    assert P % N == 0, "P must be divisible by N for equal chunk sizes"
    chunk = P // N                                # elements per chunk
    recv_buf = state.clone()                      # recv buffer (copy, not alias)

    for s in range(N - 1):
        # Each worker i: send chunk (i-s)%N to right, receive chunk (i-s-1)%N from left
        for i in range(N):
            send_chunk = (i - s) % N
            recv_chunk = (i - s - 1) % N
            src  = (i - 1) % N                   # left neighbour
            # async copy from left neighbour's send-chunk into our recv buffer
            with torch.cuda.stream(streams[i]):
                recv_buf[i, recv_chunk*chunk:(recv_chunk+1)*chunk].copy_(
                    state[src, recv_chunk*chunk:(recv_chunk+1)*chunk],
                    non_blocking=True)
        # Synchronise all streams before accumulation (all receives must complete first)
        for i in range(N):
            streams[i].synchronize()
        # Accumulate: state[i][recv_chunk] += recv_buf[i][recv_chunk]
        for i in range(N):
            recv_chunk = (i - s - 1) % N
            state[i, recv_chunk*chunk:(recv_chunk+1)*chunk] += \
                recv_buf[i, recv_chunk*chunk:(recv_chunk+1)*chunk]
    return state   # state[i][chunk_i] is now the full sum of chunk i (not yet distributed)


def assert_reduce_scatter_correct(N: int = 4, P: int = 64) -> None:
    """state[i][chunk i of size P//N] should be the sum across all workers."""
    assert P % N == 0
    chunk = P // N
    torch.manual_seed(0)
    originals = torch.randn(N, P, device="cuda")
    state     = originals.clone()
    streams   = [torch.cuda.Stream() for _ in range(N)]
    reduce_scatter_ring(state, streams)
    for i in range(N):
        expected_chunk = originals[:, i*chunk:(i+1)*chunk].sum(dim=0)
        got_chunk = state[i, i*chunk:(i+1)*chunk]
        assert torch.allclose(got_chunk, expected_chunk, atol=1e-4), \
            f"worker {i} chunk wrong: max err {(got_chunk - expected_chunk).abs().max():.2e}"
    print(f"[ok] reduce-scatter: state[i][chunk i] = sum of chunk i  (N={N}, P={P})")


if __name__ == "__main__":
    assert_reduce_scatter_correct(N=4, P=64)
    assert_reduce_scatter_correct(N=8, P=256)
    print("[ok] reduce-scatter phase verified for N=4 and N=8")
    print("Next: p05/03 adds the all-gather phase to complete ring all-reduce.")
