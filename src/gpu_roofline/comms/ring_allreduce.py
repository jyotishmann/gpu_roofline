# src/comms/ring_allreduce.py — ring all-reduce (reduce-scatter + all-gather)
import torch
import json, pathlib
from gpu_roofline.comms.naive_allreduce import (make_workers, allreduce_bytes_naive,
                                    benchmark_allreduce, persist_ar_result, naive_allreduce)
from gpu_roofline.comms.naive_allreduce import assert_allreduce_correct


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


def all_gather_ring(state: torch.Tensor, streams: list) -> torch.Tensor:
    """
    Phase 2 of ring all-reduce: N-1 copy steps to distribute the reduced chunks.
    Precondition: state[i][chunk i] holds the fully-reduced sum (from reduce-scatter).
    Postcondition: state[i] == state[j] for all i, j (all workers have the full result).
    """
    N, P = state.shape
    chunk = P // N
    for s in range(N - 1):
        for i in range(N):
            # The chunk we currently hold that is "ready" to share
            send_chunk = (i - s + 1) % N     # +1: after reduce-scatter we own chunk i
            src = (i - 1) % N
            recv_chunk = (i - s) % N
            with torch.cuda.stream(streams[i]):
                state[i, recv_chunk*chunk:(recv_chunk+1)*chunk].copy_(
                    state[src, recv_chunk*chunk:(recv_chunk+1)*chunk],
                    non_blocking=True)
        for i in range(N):
            streams[i].synchronize()
    return state   # state[i] == full reduced gradient for all i


def ring_allreduce(state: torch.Tensor) -> torch.Tensor:
    """Complete ring all-reduce: reduce-scatter + all-gather."""
    N = state.shape[0]
    streams = [torch.cuda.Stream() for _ in range(N)]
    reduce_scatter_ring(state, streams)
    state /= N            # normalise after reduce-scatter (before broadcasting)
    all_gather_ring(state, streams)
    return state


def verify_ring_allreduce() -> None:
    assert_allreduce_correct(ring_allreduce, N=4,  P=1024)
    assert_allreduce_correct(ring_allreduce, N=8,  P=2048)
    assert_allreduce_correct(ring_allreduce, N=4,  P=4097)  # non-round P — tests boundary handling
    print("[ok] ring_allreduce end-to-end correct for N=4, N=8, and non-round P")


def run_reduce_scatter_checks() -> None:
    """Phase-1 correctness gate. Called from the p05/03 __main__ before the full ring test."""
    assert_reduce_scatter_correct(N=4, P=64)
    assert_reduce_scatter_correct(N=8, P=256)
    print("[ok] reduce-scatter phase verified for N=4 and N=8")


def allreduce_bytes_ring(N: int, P: int) -> dict:
    """Bytes sent+received per worker in ring: 2(N-1)/N x P x 4, equally across all."""
    per_worker_bytes = 2 * (N - 1) * P * 4 // N
    return {i: per_worker_bytes for i in range(N)}


if __name__ == "__main__":
    # Phase 1 regression (reduce-scatter must still pass after all-gather is added)
    run_reduce_scatter_checks()

    # Phase 2 + full ring
    verify_ring_allreduce()

    from gpu_roofline.harness.device import probe_device
    dev = probe_device()
    
    P = 1 << 22                             # 4M floats ≈ 16 MB — latency-amortised regime
    print(f"\nBandwidth comparison  (P={P//1024}K floats, N=4, peak={dev.peak_bw_gbps:.0f} GB/s):")
    r_naive = benchmark_allreduce(naive_allreduce, 4, P, dev)
    r_ring  = benchmark_allreduce(ring_allreduce,  4, P, dev)
    print(f"  naïve : root={r_naive['root_bw_gbps']:.0f} GB/s, leaf={r_naive['leaf_bw_gbps']:.0f} GB/s")
    ring_bw = allreduce_bytes_ring(4, P)[0] / r_ring["median_s"] / 1e9
    print(f"  ring  : all workers ≈ {ring_bw:.0f} GB/s  (target: ~{dev.peak_bw_gbps:.0f})")
    r_ring["name"] = "ring_allreduce_N4"
    persist_ar_result(r_ring)
