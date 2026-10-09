# src/gpu_roofline/comms/ring_allreduce.py — ring all-reduce (reduce-scatter + all-gather)
import torch
import json, pathlib
from gpu_roofline.comms.naive_allreduce import (make_workers, allreduce_bytes_naive,
                                    benchmark_allreduce, persist_ar_result, naive_allreduce)
from gpu_roofline.comms.naive_allreduce import assert_allreduce_correct


def reduce_scatter_ring(state: torch.Tensor, streams: list) -> torch.Tensor:
    """
    Phase 1 of ring all-reduce: N-1 steps, each worker accumulates one received chunk.
    After this, state[i][chunk (i+1)%N] holds the correct sum of that chunk
    state: [N, P] — worker i's gradient in state[i]
    streams: list of N CUDA streams (one per worker; allows concurrent copies)
    """
    N, P = state.shape
    assert P % N == 0, "P must be divisible by N for equal chunk sizes"
    chunk = P // N                                # elements per chunk
    recv_buf = state.clone()                      # recv buffer (copy, not alias)

    for s in range(N - 1):
        # Step s: worker i sends chunk (i-s)%N to its right neighbour and receives
        # chunk (i-s-1)%N from its left neighbour (src). The left neighbour's send
        # chunk is (src-s)%N == (i-s-1)%N, so sender and receiver agree.

        # Order the side streams after the default stream. The previous step's
        # accumulation (+=) and the initial clone ran on the default stream, and the
        # copies below must not read `state` until they are done.
        cur = torch.cuda.current_stream()
        for st in streams:
            st.wait_stream(cur)

        # Phase A: all N receives are issued from the same snapshot of `state`.
        # `state` is not modified during this phase, so the order of i doesn't matter.
        for i in range(N):
            recv_chunk = (i - s - 1) % N
            src = (i - 1) % N                    # left neighbour
            lo, hi = recv_chunk * chunk, (recv_chunk + 1) * chunk
            with torch.cuda.stream(streams[i]):
                recv_buf[i, lo:hi].copy_(state[src, lo:hi], non_blocking=True)

        # All receives must finish before any accumulation starts (otherwise a worker
        # could modify a chunk that its right neighbour is still reading).
        for st in streams:
            cur.wait_stream(st)

        # Phase B: accumulate on the default stream, ordered after every copy above.
        for i in range(N):
            recv_chunk = (i - s - 1) % N
            lo, hi = recv_chunk * chunk, (recv_chunk + 1) * chunk
            state[i, lo:hi] += recv_buf[i, lo:hi]

    # After N-1 steps, worker i holds the fully reduced chunk (i+1) % N, not chunk i.
    # (The final step s = N-2 receives chunk (i-N+1) % N == (i+1) % N.)
    return state


def assert_reduce_scatter_correct(N: int = 4, P: int = 64) -> None:
    """state[i][chunk i of size P//N] should be the sum across all workers."""
    assert P % N == 0
    chunk = P // N
    torch.manual_seed(0)
    originals = torch.randn(N, P, device="cuda")
    state     = originals.clone()
    streams   = [torch.cuda.Stream() for _ in range(N)]
    reduce_scatter_ring(state, streams)
    # worker i owns chunk (i+1) % N after N-1 ring steps
    for i in range(N):
        c = (i + 1) % N
        expected_chunk = originals[:, c*chunk:(c+1)*chunk].sum(dim=0)
        got_chunk = state[i, c*chunk:(c+1)*chunk]
    
        assert torch.allclose(got_chunk, expected_chunk, atol=1e-4), \
            f"worker {i} chunk wrong: max err {(got_chunk - expected_chunk).abs().max():.2e}"
    print(f"[ok] reduce-scatter: state[i][chunk (i+1)%N] = sum of that chunk  (N={N}, P={P})")


def all_gather_ring(state: torch.Tensor, streams: list) -> torch.Tensor:
    """
    Phase 2 of ring all-reduce: N-1 copy steps to distribute the reduced chunks.
    Precondition: After this, state[i][chunk (i+1)%N] holds the correct sum of that chunk
    Postcondition: state[i] == state[j] for all i, j (all workers have the full result).
    """
    N, P = state.shape
    chunk = P // N
    cur = torch.cuda.current_stream()
    for s in range(N - 1):
        # Order every side stream after the default stream. This covers the
        # reduce-scatter accumulation and the `state /= N` in ring_allreduce(),
        # both of which ran on the default stream.
        for st in streams:
            st.wait_stream(cur)

        # Step s: worker i sends chunk (i-s+1)%N to its right neighbour and receives
        # chunk (i-s)%N from its left neighbour. At s=0, after reduce-scatter,
        # worker i owns chunk (i+1)%N, so the first chunk it forwards is that one.
        for i in range(N):
            src = (i - 1) % N                         # left neighbour
            send_chunk = (src - s + 1) % N            # what src sends this step
            recv_chunk = (i - s) % N                  # what we receive this step
            assert send_chunk == recv_chunk           # sender and receiver agree
            lo, hi = recv_chunk * chunk, (recv_chunk + 1) * chunk
            with torch.cuda.stream(streams[i]):
                state[i, lo:hi].copy_(state[src, lo:hi], non_blocking=True)

        # Join: the next step reads chunks written on other workers' streams, so all
        # copies must complete first. This orders the work on the GPU without
        # blocking the host.
        for st in streams:
            cur.wait_stream(st)

    return state   # state[i] == full reduced gradient for every worker i


def ring_allreduce(state: torch.Tensor) -> torch.Tensor:
    """Complete ring all-reduce: reduce-scatter + all-gather. Works in place on `state`.

    P need not be divisible by N: the buffer is zero-padded up to the next multiple of
    N, reduced, then trimmed back. Zeros are the identity for the sum, so the result
    for the first P elements is exact. The padding cost is at most N-1 elements per
    worker, which is negligible next to P.
    """
    N, P = state.shape
    streams = [torch.cuda.Stream() for _ in range(N)]

    pad = (-P) % N                                   # 0 when P is already divisible
    # F.pad returns a new tensor, so when padding is needed we reduce on the copy and
    # write the result back into `state` at the end. When pad == 0 we work on
    # `state` directly, which keeps the common path copy-free.
    work = state if pad == 0 else torch.nn.functional.pad(state, (0, pad))

    reduce_scatter_ring(work, streams)
    work /= N                                        # normalise to the mean before broadcast
    all_gather_ring(work, streams)

    if pad:
        state.copy_(work[:, :P])                     # trim padding; ordered after the ring
                                                     # because all_gather_ring joins its
                                                     # streams on the default stream
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
    r_naive = benchmark_allreduce(naive_allreduce, dev, N=4, P=P)
    r_ring  = benchmark_allreduce(ring_allreduce,  dev, N=4, P=P)
    print(f"  naïve : root={r_naive['root_bw_gbps']:.0f} GB/s, leaf={r_naive['leaf_bw_gbps']:.0f} GB/s")
    ring_bw = allreduce_bytes_ring(4, P)[0] / r_ring["median_s"] / 1e9
    print(f"  ring  : all workers ≈ {ring_bw:.0f} GB/s  (target: ~{dev.peak_bw_gbps:.0f})")
    r_ring["name"] = "ring_allreduce_N4"
    persist_ar_result(r_ring)
