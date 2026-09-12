# src/kvcache/batch_scheduler.py — continuous batching simulation
from collections import deque
import torch
from gpu_roofline.kvcache.page_pool import PagePool, PAGE_SIZE, append_token
from gpu_roofline.kvcache.paged_attn import paged_attention


class Sequence:
    """A pending or active generation request."""
    def __init__(self, seq_id: int, target_len: int, n_heads: int, head_dim: int):
        self.seq_id     = seq_id
        self.target_len = target_len      # how many tokens to generate (known for simulation)
        self.n_heads    = n_heads
        self.head_dim   = head_dim
        self.steps_done = 0


def simulate_serving(requests: list[tuple[int, int]],   # [(seq_id, target_len)]
                     max_pages: int = 128,
                     n_heads:   int = 4,
                     head_dim:  int = 32,
                     max_batch: int = 8) -> dict:
    """
    Continuous batching: process one token per step for each active sequence.
    Admit new sequences when pages are available; free pages on completion.
    Returns: steps_taken, total_tokens_generated, avg_utilisation.
    """
    pool     = PagePool(max_pages, n_heads, head_dim)
    waiting  = deque([Sequence(sid, tlen, n_heads, head_dim)
                      for sid, tlen in requests])
    active   = []
    completed = []
    step, total_tokens, util_sum = 0, 0, 0.0

    while waiting or active:
        # Admit new sequences (up to max_batch and pool capacity)
        while waiting and len(active) < max_batch:
            seq = waiting[0]
            pages_needed = (seq.target_len + PAGE_SIZE - 1) // PAGE_SIZE
            if len(pool.free_list) >= pages_needed:
                active.append(waiting.popleft())
            else:
                break                               # pool too full for next request; wait

        if not active:
            break                                   # deadlock guard (shouldn't happen)

        # One decoding step for each active sequence
        done_this_step = []
        for seq in active:
            k = torch.randn(n_heads, head_dim, device="cuda")
            v = torch.randn(n_heads, head_dim, device="cuda")
            append_token(pool, seq.seq_id, k, v)
            q = torch.randn(n_heads, head_dim, device="cuda")
            _ = paged_attention(q, pool, seq.seq_id, pool.seq_lens[seq.seq_id])
            seq.steps_done += 1
            total_tokens   += 1
            if seq.steps_done >= seq.target_len:
                done_this_step.append(seq)

        # Free completed sequences
        for seq in done_this_step:
            active.remove(seq)
            pool.free_sequence(seq.seq_id)
            completed.append(seq.seq_id)

        util_sum += pool.utilisation
        step     += 1

    return {"steps": step, "total_tokens": total_tokens,
            "avg_util": util_sum / max(step, 1),
            "completed": len(completed)}


def generate_requests(n_short: int = 20, n_long: int = 5,
                      seed: int = 0) -> list[tuple[int, int]]:
    torch.manual_seed(seed)
    short_lens = (torch.randint(10, 51, (n_short,)).tolist())
    long_lens  = (torch.randint(200, 501, (n_long,)).tolist())
    all_lens   = short_lens + long_lens
    torch.manual_seed(seed + 1)
    perm = torch.randperm(len(all_lens)).tolist()
    return [(i, all_lens[perm[i]]) for i in range(len(all_lens))]


if __name__ == "__main__":
    requests = generate_requests()
    total_tokens = sum(t for _, t in requests)
    print(f"Request stream: {len(requests)} requests, "
          f"total {total_tokens} tokens, "
          f"mean len {total_tokens//len(requests)}")

    result = simulate_serving(requests, max_pages=256, max_batch=8)
    print(f"\nContinuous batching result:")
    print(f"  steps taken     : {result['steps']}")
    print(f"  tokens generated: {result['total_tokens']}")
    print(f"  requests done   : {result['completed']}")
    print(f"  avg pool util   : {result['avg_util']:.2f}")


if __name__ == "__main__":
    requests = generate_requests(n_short=20, n_long=5)
    result = simulate_serving(requests, max_pages=256, max_batch=8)
    print(f"Simulation complete: {result}")
    print("\Next part will compare paged vs pre-allocated utilisation on the same workload.")
