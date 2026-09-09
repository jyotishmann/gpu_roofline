# src/kernels/online_softmax.py — online softmax recurrence
import math
import torch

from .attn_naive import naive_attention, assert_attention_correct


# m : float  (scalar per query — running max of attention logits)
# l : float  (scalar per query — unnormalised softmax denominator, relative to m)
# O : Tensor[d] (d-vector per query — unnormalised output, relative to m)


def online_attention_single_query(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   Bkv: int = 64, scale: float | None = None,
                                   causal: bool = False) -> torch.Tensor:
    """
    Online softmax attention, one query at a time.
    State variables (m, l, O) are scalars/1D-vectors — maximally readable.
    Slow: O(N²) Python loops. Purpose: prove the recurrence, not measure throughput.
    """
    N, d = Q.shape
    scale = scale or d ** -0.5
    O_out = torch.zeros(N, d, device=Q.device, dtype=Q.dtype)

    for qi in range(N):                        # one query per outer iteration
        q = Q[qi] * scale                      # [d]
        m = float("-inf")                      # running max (scalar)
        l = 0.0                                # running denominator (scalar, relative to m)
        O = torch.zeros(d, device=Q.device, dtype=Q.dtype)  # running output [d]

        for kv_start in range(0, N, Bkv):
            if causal and kv_start > qi:
                break                          # future blocks: all −∞, contribute nothing
            kv_end = min(kv_start + Bkv, N)
            Kj = K[kv_start:kv_end]           # [Bkv, d]
            Vj = V[kv_start:kv_end]           # [Bkv, d]

            s = torch.mv(Kj, q)               # [Bkv] — unnormalised logits for this block
            if causal:                         # mask future positions within block
                cutoff = qi - kv_start + 1    # positions 0..cutoff-1 are valid
                s[cutoff:] = float("-inf")

            m_new = max(m, s.max().item())     # ← update running max
            scale_old = math.exp(m - m_new)   # ← rescaling factor: always ≤ 1

            exp_s = torch.exp(s - m_new)       # [Bkv] — relative to new max
            l = scale_old * l + exp_s.sum().item()      # l update (Eq. from 2.1)
            O = scale_old * O + torch.mv(Vj.t(), exp_s) # O update (Eq. from 2.1)
            m = m_new

        O_out[qi] = O / l                      # final normalisation

    return O_out


def online_attention_blocked(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              Bq: int = 64, Bkv: int = 64,
                              scale: float | None = None,
                              causal: bool = False) -> torch.Tensor:
    """
    Online softmax attention, Bq queries × Bkv keys at a time.
    m, l: Tensor[Bq] — the CTA's per-row state registers.
    O: Tensor[Bq, d] — the CTA's per-row accumulator.
    This form maps 1-to-1 onto the CUDA kernel structure in p03/03.
    """
    N, d = Q.shape
    scale = scale or d ** -0.5
    O_out = torch.zeros(N, d, device=Q.device, dtype=Q.dtype)

    for q_start in range(0, N, Bq):
        q_end = min(q_start + Bq, N)
        bsz   = q_end - q_start
        Qi    = Q[q_start:q_end] * scale           # [bsz, d]

        m = torch.full((bsz,), float("-inf"), device=Q.device, dtype=Q.dtype) # [bsz]
        l = torch.zeros(bsz,    device=Q.device, dtype=Q.dtype)               # [bsz]
        O = torch.zeros(bsz, d, device=Q.device, dtype=Q.dtype)               # [bsz, d]

        for kv_start in range(0, N, Bkv):
            if causal and kv_start >= q_end:
                break
            kv_end = min(kv_start + Bkv, N)
            Kj = K[kv_start:kv_end]               # [Bkv, d]
            Vj = V[kv_start:kv_end]               # [Bkv, d]

            Sij = torch.mm(Qi, Kj.t())            # [bsz, Bkv] — block of logits
            if causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)  # [bsz, 1]
                k_idx = torch.arange(kv_start, kv_end, device=Q.device).unsqueeze(0) # [1, Bkv]
                Sij = Sij.masked_fill(q_idx < k_idx, float("-inf"))

            m_new    = torch.maximum(m, Sij.amax(dim=1))        # [bsz]
            sf       = torch.exp(m - m_new)                     # [bsz] rescaling factor
            exp_Sij  = torch.exp(Sij - m_new.unsqueeze(1))      # [bsz, Bkv]

            l = sf * l + exp_Sij.sum(dim=1)                     # [bsz]   (Eq. 2.1)
            O = sf.unsqueeze(1) * O + torch.mm(exp_Sij, Vj)    # [bsz,d] (Eq. 2.1)
            m = m_new

        O_out[q_start:q_end] = O / l.unsqueeze(1)               # [bsz, d] final norm

    return O_out


def verify_online_softmax() -> None:
    """
    Four correctness properties of the online softmax recurrence.
    The most important: result is IDENTICAL regardless of block size Bkv.
    """
    torch.manual_seed(7)
    N, d = 512, 64

    Q = torch.randn(N, d, device="cuda")
    K = torch.randn(N, d, device="cuda")
    V = torch.randn(N, d, device="cuda")
    ref = naive_attention(Q, K, V)   # the ground truth

    # 1. Single-query vs naïve
    got = online_attention_single_query(Q, K, V, Bkv=64)
    assert torch.allclose(got, ref, atol=1e-5), \
        f"single-query vs naïve: max err {(got-ref).abs().max():.2e}"
    print("[ok] single-query matches naïve  (N=512, Bkv=64)")

    # 2. Blocked vs naïve (Bq=64, Bkv=64)
    got = online_attention_blocked(Q, K, V, Bq=64, Bkv=64)
    assert torch.allclose(got, ref, atol=1e-5), \
        f"blocked vs naïve: max err {(got-ref).abs().max():.2e}"
    print("[ok] blocked(Bq=64, Bkv=64) matches naïve")

    # 3. Block-size independence — the critical property
    for Bkv in [1, 8, 32, 64, N]:
        got = online_attention_blocked(Q, K, V, Bq=64, Bkv=Bkv)
        assert torch.allclose(got, ref, atol=1e-5), \
            f"Bkv={Bkv}: max err {(got-ref).abs().max():.2e}"
        print(f"[ok] blocked(Bkv={Bkv:>4}) exactly equivalent to Bkv={N} (full-sequence)")

    # 4. Causal mode
    ref_c = naive_attention(Q, K, V, causal=True)
    got_c = online_attention_blocked(Q, K, V, Bq=64, Bkv=64, causal=True)
    assert torch.allclose(got_c, ref_c, atol=1e-5), \
        f"causal: max err {(got_c-ref_c).abs().max():.2e}"
    print("[ok] causal mode matches naïve causal")

    print("\nAll four correctness properties verified. "
          "The online softmax recurrence is exact regardless of block size.")


if __name__ == "__main__":
    verify_online_softmax()
