# src/kernels/flash_fwd_causal.py — causal mask + logsumexp

import torch
from torch.utils.cpp_extension import load_inline
import sys
from gpu_roofline.harness.device import probe_device
from .attn_naive import (assert_attention_correct, naive_hbm_bytes, flash_hbm_bytes,
                          attention_flops, benchmark_attention,
                          print_attention_report, persist_result)

_FA_CAUSAL_KERNEL = r"""
template<int D, int BKV>
__global__ void flash_attn_fwd_causal(
        const float* __restrict__ Q, const float* __restrict__ K,
        const float* __restrict__ V, float* __restrict__ O,
        float* __restrict__ LSE,           // [N] logsumexp buffer for backward pass
        int N, float scale, bool causal) {

    extern __shared__ float smem[];
    float* Ks   = smem;
    float* Vs   = smem + BKV * D;
    float* Ss   = smem + 2 * BKV * D;
    float* wbuf = smem + 2 * BKV * D + BKV;

    const int qi = blockIdx.x;
    const int tx = threadIdx.x;
    const float q = Q[qi * D + tx] * scale;
    float m = -1e9f, l = 0.0f, acc = 0.0f;

    for (int kv_s = 0; kv_s < N; kv_s += BKV) {
        // ── Case A: entire tile is future → skip ──────────────────────────────
        if (causal && kv_s > qi) break;

        const int kv_len = min(BKV, N - kv_s);

        // Cooperative load K/V tile
        for (int j = 0; j < kv_len; j++) {
            Ks[j * D + tx] = K[(kv_s + j) * D + tx];
            Vs[j * D + tx] = V[(kv_s + j) * D + tx];
        }
        __syncthreads();

        // Dot products
        for (int j = 0; j < kv_len; j++) {
            float p = q * Ks[j * D + tx];
            for (int off = 16; off > 0; off >>= 1)
                p += __shfl_down_sync(0xffffffffu, p, off);
            if (tx % 32 == 0) wbuf[tx / 32] = p;
            __syncthreads();
            if (tx == 0) {
                float s = wbuf[0];
                for (int w = 1; w < (D+31)/32; w++) s += wbuf[w];
                // ── Case B: mask future positions within a partial tile ────────
                Ss[j] = (causal && (kv_s + j) > qi) ? -1e9f : s;
            }
            __syncthreads();
        }

        // Online softmax update (unchanged from p03/03)
        float m_new = m;
        for (int j = 0; j < kv_len; j++) m_new = max(m_new, Ss[j]);
        const float sf = expf(m - m_new);
        acc = sf * acc;
        float l_delta = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            const float e = expf(Ss[j] - m_new);
            l_delta += e;
            acc     += e * Vs[j * D + tx];
        }
        l = sf * l + l_delta;
        m = m_new;
        __syncthreads();
    }

    O[qi * D + tx] = acc / l;
    if (tx == 0) LSE[qi] = m + logf(l);   // logsumexp: log(Σ exp(s-m)) + m = log(Σ exp(s))
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_CAUSAL_LAUNCHER = r"""
template<int D, int BKV>
void _launch_c(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
               torch::Tensor O, torch::Tensor LSE, float scale, bool causal) {
    const int N   = (int)Q.size(0);
    const size_t smem = (2*BKV*D + BKV + (D+31)/32) * sizeof(float);
    cudaFuncSetAttribute(flash_attn_fwd_causal<D,BKV>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    flash_attn_fwd_causal<D,BKV><<<N, D, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(), LSE.data_ptr<float>(),
        N, scale, causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flash_fwd_causal_launch(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               torch::Tensor O, torch::Tensor LSE,
                               double scale_d, long long bkv, bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda()
                && O.is_cuda() && LSE.is_cuda(), "flash_fwd_causal: CUDA required");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "float32 only");
    const int D = (int)Q.size(1);
    const float scale = (float)scale_d;
    if      (bkv==64 && D==32)  _launch_c<32, 64>(Q,K,V,O,LSE,scale,causal);
    else if (bkv==64 && D==64)  _launch_c<64, 64>(Q,K,V,O,LSE,scale,causal);
    else if (bkv==64 && D==128) _launch_c<128,64>(Q,K,V,O,LSE,scale,causal);
    else TORCH_CHECK(false, "flash_fwd_causal: unsupported (D, BKV)");
}
"""

_mod_c = load_inline(
    name="p03_flash_fwd_causal",
    cpp_sources="void flash_fwd_causal_launch(torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,torch::Tensor,double,long long,bool);",
    cuda_sources=_INCLUDES + _FA_CAUSAL_KERNEL + _CAUSAL_LAUNCHER,
    functions=["flash_fwd_causal_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True, verbose=False,
)


def flash_fwd_causal(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     causal: bool = True, scale: float | None = None,
                     bkv: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (O, LSE) — output and logsumexp buffer for the backward pass."""
    N, D = Q.shape
    scale = scale or D ** -0.5
    O   = torch.zeros(N, D, device=Q.device, dtype=Q.dtype)
    LSE = torch.zeros(N,    device=Q.device, dtype=Q.dtype)
    _mod_c.flash_fwd_causal_launch(Q.contiguous(), K.contiguous(), V.contiguous(),
                                    O, LSE, scale, bkv, causal)
    return O, LSE


def _fa_causal_wrapper(Q, K, V, causal=False):
    O, _ = flash_fwd_causal(Q, K, V, causal=causal)
    return O


if __name__ == "__main__":
    dev = probe_device()
    assert_attention_correct(_fa_causal_wrapper, causal=False, atol=1e-5)
    assert_attention_correct(_fa_causal_wrapper, causal=True,  atol=1e-5)
    print()
    N, D = 2048, 64
    Q = torch.randn(N, D, device="cuda")
    K = torch.randn(N, D, device="cuda")
    V = torch.randn(N, D, device="cuda")
    for causal in (False, True):
        fn = lambda Q=Q, K=K, V=V, c=causal: flash_fwd_causal(Q, K, V, causal=c)[0]
        r  = benchmark_attention(f"flash_fwd_causal={causal}", fn, Q, K, V, N, D, dev)
        print_attention_report(r, dev)
        persist_result(r)
    print("\nCausal should be ~2× faster than non-causal "
          "(half the KV tiles skipped via the `break` guard).")
