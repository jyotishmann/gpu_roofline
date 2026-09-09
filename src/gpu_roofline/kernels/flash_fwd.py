# src/kernels/flash_fwd.py — FlashAttention forward kernel

import torch
from torch.utils.cpp_extension import load_inline
import sys, math
from gpu_roofline.harness.device import probe_device
from .attn_naive import (assert_attention_correct, naive_hbm_bytes,
                          flash_hbm_bytes, io_ratio, attention_flops,
                          benchmark_attention, print_attention_report, persist_result)

_FA_KERNEL = r"""
template<int D, int BKV>
__global__ void flash_attn_fwd(
        const float* __restrict__ Q,   // [N, D]
        const float* __restrict__ K,   // [N, D]
        const float* __restrict__ V,   // [N, D]
        float* __restrict__ O,         // [N, D]
        int N, float scale) {

    // ── Shared memory layout (labelled against Python Cell 2.3) ──────────────
    extern __shared__ float smem[];
    float* Ks   = smem;                        // Kj  [BKV][D]
    float* Vs   = smem + BKV * D;             // Vj  [BKV][D]
    float* Ss   = smem + 2 * BKV * D;        // Sij [BKV] — dot products for tile
    float* wbuf = smem + 2 * BKV * D + BKV;  // warp-reduce scratch [D/32]

    const int qi = blockIdx.x;          // query row this CTA owns
    const int tx = threadIdx.x;         // 0 .. D-1

    // q[tx] stays in registers across ALL KV iterations (never reloaded from smem)
    const float q = Q[qi * D + tx] * scale;

    // Online-softmax register state — same scalar value in all D threads (redundant, sync-free)
    float m   = -1e9f;   // running max         (Cell 2.3: m)
    float l   = 0.0f;    // running denominator (Cell 2.3: l)
    float acc = 0.0f;    // O[tx] accumulator   (Cell 2.3: O[tx])

    for (int kv_s = 0; kv_s < N; kv_s += BKV) {
        const int kv_len = min(BKV, N - kv_s);

        // ── PHASE 1: cooperative load K/V tile + compute Ss[] ─────────────────
        for (int j = 0; j < kv_len; j++) {
            Ks[j * D + tx] = K[(kv_s + j) * D + tx];   // column tx of K row kv_s+j
            Vs[j * D + tx] = V[(kv_s + j) * D + tx];   // column tx of V row kv_s+j
        }
        __syncthreads();

        // Dot product S[j] = q · K_tile[j] via two-warp reduction
        for (int j = 0; j < kv_len; j++) {
            float p = q * Ks[j * D + tx];                     // partial product
            for (int off = 16; off > 0; off >>= 1)
                p += __shfl_down_sync(0xffffffffu, p, off);   // warp-level reduce
            if (tx % 32 == 0) wbuf[tx / 32] = p;             // lane 0 of each warp stores
            __syncthreads();
            if (tx == 0) {                                     // thread 0 combines warp sums
                float s = wbuf[0];
                for (int w = 1; w < (D + 31) / 32; w++) s += wbuf[w];
                Ss[j] = s;                                     // written to smem for all to read
            }
            __syncthreads();
        }

        // ── PHASE 2: online-softmax update (all-register, no syncs) ──────────
        // Find tile max  → new running max
        float m_new = m;
        for (int j = 0; j < kv_len; j++) m_new = max(m_new, Ss[j]);

        const float sf = expf(m - m_new);   // rescaling factor (≤ 1; Cell 2.3: exp_diff)
        acc = sf * acc;                       // rescale O accumulator
        float l_delta = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            const float e = expf(Ss[j] - m_new);
            l_delta += e;                     // same for all D threads (redundant but sync-free)
            acc     += e * Vs[j * D + tx];   // different per tx: accumulates V[j][tx]
        }
        l = sf * l + l_delta;               // update denominator
        m = m_new;                           // update running max
        __syncthreads();                     // protect Ks/Vs smem before next tile overwrites
    }

    O[qi * D + tx] = acc / l;              // final normalisation — no sync needed (per-thread)
}
"""


_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"

_FA_LAUNCHER = r"""
template<int D, int BKV>
void _launch(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
             torch::Tensor O, float scale) {
    const int N = (int)Q.size(0);
    // smem = Ks[BKV*D] + Vs[BKV*D] + Ss[BKV] + wbuf[D/32]  — all float32
    const size_t smem = (2*BKV*D + BKV + (D+31)/32) * sizeof(float);
    cudaFuncSetAttribute(flash_attn_fwd<D,BKV>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    flash_attn_fwd<D,BKV><<<N, D, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(), N, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flash_fwd_launch(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                       torch::Tensor O, double scale_d, long long bkv) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda(),
                "flash_fwd: CUDA tensors required");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "flash_fwd: float32 only");
    const int D = (int)Q.size(1);
    const float scale = (float)scale_d;
    if      (bkv == 64 && D == 32)  _launch<32, 64>(Q,K,V,O,scale);
    else if (bkv == 64 && D == 64)  _launch<64, 64>(Q,K,V,O,scale);
    else if (bkv == 64 && D == 128) _launch<128,64>(Q,K,V,O,scale);
    else TORCH_CHECK(false, "flash_fwd: unsupported (D, BKV); add to dispatch table");
}
"""

_mod_fa = load_inline(
    name="p03_flash_fwd",
    cpp_sources="void flash_fwd_launch(torch::Tensor,torch::Tensor,torch::Tensor,"
                "torch::Tensor,double,long long);",
    cuda_sources=_INCLUDES + _FA_KERNEL + _FA_LAUNCHER,
    functions=["flash_fwd_launch"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True, verbose=False,
)


def flash_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
              scale: float | None = None, bkv: int = 64,
              causal: bool = False) -> torch.Tensor:
    """FlashAttention forward pass."""
    if causal:
        raise NotImplementedError("causal masking added in p03/04")
    N, D = Q.shape
    scale = scale or D ** -0.5
    O = torch.zeros(N, D, device=Q.device, dtype=Q.dtype)
    _mod_fa.flash_fwd_launch(Q.contiguous(), K.contiguous(),
                              V.contiguous(), O, scale, bkv)
    return O


if __name__ == "__main__":
    dev = probe_device()

    # Correctness — reuse the imported gate exactly
    assert_attention_correct(flash_fwd, causal=False, atol=1e-5)

    # Benchmark: FA vs naïve side-by-side
    N, D = 2048, 64
    Q = torch.randn(N, D, device="cuda")
    K = torch.randn(N, D, device="cuda")
    V = torch.randn(N, D, device="cuda")

    r_naive = persist_result.__module__  # just to trigger the import check
    r_fa = benchmark_attention("flash_fwd", flash_fwd, Q, K, V, N, D, dev)
    print_attention_report(r_fa, dev)

    # Print the IO-complexity comparison explicitly
    nb = naive_hbm_bytes(N, D)
    fb = flash_hbm_bytes(N, D)
    print(f"\n  theoretical IO reduction : {nb/fb:.0f}×  (naïve {nb/1e6:.0f} MB → FA {fb/1e6:.1f} MB)")
    print(f"  note: flash_fwd reports effective BW against the FA minimum bytes ({fb/1e6:.1f} MB);")
    print(f"        a high % peak BW means we are saturating the bus with the RIGHT data.")
    persist_result(r_fa)
