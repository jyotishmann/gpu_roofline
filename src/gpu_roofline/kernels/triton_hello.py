# src/kernels/triton_hello.py — Triton environment check and hello-kernel
import torch
import triton # type: ignore
import triton.language as tl # type: ignore
import sys
from gpu_roofline.harness.device import probe_device


def check_triton_env() -> dict:
    """Verify Triton is installed; report version and relevant hardware capabilities."""
    dev = probe_device()
    has_cp_async = dev.cc >= (8, 0)   # cp.async (hardware async copy) requires sm_80+
    has_tensor_cores = dev.cc >= (7, 0)  # WMMA/MMA tensor cores from sm_70+
    info = {
        "triton_version": triton.__version__,
        "device": dev.name,
        "cc": dev.cc,
        "has_cp_async": has_cp_async,    # determines max useful num_stages
        "has_tensor_cores": has_tensor_cores,  # determines whether tl.dot uses MMA
    }
    print(f"Triton {triton.__version__}  on  {dev.name}  (sm_{dev.cc[0]}{dev.cc[1]})")
    print(f"  cp.async (hardware async copy) : {'yes — num_stages > 1 is meaningful' if has_cp_async else 'no  — num_stages > 1 has no hardware benefit on sm_75'}")
    print(f"  tensor cores (MMA)             : {'yes — tl.dot routes to MMA' if has_tensor_cores else 'no'}")
    return info

@triton.jit
def _vec_add_kernel(A_ptr, B_ptr, C_ptr, N,
                    BLOCK: tl.constexpr):        # compile-time constant, like template<int BLOCK>
    pid    = tl.program_id(0)                    # block index — like blockIdx.x
    offs   = pid * BLOCK + tl.arange(0, BLOCK)  # element indices for this block
    mask   = offs < N                            # boundary guard (like the (j < kv_len) guards)
    a = tl.load(A_ptr + offs, mask=mask)         # cooperative load from global memory
    b = tl.load(B_ptr + offs, mask=mask)
    tl.store(C_ptr + offs, a + b, mask=mask)    # write result


def triton_add(a: torch.Tensor, b: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    c = torch.empty_like(a)
    grid = (triton.cdiv(a.numel(), BLOCK),)      # ceil(N/BLOCK) blocks
    _vec_add_kernel[grid](a, b, c, a.numel(), BLOCK=BLOCK)
    return c


def verify_triton_hello() -> None:
    torch.manual_seed(0)
    n = 1_000_003   # prime: tests the boundary guard
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    got = triton_add(a, b)
    ref = a + b
    assert torch.allclose(got, ref, atol=1e-6), \
        f"triton_add disagrees with torch: max err {(got-ref).abs().max():.2e}"
    print("[ok] triton_add matches torch a+b  (N=1_000_003, prime — tests boundary guard)")
    print("     First call triggers JIT compilation; subsequent calls use the cached binary.")


if __name__ == "__main__":
    check_triton_env()
    verify_triton_hello()
