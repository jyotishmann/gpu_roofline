# src/kernels/vector_add.py — first CUDA kernel, built across cells 2.1–2.5

import torch
from torch.utils.cpp_extension import load_inline

_ADD_KERNEL = r"""
__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];  // one thread, one element; the grid-stride loop scales it to all N
}
"""

_ADD_LAUNCHER = r"""
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "vector_add: tensors must be CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32,        "vector_add: float32 only");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "vector_add: tensors must be contiguous");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == c.numel(), "vector_add: size mismatch");
    const long long n = a.numel();
    const int block = 256;                              // ADR-005 default; p00/06 will sweep it
    const int grid  = (int)((n + block - 1) / block);
    add_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_INCLUDES = "#include <torch/extension.h>\n#include <c10/cuda/CUDAException.h>\n"
_CUDA_SRC = _INCLUDES + _ADD_KERNEL + _ADD_LAUNCHER
_CPP_DECL = "void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c);"

_mod = load_inline(
    name="p00_vector_add",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["vector_add"],   # declared in cpp, defined in cuda → load_inline binds it for us
    with_cuda=True,
    verbose=False,
)

def add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Frontend API: allocate output if needed, launch the kernel, return c = a + b."""
    if out is None:
        out = torch.empty_like(a)  # optional reuse keeps allocation OUT of p00/03's timed region
    _mod.vector_add(a, b, out)
    return out

def _check_correctness() -> None:
    torch.manual_seed(0)
    n = 1_000_003  # prime → forces a partial final block, testing grid-stride bounds handling
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    assert torch.equal(add(a, b), a + b), "vector_add disagrees with torch a+b"
    print(f"[ok] vector_add exact-matches torch on N={n}")


if __name__ == "__main__":
    _check_correctness()
