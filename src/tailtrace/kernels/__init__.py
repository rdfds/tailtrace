"""JIT-loaded CUDA extension; importing this module never compiles a kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def extension():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is required; select the reference implementation on CPU")
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    return load(
        name="tailtrace_residual_rmsnorm_v1",
        sources=[str(root / "bindings.cpp"), str(root / "residual_rmsnorm.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class _ResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        if eps <= 0:
            raise ValueError("eps must be positive")
        x, residual, weight = x.contiguous(), residual.contiguous(), weight.contiguous()
        y, inv = extension().forward(x, residual, weight, eps)
        ctx.save_for_backward(x, residual, weight, inv)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, residual, weight, inv = ctx.saved_tensors
        dx, dw = extension().backward(dy.contiguous(), x, residual, weight, inv)
        return dx, dx, dw, None


def fused_residual_rmsnorm(x, residual, weight, eps=1e-6):
    """First-order autograd only. CUDA fp32/fp16/bf16, matching dtypes, no broadcasting."""
    return _ResidualRMSNorm.apply(x, residual, weight, eps)
