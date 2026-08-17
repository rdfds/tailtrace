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


