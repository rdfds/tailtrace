from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def residual_rmsnorm(x, residual, weight, eps=1e-6, implementation="reference"):
    if implementation == "cuda":
        from tailtrace.kernels import fused_residual_rmsnorm

        return fused_residual_rmsnorm(x, residual, weight, eps)
    accum = torch.float64 if x.dtype == torch.float64 else torch.float32
    z = x.to(accum) + residual.to(accum)
    y = z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps) * weight.to(accum)
    return y.to(x.dtype)


