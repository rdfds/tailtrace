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


class Block(nn.Module):
    def __init__(self, width, heads, kernel):
        super().__init__()
        self.heads, self.kernel = heads, kernel
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(),
            nn.Linear(4 * width, width, bias=False),
        )
        self.norm1 = nn.Parameter(torch.ones(width))
        self.norm2 = nn.Parameter(torch.ones(width))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (a.view(b, t, self.heads, d // self.heads).transpose(1, 2) for a in (q, k, v))
        # Right padding requires no attention mask for valid causal positions.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        x = residual_rmsnorm(a.to(x.dtype), x, self.norm1.to(x.dtype), implementation=self.kernel)
        return residual_rmsnorm(
            self.ff(x).to(x.dtype), x, self.norm2.to(x.dtype), implementation=self.kernel
        )


