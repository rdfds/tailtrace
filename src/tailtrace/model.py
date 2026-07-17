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


class CausalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.max_length, config.width)
        self.blocks = nn.ModuleList(
            [Block(config.width, config.heads, config.kernel) for _ in range(config.layers)]
        )
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens) + self.positions(
            torch.arange(tokens.shape[1], device=tokens.device)
        )
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def make_batch(ids, lengths, vocab_size, seed, device):
    t = max(lengths[i] - 1 for i in ids)
    x = torch.zeros((len(ids), t), dtype=torch.long)
    target = torch.full_like(x, -100)
    for row, i in enumerate(ids):
        g = torch.Generator().manual_seed(seed * 1000003 + i)
        seq = torch.randint(vocab_size, (lengths[i],), generator=g)
        x[row, : lengths[i] - 1] = seq[:-1]
        target[row, : lengths[i] - 1] = seq[1:]
    if device.type == "cuda":
        x, target = x.pin_memory(), target.pin_memory()
    return x.to(device, non_blocking=True), target.to(device, non_blocking=True)


def token_loss(logits, target, global_tokens, world_size):
    # DDP/FSDP average rank gradients. Undo that average before dividing by global tokens.
    stable_logits = logits if logits.dtype == torch.float64 else logits.float()
    local_sum = F.cross_entropy(
        stable_logits.flatten(0, 1), target.flatten(), ignore_index=-100, reduction="sum"
    )
    return local_sum * (world_size / global_tokens), local_sum.detach()
