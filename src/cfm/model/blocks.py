"""Transformer building blocks (plan.md §7): RMSNorm, RoPE, causal attention with
QK-norm, SwiGLU. No biases anywhere."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, base: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    ang = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()                         # [T, head_dim/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D]; rotates pairs (x[..., :D/2], x[..., D/2:])."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, hidden: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.mlp = SwiGLU(d, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        return x + self.drop(self.mlp(self.norm2(x)))
