"""Patch backbone + heads (plan.md §6-7).

Input [B, T, P*F] (T patches of P candles x F features) -> causal transformer ->
per position:
- return logits [B, T, H, K] over K = 66 bins for each return horizon;
- volatility log-ratios u [B, T, Hv]: log sigma_hat^2_h = log(h * sigma_ewma^2) + u, i.e.
  the head predicts the correction to the EWMA forecast (zero-initialised, so training
  starts from EWMA).
Init: trunc_normal(0, 0.02); residual output projections scaled by 1/sqrt(2 n_layers).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cfm.model.blocks import Block, RMSNorm, rope_tables


class Backbone(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_layers: int, n_heads: int, mlp_hidden: int,
                 dropout: float, rope_base: float, n_return: int, n_classes: int, n_vol: int,
                 max_len: int = 1024, in_mask: torch.Tensor | None = None):
        super().__init__()
        self.n_return, self.n_classes, self.n_vol = n_return, n_classes, n_vol
        self.patch_embed = nn.Linear(in_dim, d_model, bias=False)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, mlp_hidden, dropout) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.ret_head = nn.Linear(d_model, n_return * n_classes, bias=False)
        self.vol_head = nn.Linear(d_model, n_vol, bias=False)
        cos, sin = rope_tables(max_len, d_model // n_heads, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("in_mask", in_mask, persistent=False)   # zeroes dropped input features
        self._init(n_layers)

    def _init(self, n_layers: int) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
        for b in self.blocks:
            for lin in (b.attn.out, b.mlp.w2):
                nn.init.trunc_normal_(lin.weight, std=0.02 / math.sqrt(2 * n_layers), a=-0.04, b=0.04)
        nn.init.zeros_(self.vol_head.weight)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_mask is not None:
            x = x * self.in_mask
        h = self.patch_embed(x)
        for b in self.blocks:
            h = b(h, self.rope_cos, self.rope_sin)
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(x)
        B, T, _ = h.shape
        logits = self.ret_head(h).view(B, T, self.n_return, self.n_classes)
        return logits, self.vol_head(h)

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on matrices only: not on norms, biases or the patch embedding."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (decay if p.ndim >= 2 and not name.startswith("patch_embed") else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def build_backbone(bcfg: dict, dcfg: dict, n_features: int) -> Backbone:
    from cfm.data.features import FEATURES

    m, t = bcfg["model"], dcfg["targets"]
    in_mask = None
    if m.get("drop_features"):
        unknown = set(m["drop_features"]) - set(FEATURES)
        assert not unknown and n_features == len(FEATURES), f"unknown features {unknown}"
        keep = torch.tensor([f not in m["drop_features"] for f in FEATURES], dtype=torch.float32)
        in_mask = keep.repeat(t["patch_minutes"])            # input is [P * F], candle-major
    return Backbone(in_dim=t["patch_minutes"] * n_features, d_model=m["d_model"],
                    n_layers=m["n_layers"], n_heads=m["n_heads"], mlp_hidden=m["mlp_hidden"],
                    dropout=m["dropout"], rope_base=m["rope_base"],
                    n_return=len(t["return_horizons"]), n_classes=t["bins"]["n_inner"] + 2,
                    n_vol=len(t["vol_horizons"]), max_len=max(1024, bcfg["data"]["context_patches"]),
                    in_mask=in_mask)
