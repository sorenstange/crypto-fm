"""Losses (plan.md §6): CE over return bins, QLIKE for volatility, z-loss.

All computed in float32 and averaged over valid positions only (a target is invalid
when its horizon leaves the segment).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

RV_FLOOR = 1e-12


def bins_of(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """z -> class in [0, len(edges)] (same convention as targets.to_bins, side='right')."""
    return torch.bucketize(z, edges, right=True)


def soft_targets(z: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss-style targets -> [..., E + 1]: N(c, sigma^2) integrated over the bins (tails
    included), centred on the centre c of z's hard bin rather than on z itself. Exact-zero
    returns (~10% at h1) lie on the edge at 0; centring on z would split them 50/50 between
    two bins while the hard CE used for evaluation puts them in the upper one."""
    w = edges[1] - edges[0]
    centers = torch.cat([edges[:1] - w / 2, (edges[:-1] + edges[1:]) / 2, edges[-1:] + w / 2])
    c = centers[bins_of(z, edges)]
    cdf = torch.special.ndtr((edges - c[..., None]) / sigma)
    zeros, ones = torch.zeros_like(cdf[..., :1]), torch.ones_like(cdf[..., :1])
    return torch.cat([zeros, cdf, ones], dim=-1).diff(dim=-1)


def return_losses(logits: torch.Tensor, z: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor,
                  soft_sigma: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits [N, H, K], z [N, H], mask [N] -> (loss per horizon [H], z-loss, hard CE per point [N, H]).

    The loss per horizon is the mean hard CE, or with `soft_sigma` > 0 the mean CE against
    HL-Gauss soft targets (a training regulariser). Invalid targets (NaN z or mask False)
    get CE NaN in the per-point output."""
    logits = logits.float()
    valid = torch.isfinite(z) & mask[:, None]
    cls = bins_of(torch.nan_to_num(z), edges)
    ce = F.cross_entropy(logits.permute(0, 2, 1), cls, reduction="none")          # [N, H]
    w = valid.float()
    if soft_sigma > 0:
        soft = soft_targets(torch.nan_to_num(z), edges, soft_sigma)
        obj = -(soft * torch.log_softmax(logits, dim=-1)).sum(-1)
    else:
        obj = ce
    ce_h = (obj * w).sum(0) / w.sum(0).clamp(min=1)
    lse = torch.logsumexp(logits, dim=-1)
    zl = ((lse**2) * w).sum() / w.sum().clamp(min=1)
    return ce_h, zl, torch.where(valid, ce, torch.nan)


def qlike_losses(u: torch.Tensor, rv: torch.Tensor, sigma2: torch.Tensor, horizons: list[int],
                 mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """u [N, Hv] log-ratio to the EWMA forecast h*sigma2 -> (QLIKE per horizon, per point).

    QLIKE = q e^{-u} - ln(q e^{-u}) - 1 with q = RV / (h sigma2): >= 0, 0 iff exact, and the
    same value as `eval.metrics.qlike(RV, h sigma2 e^u)` (identical RV floor)."""
    u = u.float()
    h = torch.tensor(horizons, device=u.device, dtype=torch.float32)
    valid = torch.isfinite(rv) & mask[:, None]
    q = torch.nan_to_num(rv).clamp(min=RV_FLOOR) / (h * sigma2[:, None])
    a = torch.log(q) - u                       # ln(RV / sigma_hat^2)
    ql = torch.expm1(a) - a                    # = r - ln r - 1, accurate near r = 1
    w = valid.float()
    return (ql * w).sum(0) / w.sum(0).clamp(min=1), torch.where(valid, ql, torch.nan)
