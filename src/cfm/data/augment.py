"""Training-time sign-flip augmentation (plan.md §16, round 2).

A flipped window looks as if the price path were mirrored (log price negated) while
volume and activity stay the same: return-like features change sign, the upper and lower
wicks swap, taker imbalance and funding change sign (buyers <-> sellers, longs <-> shorts),
the position in the trailing range is mirrored, and the return targets z become -z.
Volatility, volume and calendar features are unchanged, and so are the vol targets.
Exact-zero returns stay in their bin (-0.0 bins like 0.0). Odd relations such as
short-term reversal survive the flip; up/down asymmetries (skew, leverage effect) are
averaged out.
"""

from __future__ import annotations

import torch

from cfm.data.features import FEATURES

ODD = ["body", "ret", "taker_imb", "funding", "range_pos"]
SWAP = ("upper", "lower")


def flip_tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(feature permutation, sign) over FEATURES."""
    perm = list(range(len(FEATURES)))
    a, b = FEATURES.index(SWAP[0]), FEATURES.index(SWAP[1])
    perm[a], perm[b] = b, a
    sign = torch.tensor([-1.0 if f in ODD else 1.0 for f in FEATURES], device=device, dtype=dtype)
    return torch.tensor(perm, device=device), sign


def flip_windows(x: torch.Tensor, z: torch.Tensor, flip: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B, L, P*F], z [B, L, H], flip [B] bool -> (x, z) with the flipped windows mirrored."""
    B, L, D = x.shape
    xf = (x.view(B, L, D // len(sign), len(sign))[..., perm] * sign).reshape(B, L, D)
    f = flip[:, None, None]
    return torch.where(f, xf, x), torch.where(f, -z, z)
