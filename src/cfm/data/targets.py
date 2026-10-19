"""Prediction targets (plan.md §6).

At row t (the end of a patch) the targets look forward from the close of candle t:
- returns: z_h = ln(C_{t+h} / C_t) / (sigma_t * sqrt(sum_{k=1..h} season_{t+k})),
  where sigma_t is the deseasonalised EWMA sigma known at the end of candle t and the
  seasonal factors are calendar values known in advance; z_h ~ unit variance when
  sigma is a good forecast. Binned into 64 uniform bins on [lo, hi] + 2 tail bins.
- volatility: RV_h = sum_{k=1..h} v_{t+k} / season_{t+k} (deseasonalised range-based
  realised variance), predicted as log sigma^2_h and scored with QLIKE.
Targets whose horizon leaves the segment are NaN.
"""

from __future__ import annotations

import numpy as np

from cfm.data.vol import segment_bounds


def bin_edges(cfg: dict) -> np.ndarray:
    """The n_inner + 1 inner edges; class 0 is (-inf, lo), class n_inner + 1 is [hi, inf)."""
    b = cfg["targets"]["bins"]
    return np.linspace(b["lo"], b["hi"], b["n_inner"] + 1)


def n_classes(cfg: dict) -> int:
    return cfg["targets"]["bins"]["n_inner"] + 2


def to_bins(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """z -> class index in [0, len(edges)]; -1 for NaN."""
    k = np.searchsorted(edges, z, side="right")
    return np.where(np.isnan(z), -1, k).astype(np.int16)


def _forward_sum(x: np.ndarray, h: int) -> np.ndarray:
    """s_t = x_{t+1} + ... + x_{t+h} within one segment; NaN where it would leave it."""
    cs = np.r_[0.0, np.cumsum(x)]
    n = len(x)
    out = np.full(n, np.nan)
    if n > h:
        out[:n - h] = cs[1 + h:] - cs[1:n - h + 1]
    return out


def make_targets(close: np.ndarray, vol: dict, segment_id: np.ndarray, cfg: dict) -> dict:
    """{'z': {h: z_h}, 'ret_scale': {h: ...}, 'rv': {h: RV_h}} as float64 arrays per row.

    `vol` is the dict from `vol.volatility` (needs sigma, season, v).
    """
    tc = cfg["targets"]
    n = len(close)
    vd = vol["v"] / vol["season"]
    out = {"z": {}, "ret_scale": {}, "rv": {}}
    for h in tc["return_horizons"]:
        z, scale = np.full(n, np.nan), np.full(n, np.nan)
        for a, b in segment_bounds(segment_id):
            m = b - a - h
            if m <= 0:
                continue
            seas = _forward_sum(vol["season"][a:b], h)[:m]
            scale[a:a + m] = vol["sigma"][a:a + m] * np.sqrt(seas)
            z[a:a + m] = np.log(close[a + h:b] / close[a:a + m]) / scale[a:a + m]
        out["z"][h], out["ret_scale"][h] = z, scale
    for h in tc["vol_horizons"]:
        rv = np.full(n, np.nan)
        for a, b in segment_bounds(segment_id):
            rv[a:b] = _forward_sum(vd[a:b], h)
        out["rv"][h] = rv
    return out
