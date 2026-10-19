"""Walk-forward time splits with embargo, and prediction points (plan.md §5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfm.data.config import split_bounds
from cfm.data.vol import segment_bounds

SPLITS = ["train", "val", "test"]
NO_SPLIT = -1


def split_codes(ts: pd.Series, cfg: dict) -> np.ndarray:
    """Per row: 0 train, 1 val, 2 test, -1 embargoed / outside every split.

    The first `embargo_days` of val and test are embargoed, so no window that ends in
    them can reach back into the previous split's prediction targets.
    """
    emb = pd.Timedelta(days=cfg["splits"]["embargo_days"])
    out = np.full(len(ts), NO_SPLIT, np.int8)
    for code, (name, (a, b)) in enumerate(split_bounds(cfg).items()):
        if name != "train":
            a = a + emb
        out[((ts >= a) & (ts <= b)).to_numpy()] = code
    return out


def prediction_points(minutes: np.ndarray, segment_id: np.ndarray, cfg: dict) -> np.ndarray:
    """Mask of rows at which a prediction is made: the last candle of a clock-aligned patch
    with at least `warmup_minutes` of segment history before it."""
    tc = cfg["targets"]
    p = tc["patch_minutes"]
    pos = np.empty(len(minutes), np.int64)
    for a, b in segment_bounds(segment_id):
        pos[a:b] = np.arange(b - a)
    return ((minutes + 1) % p == 0) & (pos >= tc["warmup_minutes"])
