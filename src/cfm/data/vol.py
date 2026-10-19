"""Per-candle variance, seasonality and causal EWMA volatility (plan.md §4.1).

- Per-candle variance: Garman-Klass (default), Parkinson or Rogers-Satchell.
- Seasonality: one factor per minute of week, pooled over all symbols' train data
  (each symbol's variance normalised by its own train mean and winsorised); mean 1.
  Pooled rather than per-symbol so that symbols listed after the train period (151 of
  371) get a factor too, and because per-candle variance medians are 0 for illiquid
  symbols (many flat candles), the factor uses means of winsorised variance.
- sigma2_t = lam * sigma2_{t-1} + (1 - lam) * v_t / season_t, per segment, lam = 0.5**(1/hl).
- Floor: sigma_t >= max(ratio * slow-EWMA sigma_t, abs_min), a causal adaptive floor (the
  plan's train-only 5th percentile does not exist for symbols listed after train).
- Features at candle t use sigma_{t-1} (known before t), re-seasonalised with season_t,
  which is a calendar function and known in advance: scale_t = sigma_{t-1} * sqrt(season_t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

LN2 = np.log(2.0)
MINUTES_PER_WEEK = 10080
# 1970-01-01 was a Thursday: shift so that Monday 00:00 UTC is minute-of-week 0.
EPOCH_WEEKDAY_OFFSET = 3 * 1440


def candle_variance(o, h, l, c, estimator: str = "garman_klass") -> np.ndarray:
    """Per-candle variance estimate (>= 0) from OHLC."""
    hl, co = np.log(h / l), np.log(c / o)
    if estimator == "garman_klass":
        # >= 0 because |ln C/O| <= ln H/L for valid candles.
        return np.maximum(0.5 * hl**2 - (2 * LN2 - 1) * co**2, 0.0)
    if estimator == "parkinson":
        return hl**2 / (4 * LN2)
    if estimator == "rogers_satchell":
        return np.maximum(np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o), 0.0)
    raise ValueError(f"unknown estimator {estimator!r}")


def minute_of_week(minutes: np.ndarray) -> np.ndarray:
    return (np.asarray(minutes) + EPOCH_WEEKDAY_OFFSET) % MINUTES_PER_WEEK


def season_accumulate(v: np.ndarray, minutes: np.ndarray, winsor_quantile: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """One symbol's contribution to the pooled seasonality: (sums, counts) per minute of week."""
    cap = np.quantile(v, winsor_quantile)
    vw = np.minimum(v, cap)
    mean = vw.mean()
    if not mean > 0:
        return np.zeros(MINUTES_PER_WEEK), np.zeros(MINUTES_PER_WEEK)
    mow = minute_of_week(minutes)
    return (np.bincount(mow, vw / mean, MINUTES_PER_WEEK),
            np.bincount(mow, minlength=MINUTES_PER_WEEK).astype(float))


def season_factor(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    f = sums / np.maximum(counts, 1)
    return f / f[counts > 0].mean()


def ewma(x: np.ndarray, half_life: float, init: float) -> np.ndarray:
    """y_t = lam*y_{t-1} + (1-lam)*x_t with y_{-1} = init (causal)."""
    lam = 0.5 ** (1.0 / half_life)
    y, _ = lfilter([1 - lam], [1, -lam], x, zi=[lam * init])
    return y


def segment_bounds(segment_id: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop), ...] row ranges of consecutive equal segment ids."""
    if len(segment_id) == 0:
        return []
    cut = np.flatnonzero(np.diff(segment_id)) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, len(segment_id)]
    return list(zip(starts, stops))


def volatility(df: pd.DataFrame, minutes: np.ndarray, season: np.ndarray, cfg: dict) -> dict:
    """Causal volatility of one symbol's cleaned grid (from `load_clean`).

    Returns per-row arrays:
      v         per-candle variance
      season    seasonal factor of the candle's minute of week
      sigma     floored deseasonalised EWMA sigma including candle t (known at the end of t)
      sigma_prev  the same before candle t (sigma_{t-1}; initial state at a segment start)
      scale     sigma_prev * sqrt(season): the normaliser for candle t's features/targets
    """
    vc = cfg["vol"]
    fl = vc["floor"]
    v = candle_variance(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                        df["close"].to_numpy(), vc["estimator"])
    s = season[minute_of_week(minutes)]
    vd = v / s
    sigma = np.empty_like(v)
    sigma_prev = np.empty_like(v)
    floor2 = fl["abs_min"] ** 2
    for a, b in segment_bounds(df["segment_id"].to_numpy()):
        init = max(vd[a], floor2)     # the first candle's own variance: no look-ahead
        fast = ewma(vd[a:b], vc["half_life_minutes"], init)
        slow = ewma(vd[a:b], fl["half_life_minutes"], init)
        sig = np.maximum(np.sqrt(fast), np.maximum(fl["ratio"] * np.sqrt(slow), fl["abs_min"]))
        sigma[a:b] = sig
        sigma_prev[a] = max(np.sqrt(init), fl["abs_min"])
        sigma_prev[a + 1:b] = sig[:-1]
    return {"v": v, "season": s, "sigma": sigma, "sigma_prev": sigma_prev,
            "scale": sigma_prev * np.sqrt(s)}
