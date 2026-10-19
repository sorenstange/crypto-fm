"""Evaluation metrics (plan.md §11).

Return distributions are given as class probabilities over the 66 bins (or, for
parametric baselines, as CDF values at the inner bin edges). Volatility forecasts are
variances in the same deseasonalised units as RV.

Points overlap and symbols are correlated, so confidence intervals come from a day-block
bootstrap: per-day sums are resampled with replacement (`day_bootstrap`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RV_FLOOR = 1e-12      # per-minute variance floor for logs (RV can be exactly 0: flat candles)


def cross_entropy(probs_of_true: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(probs_of_true, 1e-12))


def cdf_to_probs(cdf_edges: np.ndarray) -> np.ndarray:
    """CDF at the inner edges [n, E] -> class probabilities [n, E + 1] (tails included)."""
    n = cdf_edges.shape[0]
    full = np.concatenate([np.zeros((n, 1)), cdf_edges, np.ones((n, 1))], axis=1)
    return np.diff(full, axis=1)


def crps_binned(cdf_edges: np.ndarray, z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """CRPS restricted to [edges[0], edges[-1]]: sum_j (F(e_j) - 1{z <= e_j})^2 * width.

    The tails beyond +-5 are ignored; fine for comparing models on the same points."""
    width = edges[1] - edges[0]
    step = (z[:, None] <= edges[None, :]).astype(np.float64)
    return ((cdf_edges - step) ** 2).sum(axis=1) * width


def randomized_pit(cdf_edges: np.ndarray, cls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """PIT u = F(lower edge) + U * p(class); uniform if the forecast is calibrated.

    `cdf_edges` is [n, E], or [E] for one forecast shared by every point."""
    n = len(cls)
    if cdf_edges.ndim == 1:
        full = np.r_[0.0, cdf_edges, 1.0]
        lo, hi = full[cls], full[cls + 1]
    else:
        full = np.concatenate([np.zeros((n, 1)), cdf_edges, np.ones((n, 1))], axis=1)
        lo, hi = full[np.arange(n), cls], full[np.arange(n), cls + 1]
    return lo + rng.uniform(size=n) * (hi - lo)


def pit_deviation(u: np.ndarray, n_bins: int = 10) -> float:
    """Max |histogram frequency - 1/n_bins| of the PIT values."""
    h, _ = np.histogram(u, bins=n_bins, range=(0, 1))
    return float(np.abs(h / max(len(u), 1) - 1 / n_bins).max())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def rank_ic(pred: np.ndarray, z: np.ndarray, ts: np.ndarray, min_cross: int = 5) -> dict:
    """Pooled time-series Spearman IC and mean cross-sectional IC (per timestamp)."""
    out = {"ic_pooled": spearman(pred, z)}
    df = pd.DataFrame({"ts": ts, "p": pred, "z": z})
    df = df[df.groupby("ts")["z"].transform("size") >= min_cross]
    if len(df) == 0 or np.ptp(df["p"].to_numpy()) == 0:
        out["ic_cross"] = float("nan")
        return out
    r = df.groupby("ts")[["p", "z"]].rank()
    df["rp"], df["rz"] = r["p"], r["z"]
    ics = df.groupby("ts")[["rp", "rz"]].corr().xs("rp", level=1)["rz"]
    out["ic_cross"] = float(ics.mean())
    return out


def qlike(rv: np.ndarray, var_hat: np.ndarray) -> np.ndarray:
    """RV/sigma^2 - ln(RV/sigma^2) - 1 (>= 0, 0 iff exact). RV floored for the log; the
    floor shifts every model's loss by the same amount on the same points."""
    rv = np.maximum(rv, RV_FLOOR)
    q = rv / var_hat
    return q - np.log(q) - 1


def log_mae(rv: np.ndarray, var_hat: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(rv, RV_FLOOR)) - np.log(var_hat))


def day_bootstrap(day: np.ndarray, losses: dict[str, np.ndarray], baseline: str | None = None,
                  n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean loss per model with a 95% day-block-bootstrap CI, and the paired difference
    to `baseline` (positive = better than the baseline for a loss)."""
    days, inv = np.unique(day, return_inverse=True)
    count = np.bincount(inv, minlength=len(days)).astype(float)
    sums = {k: np.bincount(inv, v, minlength=len(days)) for k, v in losses.items()}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_boot).astype(float)
    rows = []
    for k, s in sums.items():
        boot = (w @ s) / (w @ count)
        row = {"model": k, "mean": s.sum() / count.sum(),
               "ci_lo": np.quantile(boot, 0.025), "ci_hi": np.quantile(boot, 0.975)}
        if baseline is not None:
            d = sums[baseline] - s
            bd = (w @ d) / (w @ count)
            row |= {"gain": d.sum() / count.sum(), "gain_ci_lo": np.quantile(bd, 0.025),
                    "gain_ci_hi": np.quantile(bd, 0.975)}
        rows.append(row)
    return pd.DataFrame(rows)
