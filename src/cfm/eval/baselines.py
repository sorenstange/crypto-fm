"""Baselines (plan.md §11), all fitted on train only.

Returns (in z-space, i.e. already scaled by the EWMA sigma):
- marginal: the unconditional train distribution over the 66 bins (Laplace-smoothed);
- gauss:    zero-mean Gaussian with the EWMA sigma, i.e. N(0, 1) in z-space;
- gauss_cal: N(0, s_h^2) with s_h the train RMS of z_h (calibrated width).
Volatility (variance of the deseasonalised RV_h; each with a train-fitted multiplicative
calibration constant c = mean(RV / forecast), the QLIKE-optimal one):
- ewma:     h * sigma_t^2 (uncalibrated reference) and ewma_cal;
- har:      log-HAR, ln(RV_h/h) on ln of trailing mean variance over 15 min / 4 h / 1 day,
            pooled OLS, smearing-corrected (= its calibration);
- garch:    GARCH(1,1) per symbol on deseasonalised 1-min returns (last year of train),
            filtered forward with fixed parameters, h-step variance sum.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from cfm.eval.metrics import RV_FLOOR

HAR_WINDOWS = [15, 240, 1440]


# --- returns --------------------------------------------------------------------------

def fit_marginal(cls: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(cls[cls >= 0], minlength=n_classes).astype(float) + 1.0
    return counts / counts.sum()


def gauss_cdf(edges: np.ndarray, s: float = 1.0) -> np.ndarray:
    return norm.cdf(edges / s)


def probs_to_cdf(p: np.ndarray) -> np.ndarray:
    """Class probabilities [.., E+1] -> CDF at the E inner edges."""
    return np.cumsum(p, axis=-1)[..., :-1]


# --- volatility -----------------------------------------------------------------------

def har_design(trailing: np.ndarray) -> np.ndarray:
    """[n, 3] trailing mean per-minute variances -> [n, 4] design (1, logs)."""
    return np.column_stack([np.ones(len(trailing)), np.log(np.maximum(trailing, RV_FLOOR))])


def fit_har(trailing: np.ndarray, rv: np.ndarray, h: int) -> dict:
    ok = np.isfinite(rv) & np.isfinite(trailing).all(1)
    X = har_design(trailing[ok])
    y = np.log(np.maximum(rv[ok] / h, RV_FLOOR))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = float(np.mean(rv[ok] / (h * np.exp(X @ beta))))
    return {"beta": beta.tolist(), "c": c}


def predict_har(model: dict, trailing: np.ndarray, h: int) -> np.ndarray:
    return h * np.exp(har_design(trailing) @ np.asarray(model["beta"])) * model["c"]


def fit_calibration(rv: np.ndarray, var_hat: np.ndarray) -> float:
    ok = np.isfinite(rv) & np.isfinite(var_hat) & (var_hat > 0)
    return float(np.mean(rv[ok] / var_hat[ok]))


def fit_garch(r_bp: np.ndarray) -> dict | None:
    """GARCH(1,1), zero mean, normal; r in basis points (deseasonalised 1-min returns)."""
    from arch import arch_model

    r = r_bp[np.isfinite(r_bp)]
    if len(r) < 30 * 1440:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",
                         rescale=False).fit(disp="off", options={"maxiter": 300})
    p = res.params
    omega, alpha, beta = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta <= 1):
        return None
    return {"omega": omega, "alpha": alpha, "beta": beta, "var0": float(np.mean(r**2)),
            "n": len(r), "converged": int(res.convergence_flag == 0)}


def garch_forecast(r_bp: np.ndarray, starts: list[tuple[int, int]], g: dict,
                   horizons: list[int]) -> dict[int, np.ndarray]:
    """Sum of the next h one-minute GARCH variances made at the end of each row
    (bp^2 -> variance units).

    Each segment starts from the train sample variance `var0`, which also stands in for
    NaN returns (segment starts). The h-step sum is written as
    nxt * sum_k phi^k + omega * sum_k sum_{j<k} phi^j (k < h), which stays stable for
    near-integrated fits (alpha + beta ~ 1, common on 1-minute data)."""
    om, al, be, v0 = g["omega"], g["alpha"], g["beta"], g["var0"]
    phi = al + be
    nxt = np.empty(len(r_bp))
    for a, b in starts:
        r2 = np.where(np.isfinite(r_bp[a:b]), r_bp[a:b] ** 2, v0)
        # sigma2_{t+1|t} = om + al r_t^2 + be sigma2_{t|t-1}, sigma2_{a|a-1} = var0
        nxt[a:b], _ = lfilter([1.0], [1.0, -be], om + al * r2, zi=[be * v0])
    powers = phi ** np.arange(max(horizons))
    inner = np.r_[0.0, np.cumsum(powers)[:-1]]          # sum_{j<k} phi^j for k = 0..H-1
    out = {}
    for h in horizons:
        out[h] = (nxt * powers[:h].sum() + om * inner[:h].sum()) / 1e8
    return out
