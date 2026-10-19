"""Causal candle + context features (plan.md §4.2-4.3), computed in RAM.

Features are not stored (plan.md §15.2): `build_features` turns one symbol's cleaned grid
(`clean.load_clean`) and funding into a float array when the dataset is built. Row t uses
only data up to and including candle t (tests/test_features.py perturbs the future and
checks this). Train-only statistics (seasonality, scalers) come from `fit_stats` and live
in $DATA_ROOT/stats/.

Candle features (normalised by scale_t = sigma_{t-1} * sqrt(season_t), squashed by
s*tanh(x/s)):
  body  = ln(C/O)              upper = ln(H/max(O,C))      lower = ln(min(O,C)/L)
  ret   = ln(C_t/C_{t-1})      (0 at a segment start)
  vrel  = ln(1+V) - mean of ln(1+V) over the previous `trailing_minutes` (window ends t-1)
Context:
  lnsig      ln sigma_{t-1} (deseasonalised regime level), z-scored with train stats
  dvol       trailing mean of ln(1 + V*C) (dollar-volume level, window ends at t), z-scored
  taker_imb  2*taker_buy/V - 1 (0 when V = 0)
  trades_rel ln(1+n_trades) - its trailing mean (window ends t-1)
  funding    asinh(last settled funding rate in bp); settled = funding_time <= t
  to_funding fraction of the funding interval left until the next settlement, in [0, 1]
  range_pos  position of C_t in the trailing high/low range (window ends at t), in [-1, 1]
  mod_sin/mod_cos, dow_sin/dow_cos  minute of day, day of week
  filled     1 for forward-filled candles
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.data.binance import data_root
from cfm.data.clean import minutes as to_minutes
from cfm.data.vol import (MINUTES_PER_WEEK, season_accumulate, season_factor, segment_bounds,
                          volatility, candle_variance)

FEATURES = ["body", "upper", "lower", "ret", "vrel", "lnsig", "dvol", "taker_imb", "trades_rel",
            "funding", "to_funding", "range_pos", "mod_sin", "mod_cos", "dow_sin", "dow_cos",
            "filled"]
SQUASHED = ["body", "upper", "lower", "ret", "vrel", "lnsig", "dvol", "trades_rel", "funding"]


def stats_dir() -> Path:
    return data_root() / "stats"


def _trailing_mean(x: np.ndarray, window: int, include_current: bool) -> np.ndarray:
    """Mean over the last `window` values within one segment; excluding the current value
    it is the mean of x[t-window:t] (0 at the segment start)."""
    cs = np.r_[0.0, np.cumsum(x)]
    t = np.arange(len(x))
    hi = t + 1 if include_current else t
    lo = np.maximum(hi - window, 0)
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), 0.0)


def _rolling(x: np.ndarray, window: int, how: str) -> np.ndarray:
    r = pd.Series(x).rolling(window, min_periods=1)
    return (r.max() if how == "max" else r.min()).to_numpy()


def _funding_features(ts_ms: np.ndarray, funding: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    if funding is None or len(funding) == 0:
        return np.zeros(len(ts_ms)), np.zeros(len(ts_ms))
    ft = to_minutes(funding["funding_time"]) * 60_000
    k = np.searchsorted(ft, ts_ms, side="right") - 1          # last settlement <= t
    known = k >= 0
    kk = np.maximum(k, 0)
    rate = np.where(known, funding["funding_rate"].to_numpy()[kk], 0.0)
    interval = funding["funding_interval_hours"].to_numpy().astype(np.int64)[kk] * 3_600_000
    left = (ft[kk] + interval - ts_ms) / interval
    return np.arcsinh(rate * 1e4), np.where(known, np.clip(left, 0.0, 1.0), 0.0)


def build_features(df: pd.DataFrame, funding: pd.DataFrame | None, stats: dict,
                   cfg: dict) -> tuple[np.ndarray, dict]:
    """(features [n, len(FEATURES)], aux arrays incl. the volatility dict) for one symbol."""
    fc = cfg["features"]
    win, sq = fc["trailing_minutes"], fc["squash"]
    m = to_minutes(df["ts"])
    vol = volatility(df, m, stats["season"], cfg)
    scale = vol["scale"]
    o, h, l, c = (df[x].to_numpy() for x in ["open", "high", "low", "close"])
    v = df["volume"].to_numpy()
    n = len(df)
    f = {}
    f["body"] = np.log(c / o) / scale
    f["upper"] = np.log(h / np.maximum(o, c)) / scale
    f["lower"] = np.log(np.minimum(o, c) / l) / scale
    ret = np.zeros(n)
    lv = np.log1p(v)
    lt = np.log1p(df["n_trades"].to_numpy().astype(np.float64))
    ldv = np.log1p(v * c)
    vrel, trel, dvol = np.zeros(n), np.zeros(n), np.zeros(n)
    rpos = np.zeros(n)
    for a, b in segment_bounds(df["segment_id"].to_numpy()):
        ret[a + 1:b] = np.log(c[a + 1:b] / c[a:b - 1])
        vrel[a:b] = lv[a:b] - _trailing_mean(lv[a:b], win, include_current=False)
        trel[a:b] = lt[a:b] - _trailing_mean(lt[a:b], win, include_current=False)
        vrel[a], trel[a] = 0.0, 0.0
        dvol[a:b] = _trailing_mean(ldv[a:b], win, include_current=True)
        hi, lo = _rolling(h[a:b], win, "max"), _rolling(l[a:b], win, "min")
        rng = hi - lo
        rpos[a:b] = np.where(rng > 0, 2 * (c[a:b] - lo) / np.where(rng > 0, rng, 1) - 1, 0.0)
    f["ret"] = ret / scale
    f["vrel"], f["trades_rel"] = vrel, trel
    sc = stats["scalers"]
    f["lnsig"] = (np.log(vol["sigma_prev"]) - sc["lnsig"][0]) / sc["lnsig"][1]
    f["dvol"] = (dvol - sc["dvol"][0]) / sc["dvol"][1]
    f["taker_imb"] = np.where(v > 0, 2 * df["taker_buy_base"].to_numpy() / np.where(v > 0, v, 1) - 1, 0.0)
    ts_ms = m * 60_000
    f["funding"], f["to_funding"] = _funding_features(ts_ms, funding)
    f["range_pos"] = rpos
    mod = 2 * np.pi * (m % 1440) / 1440
    dow = 2 * np.pi * ((m // 1440 + 3) % 7) / 7          # 1970-01-01 = Thursday -> Monday = 0
    f["mod_sin"], f["mod_cos"] = np.sin(mod), np.cos(mod)
    f["dow_sin"], f["dow_cos"] = np.sin(dow), np.cos(dow)
    f["filled"] = df["filled"].to_numpy().astype(np.float64)
    for k in SQUASHED:
        f[k] = sq * np.tanh(f[k] / sq)
    x = np.stack([f[k] for k in FEATURES], axis=1).astype(fc["dtype"])
    return x, {"vol": vol, "minutes": m}


# --- train-only statistics ------------------------------------------------------------

def season_contribution(df: pd.DataFrame, train_end: pd.Timestamp, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """One symbol's (sums, counts) for the pooled seasonality, train rows only, no fills."""
    keep = (df["ts"] <= train_end).to_numpy() & ~df["filled"].to_numpy()
    if keep.sum() < 1440:
        return np.zeros(MINUTES_PER_WEEK), np.zeros(MINUTES_PER_WEEK)
    d = df[keep]
    v = candle_variance(d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(),
                        d["close"].to_numpy(), cfg["vol"]["estimator"])
    return season_accumulate(v, to_minutes(d["ts"]), cfg["vol"]["seasonality"]["winsor_quantile"])


def scaler_contribution(df: pd.DataFrame, funding, season: np.ndarray, train_end: pd.Timestamp,
                        cfg: dict) -> dict:
    """Sums for the train mean/std of lnsig and dvol (rows after the 1440-minute warm-up)."""
    stats = {"season": season, "scalers": {"lnsig": (0.0, 1.0), "dvol": (0.0, 1.0)}}
    cfg = cfg | {"features": cfg["features"] | {"squash": 1e9}}    # unsquashed raw values
    x, _ = build_features(df, funding, stats, cfg)
    warm = np.zeros(len(df), bool)
    for a, b in segment_bounds(df["segment_id"].to_numpy()):
        warm[a:min(a + cfg["features"]["trailing_minutes"], b)] = True
    keep = (df["ts"] <= train_end).to_numpy() & ~warm
    out = {}
    for k in ["lnsig", "dvol"]:
        col = x[keep, FEATURES.index(k)].astype(np.float64)
        out[k] = (len(col), col.sum(), (col**2).sum())
    return out


def save_stats(season: np.ndarray, scalers: dict, meta: dict) -> None:
    d = stats_dir()
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "seasonality.npy", season)
    (d / "stats.json").write_text(json.dumps({"scalers": scalers, **meta}, indent=1, default=str))


def load_stats() -> dict:
    d = stats_dir()
    meta = json.loads((d / "stats.json").read_text())
    return {"season": np.load(d / "seasonality.npy"),
            "scalers": {k: tuple(v) for k, v in meta["scalers"].items()}, "meta": meta}


# --- CLI ------------------------------------------------------------------------------

def _load(symbol: str, segments: pd.DataFrame):
    from cfm.data.clean import load_clean
    from cfm.data.load import read_funding
    return load_clean(symbol, segments), read_funding(symbol)


def _season_job(symbol, segments, train_end, cfg):
    df, _ = _load(symbol, segments)
    return season_contribution(df, train_end, cfg)


def _scaler_job(symbol, segments, season, train_end, cfg):
    df, fr = _load(symbol, segments)
    return scaler_contribution(df, fr, season, train_end, cfg)


def _check_job(symbol, segments, stats, cfg, bounds):
    import time
    t0 = time.time()
    df, fr = _load(symbol, segments)
    x, aux = build_features(df, fr, stats, cfg)
    sec = time.time() - t0
    out = {"symbol": symbol, "rows": len(x), "seconds": sec,
           "nonfinite": int((~np.isfinite(x)).sum()),
           "scale_min": float(aux["vol"]["scale"].min())}
    moments = {}
    for split, (a, b) in bounds.items():
        sel = ((df["ts"] >= a) & (df["ts"] <= b)).to_numpy()
        xs = x[sel].astype(np.float64)
        moments[split] = (len(xs), xs.sum(0), (xs**2).sum(0), xs.min(0, initial=np.inf),
                          xs.max(0, initial=-np.inf))
    return out, moments


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from cfm.data.clean import processed_dir
    from cfm.data.config import load_data_config, split_bounds
    from cfm.data.download import universe_symbols

    p = argparse.ArgumentParser(description="fit-stats: train-only seasonality + scalers; "
                                "check: build all features and summarise them")
    p.add_argument("cmd", choices=["fit-stats", "check"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--symbols", nargs="+")
    args = p.parse_args(argv)
    cfg = load_data_config()
    bounds = split_bounds(cfg)
    train_end = bounds["train"][1]
    segments = pd.read_parquet(processed_dir() / "segments.parquet")
    symbols = args.symbols or universe_symbols()
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        if args.cmd == "fit-stats":
            parts = list(pool.map(partial(_season_job, segments=segments, train_end=train_end, cfg=cfg),
                                  symbols, chunksize=2))
            sums = sum(s for s, _ in parts)
            counts = sum(c for _, c in parts)
            season = season_factor(sums, counts)
            used = [s for s, (_, c) in zip(symbols, parts) if c.sum() > 0]
            print(f"[stats] seasonality from {len(used)} symbols with train data; factor range "
                  f"{season.min():.3f}..{season.max():.3f}; min count/bin {counts.min():.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            parts = list(pool.map(partial(_scaler_job, segments=segments, season=season,
                                          train_end=train_end, cfg=cfg), symbols, chunksize=2))
            scalers = {}
            for k in ["lnsig", "dvol"]:
                n = sum(q[k][0] for q in parts)
                s1 = sum(q[k][1] for q in parts)
                s2 = sum(q[k][2] for q in parts)
                mean = s1 / n
                scalers[k] = (float(mean), float(np.sqrt(s2 / n - mean**2)))
            save_stats(season, scalers, {"train_end": str(train_end), "symbols_with_train": used,
                                         "vol_config": cfg["vol"], "features_config": cfg["features"]})
            print(f"[stats] scalers {scalers} -> {stats_dir()} ({time.time() - t0:.0f}s)", flush=True)
            return 0
        stats = load_stats()
        res = list(pool.map(partial(_check_job, segments=segments, stats=stats, cfg=cfg, bounds=bounds),
                            symbols, chunksize=2))
    rep = pd.DataFrame([r for r, _ in res])
    lines = [f"built features for {len(rep)} symbols, {rep['rows'].sum():,} rows x {len(FEATURES)} "
             f"in {time.time() - t0:.0f}s wall ({rep['seconds'].sum():.0f}s cpu, max "
             f"{rep['seconds'].max():.1f}s/symbol); non-finite values {rep['nonfinite'].sum()}; "
             f"min scale {rep['scale_min'].min():.2e}",
             f"in-RAM size at {cfg['features']['dtype']}: "
             f"{rep['rows'].sum() * len(FEATURES) * np.dtype(cfg['features']['dtype']).itemsize / 1e9:.1f} GB"]
    for split in bounds:
        n = sum(m[split][0] for _, m in res)
        s1 = sum(m[split][1] for _, m in res)
        s2 = sum(m[split][2] for _, m in res)
        lo = np.min([m[split][3] for _, m in res], axis=0)
        hi = np.max([m[split][4] for _, m in res], axis=0)
        mean = s1 / max(n, 1)
        std = np.sqrt(np.maximum(s2 / max(n, 1) - mean**2, 0))
        lines.append(f"{split}: {n:,} rows")
        lines += [f"  {k:<11} mean {mean[i]:+.3f}  std {std[i]:.3f}  min {lo[i]:+.3f}  max {hi[i]:+.3f}"
                  for i, k in enumerate(FEATURES)]
    out = stats_dir() / "feature_check.txt"
    out.write_text("\n".join(lines) + "\n")
    rep.to_csv(stats_dir() / "feature_check.csv", index=False)
    print("\n".join(lines), flush=True)
    return 1 if rep["nonfinite"].sum() else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
