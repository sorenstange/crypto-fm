"""Baseline evaluation on the validation split (milestone M2).

    python -m cfm.eval.evaluate baselines [--workers 8] [--symbols ...]

1. Per symbol (parallel): cleaned grid -> volatility -> targets at the prediction points
   (patch ends after the warm-up) of train and val; trailing variances for HAR; a GARCH
   fit on the last year of train and its forecasts. Test points are never collected.
2. Fit the baselines on the train points (`baselines.py`), score the val points: CE over
   the 66 bins, CRPS, PIT calibration, rank IC (n/a for these baselines) and QLIKE /
   log-MAE for volatility; 95% day-block-bootstrap CIs and paired gains.
3. Breakdowns by liquidity group, volatility regime and month.
4. Write $DATA_ROOT/eval/baselines/ (+ fitted params in $DATA_ROOT/stats/baselines.json)
   and log everything to W&B (group `baselines`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import yaml

from cfm.data.binance import data_root
from cfm.data.clean import load_clean, minutes, processed_dir
from cfm.data.config import PROJECT_ROOT, load_data_config
from cfm.data.download import universe_symbols
from cfm.data.features import _trailing_mean, load_stats, stats_dir
from cfm.data.splits import prediction_points, split_codes
from cfm.data.targets import bin_edges, make_targets, n_classes, to_bins
from cfm.data.vol import segment_bounds, volatility
from cfm.eval import baselines as bl
from cfm.eval import metrics as mt

GARCH_TRAIN_DAYS = 365


def eval_dir():
    d = data_root() / "eval" / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect_symbol(symbol: str, segments: pd.DataFrame, season: np.ndarray, cfg: dict) -> dict:
    """Per-point arrays of one symbol at its train/val prediction points + GARCH params."""
    df = load_clean(symbol, segments)
    m = minutes(df["ts"])
    seg = df["segment_id"].to_numpy()
    vol = volatility(df, m, season, cfg)
    close = df["close"].to_numpy()
    tg = make_targets(close, vol, seg, cfg)
    split = split_codes(df["ts"], cfg)
    sel = prediction_points(m, seg, cfg) & ((split == 0) | (split == 1))

    vd = vol["v"] / vol["season"]
    trailing = np.full((len(df), len(bl.HAR_WINDOWS)), np.nan)
    r_bp = np.full(len(df), np.nan)
    bounds = segment_bounds(seg)
    for a, b in bounds:
        for j, w in enumerate(bl.HAR_WINDOWS):
            trailing[a:b, j] = _trailing_mean(vd[a:b], w, include_current=True)
        r_bp[a + 1:b] = np.log(close[a + 1:b] / close[a:b - 1]) / np.sqrt(vol["season"][a + 1:b]) * 1e4

    # GARCH on the symbol's own last GARCH_TRAIN_DAYS of train data.
    train = split == 0
    last_train = df["ts"][train].max() if train.any() else None
    fit_rows = train & (df["ts"] > last_train - pd.Timedelta(days=GARCH_TRAIN_DAYS)).to_numpy() \
        if last_train is not None else train
    garch = bl.fit_garch(r_bp[fit_rows]) if fit_rows.sum() else None
    vh = cfg["targets"]["vol_horizons"]
    gf = bl.garch_forecast(r_bp, bounds, garch, vh) if garch else {h: np.full(len(df), np.nan) for h in vh}

    out = {"minute": m[sel], "split": split[sel], "sigma2": vol["sigma"][sel] ** 2,
           "trailing": trailing[sel]}
    for h, z in tg["z"].items():
        out[f"z{h}"] = z[sel]
    for h in vh:
        out[f"rv{h}"] = tg["rv"][h][sel]
        out[f"garch{h}"] = gf[h][sel]
    return {"symbol": symbol, "points": out, "garch": garch}


def _concat(parts: list[dict], universe_rank: dict) -> dict:
    keys = parts[0]["points"].keys()
    data = {k: np.concatenate([p["points"][k] for p in parts]) for k in keys}
    data["sym"] = np.concatenate([np.full(len(p["points"]["minute"]), i, np.int32)
                                  for i, p in enumerate(parts)])
    data["rank"] = np.concatenate([np.full(len(p["points"]["minute"]), universe_rank[p["symbol"]])
                                   for p in parts])
    return data


def _boot_table(day, losses, baseline, **tags):
    t = mt.day_bootstrap(day, losses, baseline)
    for k, v in tags.items():
        t.insert(0, k, v)
    return t


def evaluate_returns(tr: dict, va: dict, cfg: dict, rng) -> tuple[pd.DataFrame, dict, list]:
    edges = bin_edges(cfg)
    K = n_classes(cfg)
    day = va["minute"] // 1440
    tables, params, extra = [], {}, []
    for h in cfg["targets"]["return_horizons"]:
        ztr, zva = tr[f"z{h}"], va[f"z{h}"]
        ztr = ztr[np.isfinite(ztr)]
        ok = np.isfinite(zva)
        z, d = zva[ok], day[ok]
        cls = to_bins(z, edges).astype(np.int64)
        p_marg = bl.fit_marginal(to_bins(ztr, edges).astype(np.int64), K)
        s_cal = float(np.sqrt(np.mean(ztr**2)))
        models = {"marginal": bl.probs_to_cdf(p_marg), "gauss": bl.gauss_cdf(edges),
                  "gauss_cal": bl.gauss_cdf(edges, s_cal)}
        params[f"h{h}"] = {"marginal": p_marg.tolist(), "gauss_cal_s": s_cal,
                           "train_points": int(len(ztr))}
        ce, crps = {}, {}
        for name, cdf in models.items():
            p = mt.cdf_to_probs(cdf[None, :])[0]
            ce[name] = mt.cross_entropy(p[cls])
            crps[name] = np.concatenate([mt.crps_binned(np.broadcast_to(cdf, (len(c), len(cdf))), c, edges)
                                         for c in np.array_split(z, max(1, len(z) // 500_000))])
            u = mt.randomized_pit(cdf, cls, rng)
            extra.append({"horizon": h, "model": name, "pit_max_dev": mt.pit_deviation(u),
                          "ic_pooled": float("nan"), "ic_cross": float("nan"), "points": int(len(z))})
        tables.append(_boot_table(d, ce, "marginal", metric="ce", horizon=h))
        tables.append(_boot_table(d, crps, "marginal", metric="crps", horizon=h))
    return pd.concat(tables, ignore_index=True), params, extra


def vol_forecasts(tr: dict, va: dict, cfg: dict) -> tuple[dict, dict]:
    """{h: {model: (train forecast, val forecast)}} and fitted params."""
    out, params = {}, {}
    for h in cfg["targets"]["vol_horizons"]:
        rtr = tr[f"rv{h}"]
        ew_tr, ew_va = h * tr["sigma2"], h * va["sigma2"]
        c_ew = bl.fit_calibration(rtr, ew_tr)
        har = bl.fit_har(tr["trailing"], rtr, h)
        g_tr, g_va = tr[f"garch{h}"], va[f"garch{h}"]
        c_g = bl.fit_calibration(rtr, g_tr)
        out[h] = {"ewma": ew_va, "ewma_cal": c_ew * ew_va,
                  "har": bl.predict_har(har, va["trailing"], h), "garch": c_g * g_va}
        params[f"h{h}"] = {"ewma_c": c_ew, "har": har, "garch_c": c_g}
    return out, params


def evaluate_vol(va: dict, fc: dict, cfg: dict) -> pd.DataFrame:
    day = va["minute"] // 1440
    tables = []
    for h in cfg["targets"]["vol_horizons"]:
        rv = va[f"rv{h}"]
        has_g = np.isfinite(fc[h]["garch"])
        for subset, mask in [("all", np.isfinite(rv)), ("garch_fitted", np.isfinite(rv) & has_g)]:
            models = {k: v for k, v in fc[h].items() if subset != "all" or k != "garch"}
            ql = {k: mt.qlike(rv[mask], v[mask]) for k, v in models.items()}
            lm = {k: mt.log_mae(rv[mask], v[mask]) for k, v in models.items()}
            tag = dict(subset=subset, horizon=h)
            tables.append(_boot_table(day[mask], ql, "ewma", metric="qlike", **tag))
            tables.append(_boot_table(day[mask], lm, "ewma", metric="log_mae", **tag))
            tables[-1]["zero_rv_share"] = float((rv[mask] <= 0).mean())
    return pd.concat(tables, ignore_index=True)


def breakdowns(tr: dict, va: dict, fc: dict, cfg: dict, n_symbols: int) -> pd.DataFrame:
    """CE gain of gauss_cal over marginal (h=15) and QLIKE gain of HAR over EWMA (h=60)
    by liquidity tercile (universe rank), volatility regime (train terciles) and month."""
    edges = bin_edges(cfg)
    z = va["z15"]
    ok = np.isfinite(z)
    ztr = tr["z15"][np.isfinite(tr["z15"])]
    p_marg = bl.fit_marginal(to_bins(ztr, edges).astype(np.int64), n_classes(cfg))
    p_gc = mt.cdf_to_probs(bl.gauss_cdf(edges, float(np.sqrt(np.mean(ztr**2))))[None, :])[0]
    cls = to_bins(np.where(ok, z, 0), edges).astype(np.int64)
    ce_gain = np.where(ok, mt.cross_entropy(p_marg[cls]) - mt.cross_entropy(p_gc[cls]), np.nan)
    rv = va["rv60"]
    q_gain = mt.qlike(rv, fc[60]["ewma"]) - mt.qlike(rv, fc[60]["har"])
    lns = np.log(va["sigma2"]) / 2
    cuts = np.quantile(np.log(tr["sigma2"]) / 2, [1 / 3, 2 / 3])
    groups = {
        "liquidity": np.array(["top", "mid", "low"])[np.minimum((va["rank"] - 1) * 3 // n_symbols, 2)],
        "vol_regime": np.array(["low", "mid", "high"])[np.searchsorted(cuts, lns)],
        "month": pd.to_datetime(va["minute"] * 60_000, unit="ms").strftime("%Y-%m").to_numpy(),
    }
    rows = []
    for gname, g in groups.items():
        for key in np.unique(g):
            sel = g == key
            rows.append({"by": gname, "group": key, "points": int(sel.sum()),
                         "ce_gain_gausscal_vs_marginal_h15": float(np.nanmean(ce_gain[sel])),
                         "qlike_gain_har_vs_ewma_h60": float(np.nanmean(q_gain[sel & np.isfinite(rv)]))})
    return pd.DataFrame(rows)


def log_wandb(cfg: dict, tables: dict, summary: dict) -> str | None:
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb

    from cfm.utils.run import collect_run_info, init_wandb
    run_dir = data_root() / "runs" / "baselines-v1"
    run = init_wandb(run_dir, {"data": cfg}, collect_run_info(), name="baselines-v1",
                     group="baselines", tags=["M2", "baselines"], job_type="eval")
    log = {"trainer/step": 0}
    for name, df in tables.items():
        log[f"tables/{name}"] = wandb.Table(dataframe=df)
    run.log(log)
    for k, v in summary.items():
        run.summary[k] = v
    url = run.url
    run.finish()
    return url


def run_baselines(symbols: list[str], workers: int) -> int:
    cfg = load_data_config()
    stats = load_stats()
    segments = pd.read_parquet(processed_dir() / "segments.parquet")
    u = yaml.safe_load(open(PROJECT_ROOT / "configs" / "universe.yaml"))
    rank = {s["symbol"]: s["rank"] for s in u["symbols"]}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(partial(collect_symbol, segments=segments, season=stats["season"], cfg=cfg),
                              symbols, chunksize=1))
    data = _concat(parts, rank)
    tr = {k: v[data["split"] == 0] for k, v in data.items()}
    va = {k: v[data["split"] == 1] for k, v in data.items()}
    print(f"[baselines] collected {len(tr['minute']):,} train / {len(va['minute']):,} val points "
          f"from {len(parts)} symbols ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    ret_table, ret_params, ret_extra = evaluate_returns(tr, va, cfg, rng)
    fc, vol_params = vol_forecasts(tr, va, cfg)
    vol_table = evaluate_vol(va, fc, cfg)
    brk = breakdowns(tr, va, fc, cfg, len(rank))
    garch = {p["symbol"]: p["garch"] for p in parts}
    n_g = sum(g is not None for g in garch.values())

    out = eval_dir()
    ret_table.to_csv(out / "returns.csv", index=False)
    pd.DataFrame(ret_extra).to_csv(out / "returns_calibration.csv", index=False)
    vol_table.to_csv(out / "vol.csv", index=False)
    brk.to_csv(out / "breakdown.csv", index=False)
    params = {"returns": ret_params, "vol": vol_params, "garch": garch,
              "train_points": int(len(tr["minute"])), "val_points": int(len(va["minute"]))}
    (stats_dir() / "baselines.json").write_text(json.dumps(params, indent=1, default=float))

    pd.set_option("display.width", 200)
    fmt = lambda df: df.to_string(index=False, float_format=lambda x: f"{x:.4f}")  # noqa: E731
    lines = [f"# Baselines on validation ({cfg['splits']['val_start']} + {cfg['splits']['embargo_days']}d "
             f"embargo … {cfg['splits']['val_end']})",
             f"points: train {len(tr['minute']):,}, val {len(va['minute']):,} (patch ends, after warm-up); "
             f"GARCH fitted for {n_g} of {len(parts)} symbols; {time.time() - t0:.0f}s",
             "", "## Returns: cross-entropy (nats) and CRPS; gain = marginal - model (CI: day bootstrap)",
             fmt(ret_table), "", "## Returns: calibration (max PIT decile deviation from 0.1)",
             fmt(pd.DataFrame(ret_extra)), "",
             "## Volatility: QLIKE and log-MAE; gain = ewma - model", fmt(vol_table), "",
             "## Breakdowns (val means)", fmt(brk)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    summ = {}
    for r in ret_table.itertuples():
        summ[f"val/{r.metric}_h{r.horizon}/{r.model}"] = r.mean
    for r in vol_table[vol_table["subset"] == "garch_fitted"].itertuples():
        summ[f"val/{r.metric}_h{r.horizon}/{r.model}"] = r.mean
    url = log_wandb(cfg, {"returns": ret_table, "returns_calibration": pd.DataFrame(ret_extra),
                          "vol": vol_table, "breakdown": brk}, summ)
    print(f"[baselines] W&B: {url}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["baselines"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--symbols", nargs="+")
    args = p.parse_args(argv)
    return run_baselines(args.symbols or universe_symbols(), args.workers)


if __name__ == "__main__":
    sys.exit(main())
