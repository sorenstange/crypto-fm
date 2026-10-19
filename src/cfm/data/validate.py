"""Validate the raw store (plan.md §2.4, M0 acceptance).

    python -m cfm.data.validate [--spot-days 10] [--spot-symbols 4]

1. Merge the per-symbol manifests into $DATA_ROOT/raw/manifest.csv and summarise file
   statuses (every file must be `ok` or `missing`).
2. Per symbol: rows vs. minutes per month, duplicates, gap statistics, OHLC validity,
   zero-volume share, funding coverage (funding gaps longer than the stated interval,
   klines without any funding) -> $DATA_ROOT/raw/validation/report.csv and monthly.csv.
3. REST spot-checks: random full days of a few symbols compared field by field against
   https://fapi.binance.com/fapi/v1/klines -> $DATA_ROOT/raw/validation/spotcheck.csv.
4. Print a summary (also written to validation/summary.txt).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.data import binance as bn
from cfm.data.download import universe_symbols
from cfm.data.load import kline_path, raw_dir, read_funding, read_klines

REST_KLINES = "https://fapi.binance.com/fapi/v1/klines"
MINUTE = pd.Timedelta(minutes=1)
COMPARE = ["open", "high", "low", "close", "volume", "n_trades", "taker_buy_base"]


def merge_manifests(root: Path) -> pd.DataFrame:
    parts = sorted((root / "manifest").glob("*.csv"))
    m = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    m.to_csv(root / "manifest.csv", index=False)
    return m


def validate_symbol(symbol: str) -> tuple[dict, pd.DataFrame]:
    k = read_klines(symbol)
    ts = k["ts"]
    step = ts.diff().iloc[1:]
    gaps = (step[step > MINUTE] / MINUTE - 1).astype(int)          # missing minutes per gap
    o, h, l, c = (k[x].to_numpy() for x in ["open", "high", "low", "close"])
    rep = {
        "symbol": symbol, "rows": len(k),
        "first_ts": ts.iloc[0], "last_ts": ts.iloc[-1],
        "span_minutes": int((ts.iloc[-1] - ts.iloc[0]) / MINUTE) + 1,
        "duplicates": int(ts.duplicated().sum()),
        "non_monotonic": int((step <= pd.Timedelta(0)).sum()),
        "n_gaps": len(gaps), "missing_minutes": int(gaps.sum()),
        "gaps_le_15m": int((gaps <= 15).sum()), "max_gap_minutes": int(gaps.max()) if len(gaps) else 0,
        "ohlc_invalid": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (l <= 0)).sum()),
        "nan_values": int(k[COMPARE].isna().sum().sum()),
        "zero_volume_share": float((k["volume"] == 0).mean()),
        "taker_gt_volume": int((k["taker_buy_base"] > k["volume"] + 1e-9).sum()),
    }
    rep["coverage"] = rep["rows"] / rep["span_minutes"]

    month = ts.dt.strftime("%Y-%m")
    monthly = month.value_counts().rename("rows").rename_axis("month").reset_index()
    monthly["minutes"] = [pd.Period(m, "M").days_in_month * 1440 for m in monthly["month"]]
    monthly.insert(0, "symbol", symbol)

    try:
        f = read_funding(symbol)
    except FileNotFoundError:
        rep["funding_rows"] = 0
        return rep, monthly
    ft = f["funding_time"]
    inside = f[(ft >= ts.iloc[0]) & (ft <= ts.iloc[-1])]
    # At an interval change (e.g. 4h -> 1h) the spacing follows either neighbour's interval.
    iv = inside["funding_interval_hours"].to_numpy()
    spacing = inside["funding_time"].diff().iloc[1:].to_numpy()
    longest = pd.to_timedelta(np.maximum(iv[1:], iv[:-1]), unit="h")
    shortest = pd.to_timedelta(np.minimum(iv[1:], iv[:-1]), unit="h")
    rep |= {"funding_rows": len(f), "funding_first": ft.iloc[0], "funding_last": ft.iloc[-1],
            "funding_intervals": ",".join(map(str, sorted(f["funding_interval_hours"].unique()))),
            # gap: a settlement is missing; irregular: settlements closer than either interval.
            "funding_gaps": int((spacing - longest > pd.Timedelta(minutes=1)).sum()),
            "funding_irregular": int((shortest - spacing > pd.Timedelta(minutes=1)).sum()),
            "funding_starts_late_h": max(0.0, (ft.iloc[0] - ts.iloc[0]) / pd.Timedelta(hours=1)),
            "funding_ends_early_h": max(0.0, (ts.iloc[-1] - ft.iloc[-1]) / pd.Timedelta(hours=1)),
            "abs_rate_max": float(f["funding_rate"].abs().max())}
    return rep, monthly


def spot_check(symbols: list[str], days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Compare random full days against the REST endpoint (only for trading symbols)."""
    rows = []
    for s in symbols:
        k = read_klines(s)
        candidates = k["ts"].dt.floor("D").unique()[1:-1]          # skip partial first/last day
        for day in rng.choice(candidates, size=min(days, len(candidates)), replace=False):
            day = pd.Timestamp(day)
            start = int(day.timestamp() * 1000)
            r = bn.session().get(REST_KLINES, params={"symbol": s, "interval": "1m", "startTime": start,
                                                     "endTime": start + 86_400_000 - 1, "limit": 1500},
                                 timeout=30)
            if r.status_code != 200:
                rows.append({"symbol": s, "day": day.date(), "status": f"HTTP {r.status_code}"})
                continue
            api = pd.DataFrame(r.json(), columns=bn.KLINE_COLUMNS)
            api = pd.DataFrame({"ts": pd.to_datetime(api["open_time"].astype("int64"), unit="ms", utc=True),
                                **{c: api[c].astype("float64") for c in COMPARE}})
            ours = k[(k["ts"] >= day) & (k["ts"] < day + pd.Timedelta(days=1))]
            j = ours.merge(api, on="ts", how="outer", suffixes=("", "_api"), indicator=True)
            both = j[j["_merge"] == "both"]
            mism = {c: int((both[c].astype("float64") != both[f"{c}_api"]).sum()) for c in COMPARE}
            rows.append({"symbol": s, "day": day.date(), "status": "ok", "rows_ours": len(ours),
                         "rows_api": len(api), "only_ours": int((j["_merge"] == "left_only").sum()),
                         "only_api": int((j["_merge"] == "right_only").sum()),
                         "mismatched_values": sum(mism.values()),
                         **{f"mm_{c}": v for c, v in mism.items()}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spot-days", type=int, default=10)
    p.add_argument("--spot-symbols", type=int, default=4, help="random trading symbols besides BTC/ETH")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    root = raw_dir()
    out = root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    universe = universe_symbols()
    m = merge_manifests(root)
    have = [s for s in universe if kline_path(s).exists()]
    say(f"universe {len(universe)} symbols; parquet for {len(have)}; "
        f"missing: {sorted(set(universe) - set(have)) or 'none'}")
    say(f"manifest {len(m)} files: {m.groupby(['data_type', 'status']).size().to_dict()}")
    bad = m[~m["status"].isin(["ok", "missing"])]
    if len(bad):
        say(f"NOT OK files:\n{bad[['symbol', 'data_type', 'period', 'status']].to_string(index=False)}")

    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(validate_symbol, have, chunksize=4))
    rep = pd.DataFrame([r for r, _ in results])
    monthly = pd.concat([mo for _, mo in results], ignore_index=True)
    rep.to_csv(out / "report.csv", index=False)
    monthly.to_csv(out / "monthly.csv", index=False)

    sizes = sum(kline_path(s).stat().st_size for s in have)
    say(f"candles {rep['rows'].sum():,} in {len(rep)} symbols; kline parquet {sizes / 1e9:.2f} GB "
        f"({sizes / rep['rows'].sum():.1f} B/candle); "
        f"funding parquet {sum(p.stat().st_size for p in (root / 'funding').glob('*.parquet')) / 1e6:.1f} MB")
    say(f"coverage (rows/span) median {rep['coverage'].median():.4f}, min {rep['coverage'].min():.4f} "
        f"({rep.loc[rep['coverage'].idxmin(), 'symbol']}); missing minutes {rep['missing_minutes'].sum():,} "
        f"in {rep['n_gaps'].sum():,} gaps ({rep['gaps_le_15m'].sum():,} of them <= 15 min)")
    for col in ["duplicates", "non_monotonic", "ohlc_invalid", "nan_values", "taker_gt_volume"]:
        n = rep[col].sum()
        say(f"{col}: {n}" + (f"  <- {rep.loc[rep[col] > 0, 'symbol'].tolist()[:10]}" if n else ""))
    say(f"zero-volume share: median {rep['zero_volume_share'].median():.2%}, "
        f"max {rep['zero_volume_share'].max():.2%} ({rep.loc[rep['zero_volume_share'].idxmax(), 'symbol']})")
    nf = rep[rep["funding_rows"] == 0]["symbol"].tolist()
    say(f"funding: missing for {nf or 'no symbol'}; gaps {int(rep['funding_gaps'].sum())} in "
        f"{int((rep['funding_gaps'] > 0).sum())} symbols; irregular {int(rep['funding_irregular'].sum())}; "
        f"starts >24h after klines: {int((rep['funding_starts_late_h'] > 24).sum())}, "
        f"ends >24h before: {int((rep['funding_ends_early_h'] > 24).sum())}; "
        f"intervals seen {sorted(set(','.join(rep['funding_intervals'].dropna()).split(',')))}")

    rng = np.random.default_rng(args.seed)
    trading = [s for s in have if rep.set_index("symbol").loc[s, "last_ts"] >= rep["last_ts"].max()]
    spot_syms = ["BTCUSDT", "ETHUSDT"] + list(rng.choice(
        [s for s in trading if s not in ("BTCUSDT", "ETHUSDT")], size=args.spot_symbols, replace=False))
    sc = spot_check(spot_syms, args.spot_days, rng)
    sc.to_csv(out / "spotcheck.csv", index=False)
    ok = sc[sc["status"] == "ok"]
    say(f"spot-check {len(sc)} symbol-days ({', '.join(spot_syms)}): {len(ok)} fetched, "
        f"value mismatches {int(ok['mismatched_values'].sum())}, rows only in ours "
        f"{int(ok['only_ours'].sum())}, only in API {int(ok['only_api'].sum())}")
    if (ok["mismatched_values"] > 0).any() or (ok["only_api"] > 0).any() or len(ok) < len(sc):
        say(sc[(sc["status"] != "ok") | (sc.get("mismatched_values", 0) > 0) | (sc.get("only_api", 0) > 0)]
            .to_string(index=False))
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
