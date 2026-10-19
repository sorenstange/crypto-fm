"""Universe selection for Binance USD-M USDT perpetuals (plan.md §2.2).

Two steps:

    python -m cfm.data.universe scan      # network: list, classify, fetch 1d klines
    python -m cfm.data.universe select    # offline: rank, cut, write configs/universe.yaml

`scan` lists every symbol in the bucket (incl. delisted ones), classifies it, and for
every candidate lists its 1m monthly files and downloads all its 1d klines. Outputs
under $DATA_ROOT/raw/universe/:
    exchange_info.json   snapshot of /fapi/v1/exchangeInfo (contract and underlying types)
    candidates.csv       one row per bucket symbol: exclusion reason or per-symbol stats
    klines_1d.parquet    daily klines of all candidates (ranking, later sanity checks)

`select` ranks the candidates by median daily quote volume, drops those with too little
history, optionally cuts at a candle budget / liquidity floor, and writes the result
with the reason for every exclusion to configs/universe.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfm.data import binance as bn

MINUTES_PER_DAY = 1440
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Contracts without an exchangeInfo entry (delisted) can't be classified from it.
# Index / basket contracts, known by name:
INDEX_BASES = {"BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL", "ALL"}
# Bases pegged to a fiat currency: no crypto price dynamics.
STABLE_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "USD1", "PYUSD",
                "USDS", "RLUSD", "EUR", "AEUR", "EURI"}
# Gold-backed tokens: commodity dynamics, not crypto (exchangeInfo lists them as COIN).
GOLD_BASES = {"PAXG", "XAUT"}


def classify(symbol: str, info: dict | None) -> str | None:
    """Exclusion reason for a bucket symbol, or None if it is a candidate.

    `info` is the symbol's exchangeInfo entry (None for delisted symbols).
    """
    if "SETTLED" in symbol:
        return "settled/relisted fragment or duplicate of the main symbol (*SETTLED)"
    if re.search(r"_\d{6}$", symbol):
        return "quarterly delivery contract"
    if not symbol.endswith("USDT"):
        return "not USDT-margined (USDC/BUSD/BTC/USD1/... quote)"
    base = symbol.removesuffix("USDT")
    if info is not None and info.get("contractType") == "TRADIFI_PERPETUAL":
        return f"TradFi perpetual ({info.get('underlyingType')}: equity/ETF/commodity/FX/pre-IPO)"
    if (info is not None and info.get("underlyingType") == "INDEX") or base in INDEX_BASES:
        return "index/basket contract"
    if base in STABLE_BASES:
        return "stablecoin vs USDT"
    if base in GOLD_BASES:
        return "gold-backed token (commodity dynamics)"
    if info is not None and info.get("underlyingType") not in (None, "COIN"):
        return f"non-crypto underlying ({info.get('underlyingType')})"
    return None


def _out_dir() -> Path:
    d = bn.data_root() / "raw" / "universe"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- scan -----------------------------------------------------------------------------

def scan_symbol(symbol: str) -> tuple[dict, pd.DataFrame]:
    """1m monthly file listing + all 1d klines of one symbol -> (stats, 1d frame)."""
    m1 = [k for k in bn.list_keys(bn.kline_prefix(symbol, "1m")) if k.key.endswith(".zip")]
    d1 = [k for k in bn.list_keys(bn.kline_prefix(symbol, "1d")) if k.key.endswith(".zip")]
    frames = []
    for k in d1:
        data = bn.get(k.url)
        if data is not None:
            frames.append(bn.read_klines_csv(bn.read_zip_csv(data)))
    months = sorted(bn.period_of(k.key) for k in m1)
    stats = {"n_1m_months": len(m1), "zip_mb_1m": sum(k.size for k in m1) / 1e6,
             "first_month_1m": months[0] if months else None,
             "last_month_1m": months[-1] if months else None,
             "n_1d_files": len(d1), "n_1d_missing": len(d1) - len(frames)}
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(daily):
        daily = daily.drop_duplicates("open_time").sort_values("open_time", ignore_index=True)
        day = pd.to_datetime(daily["open_time"], unit="ms", utc=True)
        qv = daily["quote_volume"].to_numpy()
        last_year = (day >= day.iloc[-1] - pd.Timedelta(days=365)).to_numpy()
        stats |= {"first_day": day.iloc[0].date().isoformat(),
                  "last_day": day.iloc[-1].date().isoformat(),
                  "n_days": len(daily),
                  "zero_volume_days": int((qv == 0).sum()),
                  "est_candles": len(daily) * MINUTES_PER_DAY,
                  "median_daily_qv": float(np.median(qv)),
                  "median_daily_qv_last365": float(np.median(qv[last_year]))}
        daily.insert(0, "symbol", symbol)
    return stats, daily


def scan(workers: int) -> None:
    out = _out_dir()
    t0 = time.time()
    exinfo_raw = bn.get(bn.EXCHANGE_INFO_URL)
    (out / "exchange_info.json").write_bytes(exinfo_raw)
    exinfo = {s["symbol"]: s for s in json.loads(exinfo_raw)["symbols"]}

    symbols = sorted(bn.list_prefixes(f"{bn.UM}/monthly/klines/"))
    rows = {}
    for s in symbols:
        info = exinfo.get(s)
        rows[s] = {"symbol": s, "exclusion": classify(s, info),
                   "status": info["status"] if info else "DELISTED",
                   "underlying_type": info.get("underlyingType") if info else None,
                   "onboard_date": (datetime.fromtimestamp(info["onboardDate"] / 1000, timezone.utc)
                                    .date().isoformat() if info and info.get("onboardDate") else None)}
    candidates = [s for s in symbols if rows[s]["exclusion"] is None]
    print(f"[universe] {len(symbols)} bucket symbols, {len(candidates)} candidates after "
          f"exclusions; scanning with {workers} workers", flush=True)

    dailies, failed = [], []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(scan_symbol, s): s for s in candidates}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                stats, daily = f.result()
            except Exception as e:  # noqa: BLE001 -- report and continue, rerun fixes it
                failed.append(s)
                rows[s]["exclusion"] = f"scan failed: {type(e).__name__}: {e}"[:200]
                continue
            rows[s] |= stats
            if len(daily):
                dailies.append(daily)
            if i % 50 == 0 or i == len(futs):
                print(f"[universe] {i}/{len(futs)} scanned ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(list(rows.values())).to_csv(out / "candidates.csv", index=False)
    pd.concat(dailies, ignore_index=True).to_parquet(out / "klines_1d.parquet", index=False)
    print(f"[universe] wrote {out}/candidates.csv, klines_1d.parquet "
          f"({time.time() - t0:.0f}s); failed: {failed or 'none'}", flush=True)
    if failed:
        sys.exit(1)


# --- select ---------------------------------------------------------------------------

def select(min_days: int, max_candles: float | None, min_median_qv: float,
           out_yaml: Path) -> pd.DataFrame:
    df = pd.read_csv(_out_dir() / "candidates.csv")
    df["exclusion"] = df["exclusion"].astype(object)
    pending = df["exclusion"].isna()
    df.loc[pending & (df["n_1m_months"].fillna(0) == 0), "exclusion"] = "no 1m kline files"
    pending = df["exclusion"].isna()
    df.loc[pending & (df["n_days"].fillna(0) < min_days), "exclusion"] = \
        f"history < {min_days} days"
    pending = df["exclusion"].isna()
    df.loc[pending & (df["median_daily_qv"] < min_median_qv), "exclusion"] = \
        f"median daily quote volume < {min_median_qv:,.0f} USDT"

    ranked = (df[df["exclusion"].isna()]
              .sort_values("median_daily_qv", ascending=False, ignore_index=True))
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["cum_candles"] = ranked["est_candles"].cumsum()
    ranked["cum_zip_gb"] = ranked["zip_mb_1m"].cumsum() / 1e3
    if max_candles is not None:
        keep = (ranked["cum_candles"] - ranked["est_candles"]) < max_candles
        cut = ranked.loc[~keep, "symbol"]
        df.loc[df["symbol"].isin(cut), "exclusion"] = \
            f"below liquidity cut-off (candle budget {max_candles:,.0f})"
        ranked = ranked[keep]

    def entry(r) -> dict:
        return {"symbol": r.symbol, "rank": int(r.rank), "status": r.status,
                "first_day": r.first_day, "last_day": r.last_day, "n_days": int(r.n_days),
                "est_candles": int(r.est_candles),
                "median_daily_qv_musd": round(float(r.median_daily_qv) / 1e6, 2),
                "zip_mb_1m": round(float(r.zip_mb_1m), 1)}

    excluded = {}
    for reason, g in df[df["exclusion"].notna()].groupby("exclusion"):
        excluded[reason] = sorted(g["symbol"])
    delisted = ranked[ranked["status"] != "TRADING"]
    doc = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"{bn.LIST_URL} ({bn.UM}/monthly/klines), {bn.EXCHANGE_INFO_URL}",
        "rules": {
            "keep": "USDT-margined perpetuals with a COIN underlying, incl. delisted ones",
            "min_history_days": min_days,
            "rank_by": "median daily quote volume (USDT) over the symbol's whole history",
            "max_candles": max_candles, "min_median_daily_qv": min_median_qv,
            "est_candles": "number of 1d klines x 1440 (listing days are partial)",
        },
        "summary": {"n_symbols": len(ranked),
                    "n_not_trading_now": len(delisted),
                    "est_candles": int(ranked["est_candles"].sum()),
                    "zip_gb_1m": round(float(ranked["zip_mb_1m"].sum()) / 1e3, 2)},
        "symbols": [entry(r) for r in ranked.itertuples()],
        "excluded": excluded,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        f.write("# Generated by `python -m cfm.data.universe select`; do not edit by hand.\n")
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=120,
                       default_flow_style=None)
    return ranked


def print_table(ranked: pd.DataFrame) -> None:
    show = set(range(1, 101)) | {r for r in ranked["rank"] if r % 25 == 0} | {len(ranked)}
    print(f"{'rank':>4} {'symbol':<16} {'status':<9} {'first_day':<10} {'days':>5} "
          f"{'med_qv_$M':>9} {'cum_Mcandles':>12} {'cum_zip_GB':>10}")
    for r in ranked.itertuples():
        if r.rank in show:
            print(f"{r.rank:>4} {r.symbol:<16} {r.status[:9]:<9} {r.first_day:<10} {r.n_days:>5.0f} "
                  f"{r.median_daily_qv / 1e6:>9.2f} {r.cum_candles / 1e6:>12.1f} {r.cum_zip_gb:>10.2f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("select")
    s.add_argument("--min-days", type=int, default=365)
    s.add_argument("--max-candles", type=float, default=None,
                   help="candle budget; symbols are added in liquidity order until it is reached")
    s.add_argument("--min-median-qv", type=float, default=0.0,
                   help="liquidity floor: minimum median daily quote volume in USDT")
    s.add_argument("--out", type=Path, default=PROJECT_ROOT / "configs" / "universe.yaml")
    args = p.parse_args(argv)
    if args.cmd == "scan":
        scan(args.workers)
    else:
        ranked = select(args.min_days, args.max_candles, args.min_median_qv, args.out)
        print_table(ranked)
        print(f"[universe] {len(ranked)} symbols, {ranked['est_candles'].sum() / 1e6:.0f}M "
              f"est. candles, {ranked['zip_mb_1m'].sum() / 1e3:.2f} GB of 1m zips -> {args.out}")


if __name__ == "__main__":
    main()
