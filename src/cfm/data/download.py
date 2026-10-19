"""Download, verify and convert the raw data of one or more symbols (plan.md §2.3-2.4).

Per symbol:
1. List the 1m monthly kline files and monthly fundingRate files up to `--end`
   (default: the last complete month). Months missing inside the kline range are
   filled from daily files where those exist.
2. Download every zip into memory, verify it against its `.CHECKSUM` (SHA-256;
   re-download on mismatch, a file is `failed` after 3 attempts), parse it.
3. Write $DATA_ROOT/raw/parquet/<SYMBOL>.parquet (compact klines, see convert.py),
   $DATA_ROOT/raw/funding/<SYMBOL>.parquet and, last, the per-file manifest
   $DATA_ROOT/raw/manifest/<SYMBOL>.csv. Zips are never written to disk.

Resumable at symbol level: a symbol whose manifest exists with every file `ok` or
`missing` is skipped (use --force to redo it). The per-symbol manifests are merged into
$DATA_ROOT/raw/manifest.csv by the validation step.

    python -m cfm.data.download --symbols BTCUSDT ETHUSDT
    python -m cfm.data.download --index "$LSB_JOBINDEX"     # n-th symbol of configs/universe.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfm.data import binance as bn
from cfm.data.convert import clean_klines, encode_funding, encode_klines, write_parquet_atomic
from cfm.data.load import funding_path, kline_path, raw_dir, read_funding, read_klines

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_COLUMNS = ["symbol", "data_type", "period", "url", "sha256", "status", "attempts",
                    "rows", "first_ts", "last_ts", "zip_bytes"]
MAX_ATTEMPTS = 3


def last_complete_month(today: date | None = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def universe_symbols(path: Path = PROJECT_ROOT / "configs" / "universe.yaml") -> list[str]:
    with open(path) as f:
        return [s["symbol"] for s in yaml.safe_load(f)["symbols"]]


def _months(first: str, last: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def plan_files(symbol: str, end: str) -> list[tuple[str, str, str]]:
    """(data_type, period, key) for every file to fetch, up to month `end` inclusive."""
    files = []
    monthly = {bn.period_of(k.key): k.key for k in bn.list_keys(bn.kline_prefix(symbol, "1m"))
               if k.key.endswith(".zip") and bn.period_of(k.key) <= end}
    files += [("klines_1m", p, monthly[p]) for p in sorted(monthly)]
    if monthly:
        for month in sorted(set(_months(min(monthly), max(monthly))) - set(monthly)):
            daily = bn.list_keys(f"{bn.kline_prefix(symbol, '1m', 'daily')}{symbol}-1m-{month}-")
            files += [("klines_1m_daily", bn.period_of(k.key), k.key)
                      for k in daily if k.key.endswith(".zip")]
    funding = [k.key for k in bn.list_keys(f"{bn.UM}/monthly/fundingRate/{symbol}/")
               if k.key.endswith(".zip") and bn.period_of(k.key) <= end]
    files += [("funding", bn.period_of(k), k) for k in sorted(funding)]
    return files


def fetch_verified(key: str) -> tuple[bytes | None, dict]:
    """Download a zip and its CHECKSUM; retry on mismatch. Returns (data or None, manifest info)."""
    url = f"{bn.DATA_URL}/{key}"
    info = {"url": url, "sha256": None, "status": "failed", "attempts": 0, "zip_bytes": None}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        info["attempts"] = attempt
        try:
            checksum = bn.get(url + ".CHECKSUM")
            data = bn.get(url)
        except Exception as e:  # noqa: BLE001 -- network errors after urllib3 retries
            info["error"] = f"{type(e).__name__}: {e}"[:200]
            time.sleep(2 ** attempt)
            continue
        if data is None:
            info["status"] = "missing"
            return None, info
        digest = bn.sha256(data)
        info |= {"sha256": digest, "zip_bytes": len(data)}
        if checksum is None:
            info["status"] = "no_checksum"
            return data, info
        if digest == bn.parse_checksum(checksum):
            info["status"] = "ok"
            return data, info
        info["error"] = "checksum mismatch"
        time.sleep(2 ** attempt)
    return None, info


def _fetch_parse(item: tuple[str, str, str]) -> tuple[dict, pd.DataFrame | None]:
    data_type, period, key = item
    data, info = fetch_verified(key)
    row = {"data_type": data_type, "period": period, **info}
    df = None
    if data is not None:
        try:
            csv = bn.read_zip_csv(data)
            if data_type == "funding":
                df = bn.read_funding_csv(csv)
                t = df["calc_time"]
            else:
                df = bn.read_klines_csv(csv)
                t = df["open_time"]
            row |= {"rows": len(df), "first_ts": int(t.min()) if len(df) else None,
                    "last_ts": int(t.max()) if len(df) else None}
        except Exception as e:  # noqa: BLE001 -- corrupt file: record, don't crash the symbol
            row |= {"status": "parse_error", "error": f"{type(e).__name__}: {e}"[:200]}
            df = None
    return row, df


def manifest_path(symbol: str, root: Path) -> Path:
    return root / "manifest" / f"{symbol}.csv"


def is_done(symbol: str, root: Path) -> bool:
    p = manifest_path(symbol, root)
    if not (p.exists() and kline_path(symbol, root).exists()):
        return False
    return pd.read_csv(p)["status"].isin(["ok", "missing"]).all()


def process_symbol(symbol: str, root: Path, end: str, workers: int) -> dict:
    t0 = time.time()
    files = plan_files(symbol, end)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_fetch_parse, files))
    rows = [{"symbol": symbol, **r} for r, _ in results]
    klines = [df for (r, df) in results if df is not None and r["data_type"] != "funding"]
    funding = [df for (r, df) in results if df is not None and r["data_type"] == "funding"]

    summary = {"symbol": symbol, "files": len(files),
               "failed": sum(r["status"] not in ("ok", "missing") for r in rows)}
    if klines:
        df, stats = clean_klines(pd.concat(klines, ignore_index=True))
        summary |= stats
        summary["kline_bytes"] = write_parquet_atomic(encode_klines(df), kline_path(symbol, root))
    if funding:
        fr = pd.concat(funding, ignore_index=True)
        summary["funding_rows"] = len(fr)
        summary["funding_bytes"] = write_parquet_atomic(encode_funding(fr), funding_path(symbol, root),
                                                        delta=False)
    mp = manifest_path(symbol, root)
    mp.parent.mkdir(parents=True, exist_ok=True)
    cols = MANIFEST_COLUMNS + (["error"] if any("error" in r for r in rows) else [])
    pd.DataFrame(rows).reindex(columns=cols).to_csv(mp.with_suffix(".tmp"), index=False)
    mp.with_suffix(".tmp").replace(mp)      # written last: marks the symbol as done
    summary["seconds"] = round(time.time() - t0, 1)
    if summary.get("rows"):
        summary["bytes_per_candle"] = round(summary["kline_bytes"] / summary["rows"], 2)
    return summary


REST_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"


def backfill_funding(symbol: str, root: Path, min_late_hours: float = 24.0) -> dict:
    """Fill funding history the bucket lacks (e.g. before a delist/relist) from the REST API.

    Only the stretch between the first kline and the first bucket funding row is fetched;
    the interval of each REST row is inferred from the spacing to the next settlement.
    Appends a `funding_rest` row to the symbol's manifest.
    """
    first_kline = read_klines(symbol, columns=["ts"], root=root)["ts"].iloc[0]
    have = read_funding(symbol, root)
    first_funding = have["funding_time"].iloc[0]
    late_h = (first_funding - first_kline) / pd.Timedelta(hours=1)
    if late_h <= min_late_hours:
        return {"symbol": symbol, "late_hours": late_h, "added": 0}
    start, end = int(first_kline.timestamp() * 1000), int(first_funding.timestamp() * 1000) - 1
    rows = []
    while start <= end:
        r = bn.session().get(REST_FUNDING, params={"symbol": symbol, "startTime": start,
                                                   "endTime": end, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        time.sleep(0.5)   # fundingRate shares a 500 requests / 5 min / IP limit
    if not rows:
        return {"symbol": symbol, "late_hours": late_h, "added": 0}
    rest = pd.DataFrame({"calc_time": [int(x["fundingTime"]) for x in rows],
                         "last_funding_rate": [float(x["fundingRate"]) for x in rows]})
    t = rest["calc_time"].to_numpy()
    nxt = np.append(t[1:], int(first_funding.timestamp() * 1000))
    rest["funding_interval_hours"] = np.rint((nxt - t) / 3_600_000).clip(1, 8).astype(int)
    old = have.assign(calc_time=have["funding_time"].map(lambda x: x.value // 10**6),
                      last_funding_rate=have["funding_rate"])
    merged = pd.concat([rest, old[["calc_time", "funding_interval_hours", "last_funding_rate"]]],
                       ignore_index=True)
    write_parquet_atomic(encode_funding(merged), funding_path(symbol, root), delta=False)
    mp = manifest_path(symbol, root)
    m = pd.read_csv(mp)
    m = m[m["data_type"] != "funding_rest"]
    period = f"{pd.Timestamp(t[0], unit='ms'):%Y-%m-%d}..{pd.Timestamp(t[-1], unit='ms'):%Y-%m-%d}"
    extra = pd.DataFrame([{"symbol": symbol, "data_type": "funding_rest", "period": period,
                           "url": REST_FUNDING, "status": "ok", "attempts": 1, "rows": len(rest),
                           "first_ts": int(t[0]), "last_ts": int(t[-1])}])
    pd.concat([m, extra], ignore_index=True).to_csv(mp, index=False)
    return {"symbol": symbol, "late_hours": late_h, "added": len(rest), "period": period}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", nargs="+")
    g.add_argument("--index", type=int, nargs="+",
                   help="1-based positions in configs/universe.yaml (LSF $LSB_JOBINDEX)")
    p.add_argument("--end", default=last_complete_month(), help="last month YYYY-MM (inclusive)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--root", type=Path, default=None, help="default $DATA_ROOT/raw")
    p.add_argument("--force", action="store_true")
    p.add_argument("--backfill-funding", action="store_true",
                   help="only fill missing early funding history from the REST API")
    args = p.parse_args(argv)

    root = args.root or raw_dir()
    if args.symbols:
        symbols = args.symbols
    else:
        universe = universe_symbols()
        symbols = [universe[i - 1] for i in args.index if i <= len(universe)]
    if args.backfill_funding:
        for s in symbols:
            print(f"[download] backfill {backfill_funding(s, root)}", flush=True)
        return 0
    failed = 0
    for s in symbols:
        if not args.force and is_done(s, root):
            print(f"[download] {s}: done already, skipping", flush=True)
            continue
        summary = process_symbol(s, root, args.end, args.workers)
        failed += summary["failed"]
        print(f"[download] {summary}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
