"""Phase 1 cleaning (plan.md §3) as a segment index.

The cleaned data is never stored as a second copy (plan.md §15.2). `clean_symbol` turns a
symbol's raw klines into a list of segments; `load_clean` rebuilds the cleaned minute
grid from the raw store and that index:
- gaps <= max_fill_gap_minutes inside a segment are forward-filled
  (O=H=L=C=previous close, volume/trades/taker 0, `filled`=True);
- segments are split at longer gaps and at |1-minute log return| > max_abs_log_return
  (close-to-close or open vs. previous close), unless the jump is allowlisted;
- runs of > max_identical_run flat candles (O=H=L=C=previous close) and of
  > max_zero_volume_run zero-volume candles are dropped;
- the first listing_warmup_days after listing, and after any gap >= relist_gap_days, are
  dropped;
- segments shorter than min_segment_minutes are dropped;
- taker_buy_base is clipped to volume (a few upstream rounding errors).

    python -m cfm.data.clean          # all universe symbols -> $DATA_ROOT/processed/
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cfm.data.binance import data_root
from cfm.data.config import PROJECT_ROOT, load_data_config, split_bounds
from cfm.data.download import universe_symbols
from cfm.data.load import read_klines

MINUTE_MS = 60_000
KEEP, LISTING, IDENTICAL, ZERO_VOLUME, SHORT = 0, 1, 2, 3, 4
REASONS = {LISTING: "listing_warmup", IDENTICAL: "identical_run", ZERO_VOLUME: "zero_volume_run",
           SHORT: "short_segment"}
SEGMENT_COLUMNS = ["symbol", "segment_id", "start_ts", "end_ts", "n_minutes", "n_raw_rows", "n_filled"]


def processed_dir() -> Path:
    return data_root() / "processed"


def minutes(ts: pd.Series) -> np.ndarray:
    """tz-aware UTC timestamps -> int64 minutes since the epoch."""
    return ts.dt.tz_localize(None).to_numpy().astype("datetime64[m]").astype(np.int64)


def from_minutes(m: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(m, np.int64) * MINUTE_MS, unit="ms", utc=True)


def long_runs(flag: np.ndarray, adjacent: np.ndarray, max_len: int) -> np.ndarray:
    """Mask of rows in runs of consecutive True longer than max_len.

    adjacent[i]: row i directly follows row i-1 (no missing minute); runs break otherwise.
    """
    cont = np.zeros(len(flag), bool)
    cont[1:] = flag[1:] & flag[:-1] & adjacent[1:]
    start = flag & ~cont
    run_id = np.cumsum(start) - 1
    lengths = np.bincount(run_id[flag], minlength=run_id.max() + 1 if len(run_id) else 0)
    out = np.zeros(len(flag), bool)
    out[flag] = lengths[run_id[flag]] > max_len
    return out


def clean_symbol(raw: pd.DataFrame, symbol: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Raw klines (from `read_klines`) -> (segment index, per-symbol report)."""
    c = cfg["clean"]
    m = minutes(raw["ts"])
    o, h, lo, cl, v = (raw[x].to_numpy() for x in ["open", "high", "low", "close", "volume"])
    n = len(raw)
    step = np.diff(m, prepend=m[0] - 1)                 # minutes since previous raw row
    adjacent = step == 1
    reason = np.zeros(n, np.int8)

    # Listing / relisting warm-up.
    anchor = np.where((np.arange(n) == 0) | (step >= c["relist_gap_days"] * 1440), m, np.iinfo(np.int64).min)
    anchor = np.maximum.accumulate(anchor)
    reason[m < anchor + c["listing_warmup_days"] * 1440] = LISTING

    prev_close = np.r_[np.nan, cl[:-1]]
    flat = (o == h) & (h == lo) & (lo == cl) & (cl == prev_close)
    identical = long_runs(flat, adjacent, c["max_identical_run"])
    reason[(reason == KEEP) & identical] = IDENTICAL
    zero = long_runs(v == 0, adjacent, c["max_zero_volume_run"])
    reason[(reason == KEEP) & zero] = ZERO_VOLUME

    # Jumps, measured against the previous raw row when it is within fill distance.
    near = step <= c["max_fill_gap_minutes"] + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        jump_size = np.maximum(np.abs(np.log(cl / prev_close)), np.abs(np.log(o / prev_close)))
    jump = near & (jump_size > c["max_abs_log_return"])
    jump[0] = False
    allow = {pd.Timestamp(t, tz="UTC") for s, t in c.get("jump_allowlist") or [] if s == symbol}
    if allow:
        jump &= ~raw["ts"].isin(allow).to_numpy()

    # Segments: maximal runs of kept rows without a break in between.
    kept = np.flatnonzero(reason == KEEP)
    brk = np.ones(len(kept), bool)
    if len(kept) > 1:
        k0, k1 = kept[:-1], kept[1:]
        long_gap = m[k1] - m[k0] - 1 > c["max_fill_gap_minutes"]
        brk[1:] = (k1 != k0 + 1) | long_gap | jump[k1]
    seg = np.cumsum(brk) - 1
    first = kept[brk]
    last = kept[np.r_[brk[1:], True]] if len(kept) else kept
    seg_df = pd.DataFrame({"symbol": symbol, "start_ts": from_minutes(m[first]),
                           "end_ts": from_minutes(m[last]),
                           "n_minutes": m[last] - m[first] + 1,
                           "n_raw_rows": np.bincount(seg, minlength=len(first))})
    seg_df["n_filled"] = seg_df["n_minutes"] - seg_df["n_raw_rows"]
    short = seg_df["n_minutes"].to_numpy() < c["min_segment_minutes"]
    reason[kept[short[seg]]] = SHORT
    seg_df = seg_df[~short].reset_index(drop=True)
    seg_df.insert(1, "segment_id", np.arange(len(seg_df)))

    # Report.
    hl = np.log(h / lo)
    rep = {"symbol": symbol, "raw_rows": n, "kept_rows": int((reason == KEEP).sum()),
           "kept_minutes": int(seg_df["n_minutes"].sum()), "filled": int(seg_df["n_filled"].sum()),
           "segments": len(seg_df),
           "jumps": int(jump.sum()), "long_gaps": int((step[1:] - 1 > c["max_fill_gap_minutes"]).sum()),
           "wicks_gt_thr": int((hl > c["max_abs_log_return"]).sum())}
    for r, name in REASONS.items():
        rep[f"dropped_{name}"] = int((reason == r).sum())
    rep["dropped_share"] = 1 - rep["kept_rows"] / n
    for split, (a, b) in split_bounds(cfg).items():
        s0 = seg_df["start_ts"].clip(lower=a)
        s1 = seg_df["end_ts"].clip(upper=b)
        rep[f"minutes_{split}"] = int(((s1 - s0) / pd.Timedelta(minutes=1) + 1).clip(lower=0).sum())
    return seg_df, rep


def load_clean(symbol: str, segments: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cleaned minute grid of one symbol: raw klines restricted to its segments, short
    gaps forward-filled, taker_buy_base clipped; adds `segment_id` and `filled`."""
    raw = read_klines(symbol) if raw is None else raw
    segs = segments[segments["symbol"] == symbol]
    m = minutes(raw["ts"])
    s0 = minutes(segs["start_ts"])
    lens = segs["n_minutes"].to_numpy()
    grid = np.repeat(s0 - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    seg_id = np.repeat(segs["segment_id"].to_numpy(), lens)
    idx = np.searchsorted(m, grid).clip(max=len(m) - 1)
    present = m[idx] == grid
    # Missing minutes take the last present row (segments always start with a present row).
    src = np.maximum.accumulate(np.where(present, idx, -1))
    assert (src >= 0).all(), f"{symbol}: segment starts on a missing minute"
    close = raw["close"].to_numpy()[src]
    out = {"ts": from_minutes(grid)}
    for col in ["open", "high", "low", "close"]:
        out[col] = np.where(present, raw[col].to_numpy()[idx], close)
    for col in ["volume", "n_trades", "taker_buy_base"]:
        out[col] = np.where(present, raw[col].to_numpy()[idx], 0)
    out["taker_buy_base"] = np.minimum(out["taker_buy_base"], out["volume"])
    out["segment_id"] = seg_id.astype(np.int32)
    out["filled"] = ~present
    df = pd.DataFrame(out)
    df["n_trades"] = df["n_trades"].astype(np.int32)
    return df


def check_clean(df: pd.DataFrame) -> dict:
    """Plan.md §3 acceptance checks on the cleaned grid."""
    o, h, lo, c = (df[x].to_numpy() for x in ["open", "high", "low", "close"])
    step = np.diff(minutes(df["ts"]))
    same_seg = np.diff(df["segment_id"].to_numpy()) == 0
    return {"check_nan": int(df.isna().sum().sum()),
            "check_not_1min_in_segment": int((step[same_seg] != 1).sum()),
            "check_ohlc_invalid": int(((h < np.maximum(o, c)) | (lo > np.minimum(o, c))).sum()),
            "check_taker_gt_volume": int((df["taker_buy_base"] > df["volume"]).sum())}


def _process(symbol: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = read_klines(symbol)
    segs, rep = clean_symbol(raw, symbol, cfg)
    rep |= check_clean(load_clean(symbol, segs, raw))
    rep["taker_clipped"] = int(((raw["taker_buy_base"] > raw["volume"])).sum())
    return segs, rep


def survivorship_note(symbols: list[str]) -> list[str]:
    u = yaml.safe_load(open(PROJECT_ROOT / "configs" / "universe.yaml"))
    inc = [s for s in u["symbols"] if s["status"] != "TRADING"]
    cand = pd.read_csv(data_root() / "raw" / "universe" / "candidates.csv")
    gone = cand[(cand["status"] != "TRADING") & cand["exclusion"].isna()]
    return [f"survivorship: {len(inc)} of {len(symbols)} included symbols are no longer trading "
            f"(delisted {sum(s['status'] == 'DELISTED' for s in inc)}, settling "
            f"{sum(s['status'] == 'SETTLING' for s in inc)}), e.g. "
            f"{', '.join(s['symbol'] for s in inc[:8])}",
            "survivorship: delisted/settling USDT crypto perps left out only by the history or "
            f"liquidity rules: {len(gone) - len(inc)} (see configs/universe.yaml `excluded`); "
            "no delisted contract is missing from the bucket listing"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--symbols", nargs="+")
    args = p.parse_args(argv)
    cfg = load_data_config()
    symbols = args.symbols or universe_symbols()
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(partial(_process, cfg=cfg), symbols, chunksize=2))
    segs = pd.concat([s for s, _ in results], ignore_index=True)[SEGMENT_COLUMNS]
    rep = pd.DataFrame([r for _, r in results])
    out = processed_dir()
    out.mkdir(parents=True, exist_ok=True)
    segs.to_parquet(out / "segments.parquet", index=False)
    rep.to_csv(out / "clean_report.csv", index=False)

    lines = [f"cleaned {len(symbols)} symbols in {time.time() - t0:.0f}s with config "
             f"{cfg['clean']}",
             f"raw rows {rep['raw_rows'].sum():,} -> kept rows {rep['kept_rows'].sum():,} "
             f"({rep['kept_rows'].sum() / rep['raw_rows'].sum():.2%}); kept minutes incl. fills "
             f"{rep['kept_minutes'].sum():,}; filled {rep['filled'].sum():,}; segments {len(segs):,}",
             "dropped rows: " + ", ".join(f"{n} {rep[f'dropped_{n}'].sum():,}" for n in REASONS.values()),
             f"segment breaks: jumps {rep['jumps'].sum():,}, long gaps {rep['long_gaps'].sum():,}; "
             f"candles with ln(H/L) > threshold (kept): {rep['wicks_gt_thr'].sum():,}",
             "minutes per split (before embargo): " + ", ".join(
                 f"{s} {rep[f'minutes_{s}'].sum() / 1e6:.1f}M" for s in ["train", "val", "test"]),
             f"symbols with train data: {(rep['minutes_train'] > 0).sum()} of {len(rep)}",
             "checks: " + ", ".join(f"{c} {rep[c].sum()}" for c in rep.columns if c.startswith("check_")),
             f"taker_buy_base clipped to volume: {rep['taker_clipped'].sum()} rows",
             "most dropped: " + ", ".join(f"{r.symbol} {r.dropped_share:.1%}" for r in
                                         rep.nlargest(8, "dropped_share").itertuples()),
             *survivorship_note(symbols)]
    (out / "clean_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    bad = sum(rep[c].sum() for c in rep.columns if c.startswith("check_"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
