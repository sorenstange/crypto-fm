"""Raw kline / funding frames -> compact per-symbol parquet (plan.md §2.4, §15.2).

Klines are stored losslessly as integers to fit the home quota (~15 B/candle instead of
~45 B for float64 parquet):
- prices (open/high/low/close) as int64 ticks: round(price * 10**price_decimals)
- volumes (volume, taker_buy_base) as int64 lots: round(qty * 10**qty_decimals)
- n_trades as int32, ts as timestamp[ms, UTC]
all DELTA_BINARY_PACKED + zstd. The decimals are chosen per symbol as the smallest that
reproduce every float64 value exactly, and are stored in the parquet key-value metadata.
`quote_volume` and `taker_buy_quote` are dropped (plan.md §15.2). Decode with
`cfm.data.load.read_klines`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT_VERSION = "1"
PRICE_COLUMNS = ["open", "high", "low", "close"]
QTY_COLUMNS = ["volume", "taker_buy_base"]
KLINE_STORE_COLUMNS = ["ts", *PRICE_COLUMNS, "volume", "n_trades", "taker_buy_base"]
MAX_DECIMALS = 12
MINUTE_MS = 60_000


def find_decimals(values: np.ndarray, max_decimals: int = MAX_DECIMALS) -> int:
    """Smallest d such that rint(x * 10**d) / 10**d reproduces every float64 x exactly."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    for d in range(max_decimals + 1):
        scale = 10.0 ** d
        q = np.rint(x * scale)
        if np.abs(q).max(initial=0) >= 2**53:
            break
        if np.array_equal(q / scale, x):
            return d
    raise ValueError(f"no exact decimal scale <= {max_decimals} (sample: {x[:5]})")


def _encode(x: np.ndarray, d: int) -> np.ndarray:
    q = np.rint(x * 10.0 ** d)
    assert np.array_equal(q / 10.0 ** d, x), "lossy encoding"
    return q.astype(np.int64)


def encode_klines(df: pd.DataFrame) -> pa.Table:
    """Kline frame from `binance.read_klines_csv` (sorted, deduplicated) -> arrow table."""
    pdec = find_decimals(df[PRICE_COLUMNS].to_numpy().ravel())
    qdec = find_decimals(df[QTY_COLUMNS].to_numpy().ravel())
    cols = {"ts": pa.array(df["open_time"].to_numpy(), pa.timestamp("ms", tz="UTC"))}
    for c in PRICE_COLUMNS:
        cols[c] = pa.array(_encode(df[c].to_numpy(), pdec))
    cols["volume"] = pa.array(_encode(df["volume"].to_numpy(), qdec))
    cols["n_trades"] = pa.array(df["n_trades"].to_numpy().astype(np.int32))
    cols["taker_buy_base"] = pa.array(_encode(df["taker_buy_base"].to_numpy(), qdec))
    meta = {"cfm_format": FORMAT_VERSION, "price_decimals": str(pdec), "qty_decimals": str(qdec)}
    return pa.table(cols).replace_schema_metadata(meta)


def write_parquet_atomic(table: pa.Table, path: Path, delta: bool = True) -> int:
    """Write via a temp file + rename, so a killed job never leaves a half file. Returns bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kw = {}
    if delta:
        ints = [f.name for f in table.schema
                if pa.types.is_integer(f.type) or pa.types.is_timestamp(f.type)]
        kw = {"use_dictionary": False, "column_encoding": {c: "DELTA_BINARY_PACKED" for c in ints}}
    pq.write_table(table, tmp, compression="zstd", compression_level=9,
                   row_group_size=1 << 20, **kw)
    os.replace(tmp, path)
    return path.stat().st_size


def clean_klines(df: pd.DataFrame, min_stale_tail: int = 1440) -> tuple[pd.DataFrame, dict]:
    """Sort + deduplicate on open_time, trim the stale tail; report what was found.

    Settled/delisted contracts keep producing flat zero-volume candles after trading
    ends. A trailing run of >= `min_stale_tail` zero-volume candles is dropped; all other
    cleaning happens in Phase 1.
    """
    stats = {"rows_in": len(df)}
    df = df.sort_values("open_time", kind="stable")
    dup = df.duplicated("open_time", keep="last")
    if dup.any():
        # Identical duplicates are harmless; conflicting ones are counted for the report.
        conflicting = df[df.duplicated("open_time", keep=False)].drop_duplicates().duplicated(
            "open_time").sum()
        stats["duplicate_conflicting"] = int(conflicting)
    stats["duplicates"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    traded = np.flatnonzero(df["volume"].to_numpy() > 0)
    tail = len(df) - (traded[-1] + 1 if len(traded) else 0)
    stats["stale_tail_trimmed"] = tail if tail >= min_stale_tail else 0
    if stats["stale_tail_trimmed"]:
        df = df.iloc[:len(df) - tail]
    stats["off_grid"] = int((df["open_time"] % MINUTE_MS != 0).sum())
    stats["bad_close_time"] = int((df["close_time"] != df["open_time"] + MINUTE_MS - 1).sum())
    stats["rows"] = len(df)
    return df, stats


def encode_funding(df: pd.DataFrame) -> pa.Table:
    """fundingRate frame -> funding_time (calc_time rounded to the minute), interval, rate."""
    t = df["calc_time"].to_numpy()
    t = (t + MINUTE_MS // 2) // MINUTE_MS * MINUTE_MS     # ~0-20 ms jitter -> exact minute
    out = (pd.DataFrame({"funding_time": t,
                         "funding_interval_hours": df["funding_interval_hours"].astype(np.int16),
                         "funding_rate": df["last_funding_rate"].astype(np.float64)})
           .drop_duplicates("funding_time", keep="last").sort_values("funding_time"))
    return pa.table({"funding_time": pa.array(out["funding_time"].to_numpy(), pa.timestamp("ms", tz="UTC")),
                     "funding_interval_hours": pa.array(out["funding_interval_hours"].to_numpy()),
                     "funding_rate": pa.array(out["funding_rate"].to_numpy())},
                    metadata={"cfm_format": FORMAT_VERSION})
