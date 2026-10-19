"""Read the raw per-symbol parquet written by `cfm.data.download` (plan.md §2.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cfm.data.binance import data_root
from cfm.data.convert import PRICE_COLUMNS, QTY_COLUMNS


def raw_dir() -> Path:
    return data_root() / "raw"


def kline_path(symbol: str, root: Path | None = None) -> Path:
    return (root or raw_dir()) / "parquet" / f"{symbol}.parquet"


def funding_path(symbol: str, root: Path | None = None) -> Path:
    return (root or raw_dir()) / "funding" / f"{symbol}.parquet"


def read_klines(symbol: str, columns: list[str] | None = None,
                root: Path | None = None) -> pd.DataFrame:
    """Klines with ts (datetime64[ms, UTC]), float64 prices/volumes, int32 n_trades."""
    path = kline_path(symbol, root)
    table = pq.read_table(path, columns=columns)
    meta = {k.decode(): v.decode() for k, v in (pq.read_schema(path).metadata or {}).items()}
    pscale = 10.0 ** int(meta["price_decimals"])
    qscale = 10.0 ** int(meta["qty_decimals"])
    df = table.to_pandas()
    for c in df.columns:
        if c in PRICE_COLUMNS:
            df[c] = df[c].to_numpy().astype(np.float64) / pscale
        elif c in QTY_COLUMNS:
            df[c] = df[c].to_numpy().astype(np.float64) / qscale
    return df


def read_funding(symbol: str, root: Path | None = None) -> pd.DataFrame:
    """funding_time (datetime64[ms, UTC], exact minute), funding_interval_hours, funding_rate."""
    return pq.read_table(funding_path(symbol, root)).to_pandas()
