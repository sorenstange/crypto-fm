"""Lossless kline encoding round trip, dedup and funding-time rounding."""

import numpy as np
import pandas as pd
import pytest

from cfm.data import binance as bn
from cfm.data.convert import (clean_klines, encode_funding, encode_klines, find_decimals,
                              write_parquet_atomic)
from cfm.data.load import kline_path, read_klines

CSV = b"""open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1577836860000,0.0012345,0.0012400,0.0012300,0.0012390,1500000,1577836919999,1.0,12,700000,1.0,0
1577836800000,0.0012340,0.0012350,0.0012330,0.0012345,2500000,1577836859999,1.0,10,1000000,1.0,0
1577836800000,0.0012340,0.0012350,0.0012330,0.0012345,2500000,1577836859999,1.0,10,1000000,1.0,0
1577836920000,0.0012390,0.0012390,0.0012390,0.0012390,0,1577836979999,0.0,0,0,0.0,0
"""


def test_find_decimals():
    assert find_decimals(np.array([7182.44, 7190.52, 7177.0])) == 2
    assert find_decimals(np.array([0.0012345, 0.00124])) == 7
    assert find_decimals(np.array([100.0, 2500000.0])) == 0
    with pytest.raises(ValueError):
        find_decimals(np.array([1 / 3]))


def test_clean_and_roundtrip(tmp_path):
    raw = bn.read_klines_csv(CSV)
    df, stats = clean_klines(raw)
    assert stats["duplicates"] == 1 and stats.get("duplicate_conflicting", 0) == 0
    assert stats["rows"] == 3 and stats["off_grid"] == 0 and stats["bad_close_time"] == 0
    assert df["open_time"].is_monotonic_increasing

    write_parquet_atomic(encode_klines(df), kline_path("X", tmp_path))
    back = read_klines("X", root=tmp_path)
    assert list(back.columns) == ["ts", "open", "high", "low", "close", "volume", "n_trades",
                                  "taker_buy_base"]
    assert str(back["ts"].dt.tz) == "UTC"
    assert list(back["ts"].map(lambda t: t.value // 10**6)) == list(df["open_time"])
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        assert np.array_equal(back[c].to_numpy(), df[c].to_numpy()), c   # bit-exact
    assert (back["n_trades"].to_numpy() == df["n_trades"].to_numpy()).all()


def test_stale_tail_trimmed():
    df = bn.read_klines_csv(CSV)
    _, stats = clean_klines(df)                      # 1 trailing zero-volume candle: kept
    assert stats["stale_tail_trimmed"] == 0 and stats["rows"] == 3
    out, stats = clean_klines(df, min_stale_tail=1)
    assert stats["stale_tail_trimmed"] == 1 and len(out) == 2 and out["volume"].iloc[-1] > 0


def test_conflicting_duplicates_counted():
    bad = CSV.replace(b"1577836800000,0.0012340,0.0012350,0.0012330,0.0012345,2500000,1577836859999,1.0,10,",
                      b"1577836800000,0.0012340,0.0012350,0.0012330,0.0012345,2500001,1577836859999,1.0,10,", 1)
    _, stats = clean_klines(bn.read_klines_csv(bad))
    assert stats["duplicates"] == 1 and stats["duplicate_conflicting"] == 1


def test_funding_time_rounded_to_minute():
    fr = pd.DataFrame({"calc_time": [1788220800005, 1788249599990],
                       "funding_interval_hours": [8, 8], "last_funding_rate": [1e-4, -2e-4]})
    t = encode_funding(fr).to_pandas()["funding_time"]
    ms = t.map(lambda x: x.value // 10**6)
    assert list(ms) == [1788220800000, 1788249600000]
