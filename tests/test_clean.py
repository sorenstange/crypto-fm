"""Phase 1 cleaning rules on synthetic klines."""

import copy

import numpy as np
import pandas as pd
import pytest

from cfm.data.clean import check_clean, clean_symbol, load_clean
from cfm.data.config import load_data_config

BASE = load_data_config()


def make_raw(n: int = 2000, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    """Random-walk 1m klines with valid OHLC and positive volume."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 1e-3, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 1e-3, n))
    vol = rng.uniform(1, 10, n)
    return pd.DataFrame({"ts": pd.date_range(start, periods=n, freq="min", tz="UTC"),
                         "open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "n_trades": rng.integers(1, 50, n).astype(np.int32),
                         "taker_buy_base": vol * rng.uniform(0, 1, n)})


def cfg(**clean):
    c = copy.deepcopy(BASE)
    c["clean"] |= {"listing_warmup_days": 0, "min_segment_minutes": 100} | clean
    return c


def run(raw, **clean):
    segs, rep = clean_symbol(raw, "X", cfg(**clean))
    return segs, rep, load_clean("X", segs, raw)


def test_clean_data_is_one_segment():
    segs, rep, df = run(make_raw())
    assert len(segs) == 1 and rep["kept_rows"] == 2000 and rep["filled"] == 0
    assert all(v == 0 for k, v in check_clean(df).items())


def test_short_gap_filled_long_gap_splits():
    raw = make_raw()
    raw = raw.drop(index=range(500, 510)).drop(index=range(1200, 1220)).reset_index(drop=True)
    segs, rep, df = run(raw)
    assert len(segs) == 2 and list(segs["n_filled"]) == [10, 0]
    f = df[df["filled"]]
    assert len(f) == 10 and (f["volume"] == 0).all() and (f["taker_buy_base"] == 0).all()
    prev_close = raw.loc[499, "close"]
    assert (f[["open", "high", "low", "close"]] == prev_close).all().all()
    assert all(v == 0 for v in check_clean(df).values())


def test_jump_splits():
    raw = make_raw()
    raw.loc[1000:, ["open", "high", "low", "close"]] *= 1.2          # +18% log jump at 1000
    raw.loc[1000, "open"] = raw.loc[999, "close"]                   # jump inside the candle
    raw.loc[1000, "low"] = min(raw.loc[1000, "low"], raw.loc[1000, "open"])
    segs, rep, _ = run(raw)
    assert rep["jumps"] == 1 and len(segs) == 2
    assert segs.loc[1, "start_ts"] == raw.loc[1000, "ts"]
    segs, rep, _ = run(raw, jump_allowlist=[["X", str(raw.loc[1000, "ts"].tz_localize(None))]])
    assert rep["jumps"] == 0 and len(segs) == 1


@pytest.mark.parametrize("length,dropped", [(15, 0), (16, 16)])
def test_identical_run(length, dropped):
    raw = make_raw()
    p = raw.loc[999, "close"]
    raw.loc[1000:999 + length, ["open", "high", "low", "close"]] = p
    raw.loc[1000 + length, "open"] = p
    raw.loc[1000 + length, ["high", "low"]] = [max(raw.loc[1000 + length, ["high", "close"]].max(), p),
                                               min(raw.loc[1000 + length, ["low", "close"]].min(), p)]
    segs, rep, _ = run(raw)
    assert rep["dropped_identical_run"] == dropped and len(segs) == (2 if dropped else 1)


@pytest.mark.parametrize("length,dropped", [(45, 0), (46, 46)])
def test_zero_volume_run(length, dropped):
    raw = make_raw()
    raw.loc[700:699 + length, ["volume", "taker_buy_base"]] = 0
    segs, rep, _ = run(raw)
    assert rep["dropped_zero_volume_run"] == dropped and len(segs) == (2 if dropped else 1)


def test_listing_and_relist_warmup_and_short_segments():
    raw = make_raw(n=6 * 1440)
    raw = raw.drop(index=range(3 * 1440, 3 * 1440 + 10)).reset_index(drop=True)
    relist = make_raw(n=5 * 1440, seed=1, start="2024-01-20")    # after a > 7-day gap
    raw = pd.concat([raw, relist], ignore_index=True)
    segs, rep, _ = run(raw, listing_warmup_days=1, min_segment_minutes=1440)
    # Listing day and the first relist day dropped; the 10-minute gap is filled.
    assert rep["dropped_listing_warmup"] == 2 * 1440
    assert list(segs["start_ts"]) == [pd.Timestamp("2024-01-02", tz="UTC"),
                                      pd.Timestamp("2024-01-21", tz="UTC")]
    segs, rep, _ = run(raw, listing_warmup_days=1, min_segment_minutes=5 * 1440)
    assert len(segs) == 1 and rep["dropped_short_segment"] == 4 * 1440


def test_taker_clipped():
    raw = make_raw()
    raw.loc[5, "taker_buy_base"] = raw.loc[5, "volume"] + 1
    _, _, df = run(raw)
    assert (df["taker_buy_base"] <= df["volume"]).all()


def test_candle_validity_after_cleaning_random():
    for seed in range(5):
        raw = make_raw(seed=seed)
        drop = np.random.default_rng(seed).choice(np.arange(1, 2000), 60, replace=False)
        _, _, df = run(raw.drop(index=drop).reset_index(drop=True))
        assert all(v == 0 for v in check_clean(df).values())
