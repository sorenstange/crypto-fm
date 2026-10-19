"""Causality and sanity tests for volatility and features (plan.md §4.5)."""

import numpy as np
import pandas as pd
import pytest

from cfm.data.clean import clean_symbol, load_clean
from cfm.data.features import FEATURES, build_features
from cfm.data.vol import MINUTES_PER_WEEK, candle_variance, ewma, minute_of_week
from test_clean import cfg as clean_cfg, make_raw

CFG = clean_cfg()
STATS = {"season": 1 + 0.5 * np.sin(np.arange(MINUTES_PER_WEEK) * 2 * np.pi / 1440),
         "scalers": {"lnsig": (-7.0, 1.0), "dvol": (5.0, 2.0)}}


def make_clean(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    raw = make_raw(n, seed=seed)
    raw = raw.drop(index=range(2000, 2008)).drop(index=range(4000, 4100)).reset_index(drop=True)
    segs, _ = clean_symbol(raw, "X", CFG)
    df = load_clean("X", segs, raw)
    assert df["segment_id"].nunique() == 2 and df["filled"].sum() == 8
    return df


def make_funding(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(df["ts"].iloc[0].floor("8h"), df["ts"].iloc[-1], freq="8h")
    return pd.DataFrame({"funding_time": t, "funding_interval_hours": np.full(len(t), 8, np.int16),
                         "funding_rate": rng.normal(0, 1e-4, len(t))})


def perturb_after(df: pd.DataFrame, fr: pd.DataFrame, t: int, seed: int = 1):
    """Replace every value after row t (and every funding settlement after its time)."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    k = np.arange(len(d)) > t
    n = k.sum()
    c = d.loc[k, "close"].to_numpy() * np.exp(rng.normal(0, 0.02, n))
    o = c * np.exp(rng.normal(0, 0.01, n))
    d.loc[k, "open"], d.loc[k, "close"] = o, c
    d.loc[k, "high"] = np.maximum(o, c) * np.exp(rng.uniform(0, 0.01, n))
    d.loc[k, "low"] = np.minimum(o, c) * np.exp(-rng.uniform(0, 0.01, n))
    d.loc[k, "volume"] = rng.uniform(0, 100, n)
    d.loc[k, "taker_buy_base"] = d.loc[k, "volume"] * rng.uniform(0, 1, n)
    d.loc[k, "n_trades"] = rng.integers(0, 500, n).astype(np.int32)
    f = fr.copy()
    later = (f["funding_time"] > d["ts"].iloc[t]).to_numpy()
    f.loc[later, "funding_rate"] = rng.normal(0, 1e-3, later.sum())
    f.loc[later, "funding_interval_hours"] = 4
    return d, f


@pytest.mark.parametrize("t", [0, 1, 1439, 1999, 2001, 2004, 3891, 3892, 5000])
def test_features_and_sigma_are_causal(t):
    df = make_clean()
    fr = make_funding(df)
    x1, a1 = build_features(df, fr, STATS, CFG)
    d2, f2 = perturb_after(df, fr, t)
    x2, a2 = build_features(d2, f2, STATS, CFG)
    np.testing.assert_array_equal(x1[:t + 1], x2[:t + 1])
    for key in ["v", "sigma", "sigma_prev", "scale"]:
        np.testing.assert_array_equal(a1["vol"][key][:t + 1], a2["vol"][key][:t + 1])
    assert not np.array_equal(x1[t + 1:], x2[t + 1:])       # the perturbation is not vacuous


def test_scale_uses_previous_sigma():
    df = make_clean()
    _, aux = build_features(df, None, STATS, CFG)
    vol = aux["vol"]
    same = np.r_[False, np.diff(df["segment_id"].to_numpy()) == 0]
    np.testing.assert_array_equal(vol["sigma_prev"][same], vol["sigma"][np.flatnonzero(same) - 1])
    np.testing.assert_allclose(vol["scale"], vol["sigma_prev"] * np.sqrt(vol["season"]))


def test_funding_only_after_settlement():
    df = make_clean()
    fr = make_funding(df)
    x, _ = build_features(df, fr, STATS, CFG)
    i_rate, i_left = FEATURES.index("funding"), FEATURES.index("to_funding")
    t_settle = fr["funding_time"].iloc[2]
    at = int(np.flatnonzero((df["ts"] == t_settle).to_numpy())[0])
    expect = lambda r: 5 * np.tanh(np.arcsinh(r * 1e4) / 5)  # noqa: E731
    assert x[at, i_rate] == pytest.approx(expect(fr["funding_rate"].iloc[2]), rel=1e-5)
    assert x[at - 1, i_rate] == pytest.approx(expect(fr["funding_rate"].iloc[1]), rel=1e-5)
    assert x[at, i_left] == pytest.approx(1.0) and x[at - 1, i_left] == pytest.approx(1 / 480, abs=1e-6)


def test_features_finite_and_bounded():
    df = make_clean()
    x, _ = build_features(df, make_funding(df), STATS, CFG)
    assert np.isfinite(x).all() and x.shape == (len(df), len(FEATURES))
    sq = CFG["features"]["squash"]
    assert (np.abs(x) <= sq + 1e-6).all()
    for k in ["upper", "lower"]:
        assert (x[:, FEATURES.index(k)] >= 0).all()
    for k in ["taker_imb", "range_pos"]:
        assert (np.abs(x[:, FEATURES.index(k)]) <= 1 + 1e-6).all()


def test_variance_estimators_nonnegative():
    raw = make_raw(seed=3)
    for est in ["garman_klass", "parkinson", "rogers_satchell"]:
        v = candle_variance(raw["open"], raw["high"], raw["low"], raw["close"], est)
        assert (np.asarray(v) >= 0).all()


def test_ewma_and_minute_of_week():
    x = np.r_[np.zeros(59), 1.0, np.zeros(60)]
    y = ewma(x, half_life=60, init=0.0)
    assert y[59] > 0 and y[119] == pytest.approx(y[59] / 2)       # half-life of 60 rows
    monday = pd.Timestamp("2024-01-01", tz="UTC")                  # a Monday
    assert minute_of_week(np.array([int(monday.timestamp() // 60)]))[0] == 0
