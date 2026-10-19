"""Targets, splits, metrics and baselines."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cfm.data.clean import minutes
from cfm.data.features import build_features
from cfm.data.splits import prediction_points, split_codes
from cfm.data.targets import bin_edges, make_targets, n_classes, to_bins
from cfm.eval import baselines as bl
from cfm.eval import metrics as mt
from test_features import CFG, STATS, make_clean, perturb_after


def _targets(df):
    _, aux = build_features(df, None, STATS, CFG)
    return aux, make_targets(df["close"].to_numpy(), aux["vol"], df["segment_id"].to_numpy(), CFG)


def test_targets_alignment_and_segments():
    df = make_clean()
    aux, tg = _targets(df)
    seg = df["segment_id"].to_numpy()
    c = df["close"].to_numpy()
    for h in CFG["targets"]["return_horizons"]:
        z = tg["z"][h]
        t = 100
        expect = np.log(c[t + h] / c[t]) / (aux["vol"]["sigma"][t] *
                                             np.sqrt(aux["vol"]["season"][t + 1:t + h + 1].sum()))
        assert z[t] == pytest.approx(expect)
        # NaN exactly where t + h leaves the segment.
        same = np.r_[seg[h:] == seg[:-h], np.zeros(h, bool)]
        assert (np.isfinite(z) == same).all()
    rv = tg["rv"][15]
    vd = aux["vol"]["v"] / aux["vol"]["season"]
    assert rv[200] == pytest.approx(vd[201:216].sum())


def test_target_scale_is_causal():
    """The scale of the target made at t must not depend on data after t."""
    df = make_clean()
    _, tg1 = _targets(df)
    t = 3000
    d2, _ = perturb_after(df, pd.DataFrame({"funding_time": pd.DatetimeIndex([], tz="UTC"),
                                            "funding_interval_hours": [], "funding_rate": []}), t)
    _, tg2 = _targets(d2)
    for h in CFG["targets"]["return_horizons"]:
        np.testing.assert_array_equal(tg1["ret_scale"][h][:t + 1], tg2["ret_scale"][h][:t + 1])


def test_bins():
    e = bin_edges(CFG)
    assert len(e) == 65 and n_classes(CFG) == 66
    b = to_bins(np.array([-9.0, -5.0, -4.99, 0.0, 4.99, 5.0, 9.0, np.nan]), e)
    assert list(b) == [0, 1, 1, 33, 64, 65, 65, -1]


def test_prediction_points_and_splits():
    df = make_clean()
    m = minutes(df["ts"])
    p = prediction_points(m, df["segment_id"].to_numpy(), CFG)
    assert ((m[p] + 1) % 15 == 0).all()
    first = df.groupby("segment_id")["ts"].transform("min")
    assert ((df["ts"][p] - first[p]) >= pd.Timedelta(minutes=1440)).all()
    ts = pd.Series(pd.to_datetime(["2024-12-31 23:59", "2025-01-02 00:00", "2025-01-04 00:00",
                                   "2025-10-02 00:00", "2025-10-05 00:00", "2026-10-01 00:00"], utc=True))
    assert list(split_codes(ts, CFG)) == [0, -1, 1, -1, 2, -1]


def test_qlike_minimised_by_truth():
    rng = np.random.default_rng(0)
    true_var = np.exp(rng.normal(0, 1, 100_000))
    rv = true_var * rng.chisquare(5, 100_000) / 5
    losses = {c: mt.qlike(rv, c * true_var).mean() for c in [0.7, 1.0, 1.4]}
    assert losses[1.0] < losses[0.7] and losses[1.0] < losses[1.4]


def test_ce_and_crps_gaussian():
    e = bin_edges(CFG)
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 50_000)
    cls = to_bins(z, e).astype(int)
    p = mt.cdf_to_probs(bl.gauss_cdf(e)[None, :])[0]
    assert p.sum() == pytest.approx(1.0)
    ce = mt.cross_entropy(p[cls]).mean()
    # CE of the binned N(0,1) on N(0,1) data = entropy of the binned distribution.
    assert ce == pytest.approx(-(p * np.log(p)).sum(), abs=0.02)
    crps = mt.crps_binned(np.broadcast_to(bl.gauss_cdf(e), (len(z), len(e))), z, e).mean()
    assert crps == pytest.approx(1 / np.sqrt(np.pi), rel=0.01)    # E CRPS of N(0,1) on N(0,1)
    u = mt.randomized_pit(bl.gauss_cdf(e), cls, rng)
    assert mt.pit_deviation(u) < 0.01
    assert mt.pit_deviation(mt.randomized_pit(bl.gauss_cdf(e, 2.0), cls, rng)) > 0.03


def test_marginal_and_bootstrap():
    p = bl.fit_marginal(np.array([0, 0, 1, -1]), 3)
    np.testing.assert_allclose(p, [3 / 6, 2 / 6, 1 / 6])
    day = np.repeat(np.arange(50), 100)
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, len(day))
    t = mt.day_bootstrap(day, {"a": a, "b": a - 0.5}, baseline="a")
    row = t.set_index("model").loc["b"]
    assert row["gain"] == pytest.approx(0.5) and row["gain_ci_lo"] == pytest.approx(0.5)
    assert t.set_index("model").loc["a", "ci_lo"] < a.mean() < t.set_index("model").loc["a", "ci_hi"]


def test_garch_forecast_matches_recursion():
    g = {"omega": 0.05, "alpha": 0.1, "beta": 0.85, "var0": 1.3}
    rng = np.random.default_rng(3)
    r = rng.normal(0, 1, 300)
    r[0] = np.nan
    f = bl.garch_forecast(r, [(0, 300)], g, [1, 5])
    s = 1.3
    for t in range(300):
        r2 = 1.3 if np.isnan(r[t]) else r[t] ** 2
        s = 0.05 + 0.1 * r2 + 0.85 * s            # sigma2_{t+1|t}
        k = [s]
        for _ in range(4):
            k.append(0.05 + 0.95 * k[-1])
        assert f[1][t] * 1e8 == pytest.approx(k[0])
        assert f[5][t] * 1e8 == pytest.approx(sum(k))


def test_garch_forecast_integrated_is_finite():
    g = {"omega": 0.2, "alpha": 0.08, "beta": 0.92, "var0": 5.0}       # alpha + beta = 1
    r = np.random.default_rng(5).normal(0, 2, 500)
    f = bl.garch_forecast(r, [(0, 500)], g, [15, 60])
    assert np.isfinite(f[60]).all() and (f[60] > f[15]).all()


def test_har_recovers_coefficients():
    rng = np.random.default_rng(4)
    trailing = np.exp(rng.normal(-14, 1, (20_000, 3)))
    beta = np.array([-1.0, 0.4, 0.3, 0.2])
    mean = np.exp(bl.har_design(trailing) @ beta)
    rv = 15 * mean * rng.lognormal(0, 0.3, len(mean))
    model = bl.fit_har(trailing, rv, 15)
    np.testing.assert_allclose(model["beta"][1:], beta[1:], atol=0.02)
    assert np.mean(rv / bl.predict_har(model, trailing, 15)) == pytest.approx(1.0, abs=1e-6)
