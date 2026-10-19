"""Checks on the real raw store (skipped when $DATA_ROOT has no data)."""

import numpy as np
import pandas as pd
import pytest

from cfm.data.clean import check_clean, load_clean, processed_dir
from cfm.data.features import build_features, load_stats, stats_dir
from cfm.data.load import kline_path, read_funding
from test_features import CFG as SYNTH_CFG, STATS as SYNTH_STATS, perturb_after

SYMBOLS = ["BTCUSDT", "LUNAUSDT", "LITUSDT", "HYPERUSDT"]
needs_data = pytest.mark.skipif(
    not (processed_dir() / "segments.parquet").exists() or not kline_path("BTCUSDT").exists(),
    reason="raw store / segment index not available")


@pytest.fixture(scope="module")
def segments():
    return pd.read_parquet(processed_dir() / "segments.parquet")


@needs_data
@pytest.mark.parametrize("symbol", SYMBOLS)
def test_candle_validity_after_cleaning(symbol, segments):
    df = load_clean(symbol, segments)
    assert len(df) > 0
    assert all(v == 0 for v in check_clean(df).values())


@needs_data
@pytest.mark.parametrize("symbol", ["BTCUSDT", "LUNAUSDT"])
def test_real_features_causal(symbol, segments):
    from cfm.data.config import load_data_config
    if (stats_dir() / "stats.json").exists():
        cfg, stats = load_data_config(), load_stats()
    else:
        cfg, stats = SYNTH_CFG, SYNTH_STATS
    df = load_clean(symbol, segments).iloc[:30_000].reset_index(drop=True)
    fr = read_funding(symbol)
    x1, a1 = build_features(df, fr, stats, cfg)
    assert np.isfinite(x1).all()
    for t in [100, 14_999, 29_000]:
        d2, f2 = perturb_after(df, fr, t)
        x2, a2 = build_features(d2, f2, stats, cfg)
        np.testing.assert_array_equal(x1[:t + 1], x2[:t + 1])
        np.testing.assert_array_equal(a1["vol"]["scale"][:t + 1], a2["vol"]["scale"][:t + 1])
