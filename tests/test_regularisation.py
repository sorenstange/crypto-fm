"""Regularisation sweep infrastructure: config overrides, HL-Gauss targets, feature dropping,
sweep run selection, paired run comparison."""

import copy

import numpy as np
import pytest
import torch
import yaml

from cfm.data.config import PROJECT_ROOT, load_data_config
from cfm.data.features import FEATURES
from cfm.data.targets import bin_edges
from cfm.eval.compare_runs import compare_runs
from cfm.losses import bins_of, return_losses, soft_targets
from cfm.model.backbone import build_backbone
from cfm.sweep import select_runs
from cfm.train import apply_overrides, parse_sets

DCFG = load_data_config()
BCFG = yaml.safe_load(open(PROJECT_ROOT / "configs" / "backbone.yaml"))
EDGES = torch.as_tensor(bin_edges(DCFG), dtype=torch.float32)


def test_overrides_parse_set_and_reject_unknown_keys():
    o = parse_sets(["model.dropout=0.2", "loss.ret_weights=[1, 1, 0.5, 0.25]",
                    "model.drop_features=[mod_sin]"])
    cfg = apply_overrides(BCFG, o)
    assert cfg["model"]["dropout"] == 0.2 and cfg["loss"]["ret_weights"] == [1, 1, 0.5, 0.25]
    assert cfg["model"]["drop_features"] == ["mod_sin"]
    assert BCFG["model"]["dropout"] == 0.1                      # original untouched
    with pytest.raises(KeyError):
        apply_overrides(BCFG, {"model.dropuot": 0.2})
    with pytest.raises(KeyError):
        apply_overrides(BCFG, {"nosection.x": 1})
    with pytest.raises(ValueError):
        parse_sets(["model.dropout"])


def test_sweep_files_are_valid():
    for f in (PROJECT_ROOT / "configs" / "sweeps").glob("*.yaml"):
        sw = yaml.safe_load(open(f))
        base = apply_overrides(BCFG, sw.get("base") or {})
        for o in sw["runs"].values():
            apply_overrides(base, o or {})


def test_select_runs_partitions():
    names = [f"r{i}" for i in range(9)]
    parts = [select_runs(names, k, 2) for k in (1, 2)]
    assert sorted(parts[0] + parts[1]) == sorted(names) and not set(parts[0]) & set(parts[1])


def test_soft_targets():
    z = torch.tensor([-7.0, -0.3, 0.0, 0.01, 2.2, 9.0])
    p = soft_targets(z, EDGES, 0.117)
    assert torch.allclose(p.sum(-1), torch.ones(len(z)), atol=1e-6) and (p >= 0).all()
    hard = soft_targets(z, EDGES, 1e-4).argmax(-1)               # sigma -> 0: one-hot of the bin
    assert torch.equal(hard, bins_of(z, EDGES))


def test_soft_loss_changes_objective_not_reported_ce():
    torch.manual_seed(0)
    logits, z = torch.randn(50, 4, 66), torch.randn(50, 4)
    z[3, 1] = torch.nan
    m = torch.ones(50, dtype=torch.bool)
    h0, zl0, ce0 = return_losses(logits, z, EDGES, m)
    h1, zl1, ce1 = return_losses(logits, z, EDGES, m, soft_sigma=0.117)
    assert torch.allclose(h0, torch.nanmean(ce0, 0), atol=1e-6)
    assert torch.equal(torch.isnan(ce0), torch.isnan(ce1)) and torch.allclose(ce0.nan_to_num(), ce1.nan_to_num())
    assert not torch.allclose(h0, h1) and torch.allclose(zl0, zl1)


def test_drop_features_are_ignored():
    b = copy.deepcopy(BCFG)
    b["model"] |= {"d_model": 64, "n_layers": 2, "n_heads": 2, "mlp_hidden": 128, "dropout": 0.0,
                   "drop_features": ["dow_sin", "dow_cos"]}
    torch.manual_seed(0)
    m = build_backbone(b, DCFG, len(FEATURES)).eval()
    x = torch.randn(2, 8, 15, len(FEATURES))
    y = m(x.reshape(2, 8, -1))[0]
    x2 = x.clone()
    x2[..., FEATURES.index("dow_sin")] += 5
    assert torch.allclose(m(x2.reshape(2, 8, -1))[0], y)
    x2[..., FEATURES.index("ret")] += 5
    assert not torch.allclose(m(x2.reshape(2, 8, -1))[0], y)
    assert "in_mask" not in m.state_dict()                      # checkpoints stay compatible


def test_compare_runs_paired_gain():
    rng = np.random.default_rng(0)
    n = 20_000
    day = np.repeat(np.arange(100), n // 100)
    base = rng.exponential(3.0, (n, 4))
    ref = {"ce": base, "qlike": rng.exponential(0.3, (n, 2))}
    better = {"ce": base - 0.01, "qlike": ref["qlike"].copy()}
    t = compare_runs({"ref": ref, "better": better}, day, "ref", [1, 5, 15, 60], [15, 60])
    row = t[(t.metric == "ce_h1") & (t.run == "better")].iloc[0]
    assert abs(row["gain"] - 0.01) < 1e-9 and row["gain_ci_lo"] > 0.0099      # paired: tight CI
    total = t[(t.metric == "val_loss") & (t.run == "better")].iloc[0]
    assert abs(total["gain"] - 0.04) < 1e-9


def test_flip_matches_features_of_a_mirrored_price_path():
    """flip_windows on the features of a path == the features of the mirrored path
    (prices K/p, high <-> low, taker buys <-> sells, funding negated); range_pos only
    approximately (range in price space), dvol not compared (kept unchanged by design)."""
    from test_features import CFG, STATS, make_clean, make_funding
    from cfm.data.augment import flip_tables, flip_windows
    from cfm.data.features import build_features

    df, n = make_clean(), 3000
    df = df[df["segment_id"] == df["segment_id"].iloc[0]].iloc[:n].reset_index(drop=True)
    fr = make_funding(df)
    k = float(df["close"].iloc[0]) ** 2
    m = df.copy()
    m["open"], m["close"] = k / df["open"], k / df["close"]
    m["high"], m["low"] = k / df["low"], k / df["high"]
    m["taker_buy_base"] = df["volume"] - df["taker_buy_base"]
    fm = fr.assign(funding_rate=-fr["funding_rate"])
    x, _ = build_features(df, fr, STATS, CFG)
    xm, _ = build_features(m, fm, STATS, CFG)
    n15 = len(x) // 15 * 15
    xt = torch.as_tensor(x[:n15].reshape(1, -1, 15 * len(FEATURES)), dtype=torch.float32)
    perm, sign = flip_tables(torch.device("cpu"), torch.float32)
    xf, zf = flip_windows(xt, torch.ones(1, xt.shape[1], 4), torch.tensor([True]), perm, sign)
    xf = xf.reshape(-1, len(FEATURES)).numpy()
    xm = xm[:n15]
    approx = {"range_pos": 0.05}
    for j, f in enumerate(FEATURES):
        if f == "dvol":        # dollar volume moves with the mirrored price level; the flip keeps it by design
            continue
        np.testing.assert_allclose(xf[:, j], xm[:, j], atol=approx.get(f, 1e-4), err_msg=f)
    assert (zf == -1).all()
    xb, _ = flip_windows(torch.as_tensor(xf.reshape(xt.shape)), zf, torch.tensor([True]), perm, sign)
    assert torch.allclose(xb, xt)                                  # flipping twice is the identity
