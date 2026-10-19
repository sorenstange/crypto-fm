"""Backbone, losses and dataset windows."""

import copy
import math

import numpy as np
import pytest
import torch
import yaml

from cfm.data.config import PROJECT_ROOT, load_data_config
from cfm.data.dataset import PatchData, WindowSampler, eval_windows, start_weights, train_starts
from cfm.data.targets import bin_edges, to_bins
from cfm.eval import metrics as mt
from cfm.losses import bins_of, qlike_losses, return_losses
from cfm.model.backbone import build_backbone

DCFG = load_data_config()
BCFG = yaml.safe_load(open(PROJECT_ROOT / "configs" / "backbone.yaml"))
F = 17


def small_model(**over):
    b = copy.deepcopy(BCFG)
    b["model"] |= {"d_model": 64, "n_layers": 2, "n_heads": 2, "mlp_hidden": 128} | over
    return build_backbone(b, DCFG, F)


def test_param_count_of_v1_config():
    n = sum(p.numel() for p in build_backbone(BCFG, DCFG, F).parameters())
    assert 4.5e6 < n < 6e6


def test_model_is_causal():
    torch.manual_seed(0)
    m = small_model(dropout=0.0).eval()
    x = torch.randn(2, 40, 15 * F)
    y1, u1 = m(x)
    x2 = x.clone()
    x2[:, 25:] = torch.randn(2, 15, 15 * F)
    y2, u2 = m(x2)
    torch.testing.assert_close(y1[:, :25], y2[:, :25])
    torch.testing.assert_close(u1[:, :25], u2[:, :25])
    assert not torch.allclose(y1[:, 25:], y2[:, 25:])


def test_initial_losses():
    torch.manual_seed(0)
    m = build_backbone(BCFG, DCFG, F).eval()
    logits, u = m(torch.randn(4, 128, 15 * F).clamp(-5, 5))
    edges = torch.as_tensor(bin_edges(DCFG), dtype=torch.float32)
    z = torch.randn(4 * 128, 4)
    ce_h, zl, _ = return_losses(logits.reshape(-1, 4, 66), z, edges, torch.ones(4 * 128, dtype=torch.bool))
    assert torch.allclose(ce_h, torch.full_like(ce_h, math.log(66)), atol=0.05)
    assert torch.all(u == 0)                                  # vol head starts at EWMA


def test_losses_match_reference():
    edges_np = bin_edges(DCFG)
    edges = torch.as_tensor(edges_np, dtype=torch.float32)
    z = torch.tensor([[-7.0, 0.1], [4.99, float("nan")], [5.0, -5.0]])
    np.testing.assert_array_equal(bins_of(torch.nan_to_num(z), edges)[[0, 2]].numpy(),
                                  to_bins(z.numpy(), edges_np)[[0, 2]])
    logits = torch.randn(3, 2, 66)
    ce_h, _, ce = return_losses(logits, z, edges, torch.ones(3, dtype=torch.bool))
    ref = torch.nn.functional.cross_entropy(logits[0, 0][None], bins_of(z[0, :1], edges))
    assert ce[0, 0] == pytest.approx(ref.item(), rel=1e-5)
    assert torch.isnan(ce[1, 1]) and ce_h[1] == pytest.approx(ce[[0, 2], 1].mean().item(), rel=1e-5)

    rv = torch.tensor([[2e-6, 3e-6], [0.0, float("nan")]])
    s2, u = torch.tensor([1e-7, 2e-7]), torch.tensor([[0.3, -0.2], [0.1, 0.0]])
    _, ql = qlike_losses(u, rv, s2, [15, 60], torch.ones(2, dtype=torch.bool))
    ref = mt.qlike(rv[0].double().numpy(), (np.array([15, 60]) * 1e-7 * np.exp(u[0].double().numpy())))
    np.testing.assert_allclose(ql[0].numpy(), ref, rtol=1e-4)
    assert torch.isfinite(ql[1, 0]) and torch.isnan(ql[1, 1])


def fake_patches(seg_lens, split_fn, n_sym=2):
    """PatchData with consecutive segments; split_fn(local, seg) -> split code."""
    local = np.concatenate([np.arange(n) for n in seg_lens]).astype(np.int32)
    seg_len = np.concatenate([np.full(n, n) for n in seg_lens]).astype(np.int32)
    seg = np.concatenate([np.full(n, i) for i, n in enumerate(seg_lens)])
    N = len(local)
    return PatchData(feats=np.zeros((N, 15, F), np.float16), z=np.zeros((N, 4), np.float32),
                     rv=np.zeros((N, 2), np.float32), sigma2=np.ones(N, np.float32),
                     har=np.ones((N, 3), np.float32), minute=np.arange(N) * 15 + 14,
                     split=np.array([split_fn(l, s) for l, s in zip(local, seg)], np.int8),
                     symbol=(seg % n_sym).astype(np.int16), local=local, seg_len=seg_len)


def test_eval_windows_score_each_point_once():
    L, Wu, S = 128, 96, 64
    d = fake_patches([700, 300, 230, 150], lambda l, s: 1 if l >= 200 or s == 3 else 0)
    starts, lo, hi = eval_windows(d, L, Wu, S)
    scored = np.concatenate([np.arange(s + a, s + b + 1) for s, a, b in zip(starts, lo, hi)])
    target = np.flatnonzero((d.split == 1) & (d.local >= Wu) & (d.seg_len >= Wu + L))
    assert len(scored) == len(np.unique(scored)) and set(scored) == set(target)
    for s, a, b in zip(starts, lo, hi):
        assert d.local[s] >= Wu and d.local[s] + L <= d.seg_len[s] and b < L
        assert a >= L - S or d.local[s] == Wu or d.local[s] + L == d.seg_len[s]


def test_train_starts_and_weights():
    L, Wu = 128, 96
    d = fake_patches([700, 300, 400], lambda l, s: 0 if l < 500 else 1)
    st = train_starts(d, L, Wu)
    assert (d.local[st] >= Wu).all() and (d.local[st] + L <= d.seg_len[st]).all()
    assert (d.split[st + L - 1] == 0).all()
    p = start_weights(d, st, 0.5)
    mass = np.bincount(d.symbol[st], weights=p)
    n = np.bincount(d.symbol[st])
    np.testing.assert_allclose(mass / mass.sum(), np.sqrt(n) / np.sqrt(n).sum())
    s1 = WindowSampler(st, p, seed=3)
    s1.sample(10)
    state = s1.state_dict()
    a = s1.sample(5)
    s2 = WindowSampler(st, p, seed=99)
    s2.load_state_dict(state)
    np.testing.assert_array_equal(a, s2.sample(5))
