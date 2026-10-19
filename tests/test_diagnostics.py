"""M3 diagnostic helpers: ECDF bin probabilities, symmetrised CE, input-lag control."""

import numpy as np
import torch

from cfm.data.config import load_data_config
from cfm.data.targets import bin_edges
from cfm.eval.diagnostics import ecdf_bin_probs
from cfm.train import InputLag

EDGES = bin_edges(load_data_config())


def test_ecdf_bin_probs_sum_to_one_and_scale():
    rng = np.random.default_rng(0)
    u = np.sort(rng.standard_normal(200_000))
    centers = np.r_[EDGES[0] - 1, (EDGES[:-1] + EDGES[1:]) / 2, EDGES[-1] + 1]   # one z per bin
    for s in [0.5, 1.0, 3.0]:
        p = ecdf_bin_probs(u, centers, np.full(len(centers), s), EDGES)
        assert abs(p.sum() - 1) < 1e-9
    # z = 2u is matched better by the s = 2 rescaled ECDF than by the unscaled one
    z = 2 * rng.standard_normal(50_000)
    ce = lambda s: -np.log(ecdf_bin_probs(u, z, np.full(len(z), s), EDGES)).mean()
    assert ce(2.0) < ce(1.0) - 0.1


def test_symmetrised_ce_matches_for_symmetric_logits():
    from cfm.eval.model_eval import GPUData, predict_points

    class Sym(torch.nn.Module):          # symmetric logits -> ce_sym == ce
        def forward(self, x):
            B, L, _ = x.shape
            lg = -torch.linspace(-3, 3, 66).abs().expand(B, L, 4, 66)
            return lg, torch.zeros(B, L, 2)

    class D:                              # minimal PatchData stand-in
        n = 40
        feats = np.zeros((n, 15, 17), np.float16)
        z = np.random.default_rng(1).standard_normal((n, 4)).astype(np.float32)
        rv = np.ones((n, 2), np.float32)
        sigma2 = np.ones(n, np.float32)

    gd = GPUData(D, torch.device("cpu"))
    pred = predict_points(Sym(), gd, np.array([0]), np.array([0]), np.array([9]), 10,
                          torch.as_tensor(EDGES, dtype=torch.float32), [15, 60], 8, symmetric=True)
    assert np.allclose(pred["ce"], pred["ce_sym"], atol=1e-5)


def test_input_lag_shifts_by_whole_patches():
    seen = {}

    class Echo(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x

    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3) + 1
    InputLag(Echo(), 2)(x)
    assert torch.equal(seen["x"][:, 2:], x[:, :3]) and (seen["x"][:, :2] == 0).all()
