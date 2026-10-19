"""Backbone evaluation on validation windows, paired with the baselines on the same points.

`predict_points` runs the model over evaluation windows and returns, for every scored
prediction point, the model's CE per return horizon, QLIKE per vol horizon and predicted
mean z. `compare` adds the baselines at exactly those points (train marginal for CE,
EWMA and HAR for QLIKE; parameters from $DATA_ROOT/stats/baselines.json) and reports
day-block-bootstrap CIs of the gains plus rank ICs.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from cfm.eval import baselines as bl
from cfm.eval import metrics as mt
from cfm.losses import qlike_losses, return_losses


class GPUData:
    """The PatchData arrays the model needs, as tensors on one device."""

    def __init__(self, d, device: torch.device):
        self.feats = torch.from_numpy(d.feats).to(device)
        self.z = torch.from_numpy(d.z).to(device)
        self.rv = torch.from_numpy(d.rv).to(device)
        self.sigma2 = torch.from_numpy(d.sigma2).to(device)
        self.device = device

    def batch(self, starts: np.ndarray | torch.Tensor, L: int):
        idx = torch.as_tensor(starts, device=self.device)[:, None] + torch.arange(L, device=self.device)
        x = self.feats[idx]                                     # [B, L, P, F]
        return idx, x.reshape(*idx.shape, -1), self.z[idx], self.rv[idx], self.sigma2[idx]


def bin_centers(edges: np.ndarray) -> np.ndarray:
    w = edges[1] - edges[0]
    return np.r_[edges[0] - w / 2, (edges[:-1] + edges[1:]) / 2, edges[-1] + w / 2]


@torch.no_grad()
def predict_points(model, data: GPUData, starts, score_from, score_to, L: int, edges_t: torch.Tensor,
                   vol_horizons: list[int], batch_size: int, amp_dtype=None, symmetric: bool = False) -> dict:
    """Per scored point: idx, CE [n, H], QLIKE [n, Hv], predicted mean z [n, H]; with
    `symmetric` also "ce_sym", the CE of the mirror-symmetrised distribution (bins are
    symmetric around 0), which removes all directional information."""
    model.eval()
    centers = torch.as_tensor(bin_centers(edges_t.cpu().numpy()), device=data.device, dtype=torch.float32)
    out = {"idx": [], "ce": [], "qlike": [], "mean": []} | ({"ce_sym": []} if symmetric else {})
    ctx = torch.autocast(data.device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    ar = torch.arange(L, device=data.device)
    for i in range(0, len(starts), batch_size):
        s = starts[i:i + batch_size]
        idx, x, z, rv, s2 = data.batch(s, L)
        with ctx:
            logits, u = model(x.to(torch.bfloat16 if amp_dtype == torch.bfloat16 else torch.float32))
        lo = torch.as_tensor(score_from[i:i + batch_size], device=data.device)[:, None]
        hi = torch.as_tensor(score_to[i:i + batch_size], device=data.device)[:, None]
        keep = (ar[None, :] >= lo) & (ar[None, :] <= hi)                        # [B, L]
        flat = keep.reshape(-1)
        lg = logits.reshape(-1, *logits.shape[2:])[flat].float()
        zz = z.reshape(-1, z.shape[-1])[flat]
        ones = torch.ones(len(zz), dtype=torch.bool, device=data.device)
        _, _, ce = return_losses(lg, zz, edges_t, ones)
        _, ql = qlike_losses(u.reshape(-1, u.shape[-1])[flat], rv.reshape(-1, rv.shape[-1])[flat],
                             s2.reshape(-1)[flat], vol_horizons, ones)
        out["idx"].append(idx.reshape(-1)[flat].cpu())
        out["ce"].append(ce.cpu())
        out["qlike"].append(ql.cpu())
        out["mean"].append((lg.softmax(-1) @ centers).cpu())
        if symmetric:
            pr = lg.softmax(-1)
            pr = 0.5 * (pr + pr.flip(-1))
            cls = torch.bucketize(torch.nan_to_num(zz), edges_t, right=True)
            ce_sym = -torch.log(pr.gather(-1, cls[..., None])[..., 0].clamp(min=1e-12))
            out["ce_sym"].append(torch.where(torch.isfinite(zz), ce_sym, torch.nan).cpu())
    model.train()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def compare(pred: dict, d, baseline_params: dict, cfg: dict, edges: np.ndarray,
            bootstrap: bool = True) -> tuple[pd.DataFrame, dict]:
    """Model vs. baselines on the model's points -> (table, flat summary dict)."""
    idx = pred["idx"]
    day = d.minute[idx] // 1440
    rows, summary = [], {"points": int(len(idx))}
    t = cfg["targets"]
    for j, h in enumerate(t["return_horizons"]):
        z = d.z[idx, j]
        ok = np.isfinite(z)
        cls = np.searchsorted(edges, z[ok], side="right")
        p_marg = np.asarray(baseline_params["returns"][f"h{h}"]["marginal"])
        losses = {"model": pred["ce"][ok, j], "marginal": mt.cross_entropy(p_marg[cls])}
        summary[f"ce_h{h}"] = float(losses["model"].mean())
        summary[f"ce_gain_h{h}"] = float(losses["marginal"].mean() - losses["model"].mean())
        ic = mt.rank_ic(pred["mean"][ok, j], z[ok], d.minute[idx][ok])
        summary[f"ic_pooled_h{h}"], summary[f"ic_cross_h{h}"] = ic["ic_pooled"], ic["ic_cross"]
        if bootstrap:
            tb = mt.day_bootstrap(day[ok], losses, "marginal")
            tb.insert(0, "horizon", h)
            tb.insert(0, "metric", "ce")
            rows.append(tb)
    for j, h in enumerate(t["vol_horizons"]):
        rv = d.rv[idx, j].astype(np.float64)
        ok = np.isfinite(rv)
        ewma = h * d.sigma2[idx].astype(np.float64)
        har = bl.predict_har(baseline_params["vol"][f"h{h}"]["har"], d.har[idx].astype(np.float64), h)
        losses = {"model": pred["qlike"][ok, j], "ewma": mt.qlike(rv[ok], ewma[ok]),
                  "har": mt.qlike(rv[ok], har[ok])}
        summary[f"qlike_h{h}"] = float(losses["model"].mean())
        summary[f"qlike_gain_vs_ewma_h{h}"] = float(losses["ewma"].mean() - losses["model"].mean())
        summary[f"qlike_gain_vs_har_h{h}"] = float(losses["har"].mean() - losses["model"].mean())
        if bootstrap:
            for base in ["ewma", "har"]:
                tb = mt.day_bootstrap(day[ok], losses, base)
                tb.insert(0, "baseline", base)
                tb.insert(0, "horizon", h)
                tb.insert(0, "metric", "qlike")
                rows.append(tb)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return table, summary
