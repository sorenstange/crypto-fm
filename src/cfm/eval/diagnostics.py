"""M3 diagnostics: where does the backbone's validation CE gain over the marginal come from?

    python -m cfm.eval.diagnostics --run-dir "$RUNS_DIR/cfm_bb_v1"

On every validation point the model scores (the same points as the final evaluation),
paired with day-block-bootstrap CIs against the train marginal:
- `marginal_ecdf`: the train marginal from an empirical CDF of z (sanity: ~ `marginal`);
- `marginal_har`: the train distribution of z / s rescaled by s = sqrt(HAR / EWMA variance
  forecast) (h1/h5/h15 use the 15-min HAR, h60 the 60-min one) -> the part of the gain a
  better volatility scale alone gives;
- `model_sym`: the model's distribution mirrored around 0 -> the part of the gain that is
  directional (CE(model_sym) - CE(model)).
Plus the share of exact-zero returns (tick-size discreteness) and the gain on zero vs.
non-zero points, and rank ICs of the model mean vs. simple reversal signals (minus the
last 1-min return / the last patch's summed return, both from the input features).
Writes `<run-dir>/eval/diagnostics_table.csv` and `diagnostics.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfm.data.config import load_data_config
from cfm.data.dataset import build_patch_data, eval_windows, n_features
from cfm.data.download import universe_symbols
from cfm.data.features import FEATURES, load_stats, stats_dir
from cfm.data.targets import bin_edges
from cfm.eval import baselines as bl
from cfm.eval import metrics as mt
from cfm.eval.model_eval import GPUData, predict_points
from cfm.model.backbone import build_backbone
from cfm.utils.checkpoint import load_checkpoint


def ecdf_bin_probs(sorted_u: np.ndarray, z: np.ndarray, s: np.ndarray, edges: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
    """Probability of z's bin when z = s * u with u ~ ECDF(sorted_u); Laplace-smoothed."""
    k = np.searchsorted(edges, z, side="right")
    lo = np.where(k > 0, edges[np.maximum(k - 1, 0)], -np.inf)
    hi = np.where(k < len(edges), edges[np.minimum(k, len(edges) - 1)], np.inf)
    c = np.searchsorted(sorted_u, hi / s, side="left") - np.searchsorted(sorted_u, lo / s, side="left")
    return (c + alpha) / (len(sorted_u) + alpha * (len(edges) + 1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--n-fit", type=int, default=4_000_000, help="train points for the ECDFs")
    args = p.parse_args(argv)

    st = load_checkpoint(args.run_dir / "checkpoints" / args.checkpoint)
    bcfg, dcfg = st["config"]["backbone"], load_data_config()
    L, W = bcfg["data"]["context_patches"], bcfg["data"]["warmup_patches"]
    H, vol_h = dcfg["targets"]["return_horizons"], dcfg["targets"]["vol_horizons"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if bcfg["train"]["precision"] == "bf16" and device.type == "cuda" else None
    bp = json.loads((stats_dir() / "baselines.json").read_text())
    symbols = universe_symbols()

    data = build_patch_data(symbols, load_stats(), dcfg, bcfg)
    gpu = GPUData(data, device)
    model = build_backbone(bcfg, dcfg, n_features()).to(device)
    model.load_state_dict(st["model"])
    edges = bin_edges(dcfg)
    edges_t = torch.as_tensor(edges, device=device, dtype=torch.float32)
    v_starts, v_from, v_to = eval_windows(data, L, W, bcfg["eval"]["stride"])
    pred = predict_points(model, gpu, v_starts, v_from, v_to, L, edges_t, vol_h,
                          bcfg["eval"]["batch_size"], amp, symmetric=True)
    del gpu
    print(f"[diag] {args.checkpoint} (step {st['step']}): {len(pred['idx']):,} validation points", flush=True)

    idx = pred["idx"]
    day, ts, sym = data.minute[idx] // 1440, data.minute[idx], data.symbol[idx]
    tr = np.flatnonzero((data.split == 0) & (data.local >= W))
    fit = np.random.default_rng(0).choice(tr, size=min(args.n_fit, len(tr)), replace=False)

    def har_ratio(i: np.ndarray, hv: int) -> np.ndarray:
        har = bl.predict_har(bp["vol"][f"h{hv}"]["har"], data.har[i].astype(np.float64), hv)
        return har / (hv * data.sigma2[i].astype(np.float64))

    ret = FEATURES.index("ret")
    rev1 = -data.feats[idx, -1, ret].astype(np.float64)
    rev15 = -data.feats[idx, :, ret].astype(np.float64).sum(1)

    tables, summary = [], {"checkpoint": args.checkpoint, "step": int(st["step"]), "points": int(len(idx))}
    for j, h in enumerate(H):
        hv = 15 if h <= 15 else 60
        z = data.z[idx, j].astype(np.float64)
        ok = np.isfinite(z)
        z = z[ok]
        zf = data.z[fit, j].astype(np.float64)
        okf = np.isfinite(zf)
        u_sorted = np.sort(zf[okf] / np.sqrt(har_ratio(fit[okf], hv)))
        cls = np.searchsorted(edges, z, side="right")
        losses = {
            "marginal": mt.cross_entropy(np.asarray(bp["returns"][f"h{h}"]["marginal"])[cls]),
            "marginal_ecdf": mt.cross_entropy(ecdf_bin_probs(np.sort(zf[okf]), z, np.ones_like(z), edges)),
            "marginal_har": mt.cross_entropy(ecdf_bin_probs(u_sorted, z, np.sqrt(har_ratio(idx[ok], hv)), edges)),
            "model_sym": pred["ce_sym"][ok, j].astype(np.float64),
            "model": pred["ce"][ok, j].astype(np.float64),
        }
        tb = mt.day_bootstrap(day[ok], losses, "marginal")
        tb.insert(0, "horizon", h)
        tables.append(tb)
        m = {k: float(v.mean()) for k, v in losses.items()}
        zero = z == 0
        mean = pred["mean"][ok, j]
        s = {
            "points": int(ok.sum()), "zero_share": float(zero.mean()), "ce": m,
            "gain_model": m["marginal"] - m["model"],
            "gain_vol_scale_har": m["marginal"] - m["marginal_har"],
            "gain_model_over_har_marginal": m["marginal_har"] - m["model"],
            "gain_directional": m["model_sym"] - m["model"],
            "gain_model_zero_pts": float((losses["marginal"] - losses["model"])[zero].mean()) if zero.any() else None,
            "gain_model_nonzero_pts": float((losses["marginal"] - losses["model"])[~zero].mean()),
            "gain_directional_nonzero_pts": float((losses["model_sym"] - losses["model"])[~zero].mean()),
            "ic_model": mt.rank_ic(mean, z, ts[ok]),
            "ic_rev1": mt.rank_ic(rev1[ok], z, ts[ok]),
            "ic_rev15": mt.rank_ic(rev15[ok], z, ts[ok]),
            "spearman_model_rev1": mt.spearman(mean, rev1[ok]),
            "spearman_model_rev15": mt.spearman(mean, rev15[ok]),
        }
        if h == H[0]:   # tick-size discreteness per symbol
            df = pd.DataFrame({"symbol": sym[ok], "zero": zero, "gain": losses["marginal"] - losses["model"]})
            g = df.groupby("symbol").agg(zero_share=("zero", "mean"), gain=("gain", "mean"), n=("zero", "size"))
            g.index = [symbols[i] for i in g.index]
            s["per_symbol_spearman_zero_share_vs_gain"] = mt.spearman(g["zero_share"].to_numpy(), g["gain"].to_numpy())
            s["top_zero_share_symbols"] = g.sort_values("zero_share", ascending=False).head(10).round(4).to_dict("index")
            s["zero_share_quantiles_over_symbols"] = g["zero_share"].quantile([0.1, 0.5, 0.9]).round(4).tolist()
        summary[f"h{h}"] = s
        print(f"[diag] h{h}: " + json.dumps({k: v for k, v in s.items() if k != "top_zero_share_symbols"}), flush=True)

    table = pd.concat(tables, ignore_index=True)
    out = args.run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "diagnostics_table.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=1))
    print("[diag] CE per model with day-bootstrap CIs (gain = vs. train marginal):\n"
          + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
