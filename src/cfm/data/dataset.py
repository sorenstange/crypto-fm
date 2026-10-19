"""Patch dataset for the backbone (plan.md §5).

Every symbol's cleaned grid is turned into features (in RAM, `features.build_features`)
and cut into clock-aligned patches of P = 15 candles (:00-:14, :15-:29, ...). Per patch
we keep its [P, F] features and, at its last candle (the prediction point), the return
targets z_h, the realised variances RV_h, the EWMA sigma^2 and the HAR inputs (so the
baselines can be scored on exactly the model's points).

Only data up to the end of validation is loaded (plus one day so the last validation
targets exist); patches ending after `val_end` are dropped. The test split never enters
a training or validation process.

Windows are `context_patches` consecutive patches of one segment, starting at least
`warmup_patches` into the segment:
- train windows: the window's last patch ends in the train split; sampled with replacement,
  each symbol's total mass ~ (its number of train windows)^symbol_weight_power;
- validation windows: deterministic, stride `eval.stride`; each validation prediction
  point is scored exactly once, in the window where it has the most context (at least
  context - stride patches, except right after the warm-up).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from functools import partial

import numpy as np
import pandas as pd

from cfm.data.clean import load_clean, minutes, processed_dir
from cfm.data.config import split_bounds
from cfm.data.features import FEATURES, _trailing_mean, build_features
from cfm.data.load import read_funding
from cfm.data.splits import split_codes
from cfm.data.targets import make_targets
from cfm.data.vol import segment_bounds
from cfm.eval.baselines import HAR_WINDOWS


@dataclass
class PatchData:
    feats: np.ndarray      # [N, P, F] feature dtype
    z: np.ndarray          # [N, H] float32, NaN where the target leaves the segment
    rv: np.ndarray         # [N, Hv] float32
    sigma2: np.ndarray     # [N] float32  EWMA sigma_t^2 at the patch end (deseasonalised)
    har: np.ndarray        # [N, 3] float32  trailing mean variances for HAR
    minute: np.ndarray     # [N] int64  minute of the patch's last candle
    split: np.ndarray      # [N] int8   split code of the patch end (0 train, 1 val, -1 none)
    symbol: np.ndarray     # [N] int16  index into `symbols`
    local: np.ndarray      # [N] int32  patch index within its segment
    seg_len: np.ndarray    # [N] int32  number of patches of its segment


def build_symbol(symbol: str, segments: pd.DataFrame, stats: dict, cfg: dict,
                 feature_dtype: str) -> dict:
    p = cfg["targets"]["patch_minutes"]
    val_end = split_bounds(cfg)["val"][1]
    df = load_clean(symbol, segments)
    df = df[df["ts"] <= val_end + pd.Timedelta(days=1)].reset_index(drop=True)
    empty = {f.name: None for f in fields(PatchData)}
    if len(df) == 0:
        return empty
    val_end_min = val_end.value // 60_000_000_000
    x, aux = build_features(df, read_funding(symbol), stats, cfg)
    vol, m = aux["vol"], aux["minutes"]
    seg = df["segment_id"].to_numpy()
    tg = make_targets(df["close"].to_numpy(), vol, seg, cfg)
    split = split_codes(df["ts"], cfg)
    vd = vol["v"] / vol["season"]

    out = {k: [] for k in empty}
    for a, b in segment_bounds(seg):
        first = a + (-m[a]) % p                                  # first clock-aligned row
        n_p = (b - first) // p
        ends = first + p * np.arange(n_p) + p - 1
        keep = m[ends] <= val_end_min
        n_p = int(keep.sum())                                    # patches are time-ordered
        if n_p == 0:
            continue
        ends = ends[:n_p]
        trailing = np.column_stack([_trailing_mean(vd[a:b], w, include_current=True)[ends - a]
                                    for w in HAR_WINDOWS])
        out["feats"].append(x[first:first + p * n_p].reshape(n_p, p, -1).astype(feature_dtype))
        out["z"].append(np.column_stack([tg["z"][h][ends] for h in cfg["targets"]["return_horizons"]]))
        out["rv"].append(np.column_stack([tg["rv"][h][ends] for h in cfg["targets"]["vol_horizons"]]))
        out["sigma2"].append(vol["sigma"][ends] ** 2)
        out["har"].append(trailing)
        out["minute"].append(m[ends])
        out["split"].append(split[ends])
        out["local"].append(np.arange(n_p, dtype=np.int32))
        out["seg_len"].append(np.full(n_p, n_p, np.int32))
    if not out["feats"]:
        return empty
    res = {k: np.concatenate(v) for k, v in out.items() if k != "symbol"}
    for k in ["z", "rv", "sigma2", "har"]:
        res[k] = res[k].astype(np.float32)
    res["symbol"] = None
    return res


def build_patch_data(symbols: list[str], stats: dict, cfg: dict, bcfg: dict,
                     workers: int | None = None) -> PatchData:
    t0 = time.time()
    segments = pd.read_parquet(processed_dir() / "segments.parquet")
    fn = partial(build_symbol, segments=segments, stats=stats, cfg=cfg,
                 feature_dtype=bcfg["data"]["feature_dtype"])
    with ProcessPoolExecutor(workers or bcfg["data"]["build_workers"]) as pool:
        parts = list(pool.map(fn, symbols, chunksize=1))
    keep = [(i, d) for i, d in enumerate(parts) if d["feats"] is not None]
    cols = {}
    for f in fields(PatchData):
        if f.name == "symbol":
            cols[f.name] = np.concatenate([np.full(len(d["minute"]), i, np.int16) for i, d in keep])
        else:
            cols[f.name] = np.concatenate([d[f.name] for _, d in keep])
    data = PatchData(**cols)
    val_end = split_bounds(cfg)["val"][1]
    assert data.minute.max() * 60_000 <= val_end.value // 10**6, "test data leaked into the dataset"
    print(f"[dataset] {len(data.minute):,} patches x {data.feats.shape[1]} x {data.feats.shape[2]} "
          f"from {len(keep)} symbols ({data.feats.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s",
          flush=True)
    return data


# --- windows --------------------------------------------------------------------------

def train_starts(d: PatchData, L: int, W: int) -> np.ndarray:
    """Global patch indices at which a train window may start."""
    s = np.flatnonzero((d.local >= W) & (d.local + L <= d.seg_len))
    return s[d.split[s + L - 1] == 0]


def start_weights(d: PatchData, starts: np.ndarray, power: float) -> np.ndarray:
    """Sampling probability per start: each symbol's total mass ~ n_symbol^power."""
    sym = d.symbol[starts]
    n = np.bincount(sym).astype(np.float64)
    w = n[sym] ** (power - 1.0)
    return w / w.sum()


class WindowSampler:
    """Weighted sampling of train window starts with a checkpointable numpy RNG."""

    def __init__(self, starts: np.ndarray, probs: np.ndarray, seed: int):
        self.starts = starts
        self.cdf = np.cumsum(probs)
        self.cdf /= self.cdf[-1]
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        return self.starts[np.minimum(np.searchsorted(self.cdf, self.rng.random(n)), len(self.starts) - 1)]

    def state_dict(self) -> dict:
        return self.rng.bit_generator.state

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state


def eval_windows(d: PatchData, L: int, W: int, stride: int, split: int = 1
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window starts, first and last scored offset) so that every prediction point of `split` with
    local index >= W is scored exactly once, with >= L - stride patches of context where
    the segment allows it."""
    starts, score_from, score_to = [], [], []
    seg_first = np.flatnonzero(d.local == 0)
    for f in seg_first:
        n = int(d.seg_len[f])
        if n < W + L:
            continue
        pts = np.flatnonzero(d.split[f + W:f + n] == split) + W       # local indices
        if len(pts) == 0:
            continue
        last_start = n - L
        # Window for point p: start s = W + stride * ceil((p - W - (L - 1)) / stride), clipped.
        s = W + stride * np.ceil(np.maximum(pts - W - (L - 1), 0) / stride).astype(int)
        s = np.minimum(s, last_start)
        for s_val in np.unique(s):
            ps = pts[s == s_val]
            starts.append(f + s_val)
            score_from.append(int(ps.min() - s_val))
            score_to.append(int(ps.max() - s_val))
            assert ps.max() - s_val < L and np.all(np.diff(ps) == 1)
    return (np.asarray(starts, np.int64), np.asarray(score_from, np.int64),
            np.asarray(score_to, np.int64))


def n_features() -> int:
    return len(FEATURES)
