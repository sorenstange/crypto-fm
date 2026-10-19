"""Paired comparison of finished backbone runs on identical validation points.

    python -m cfm.eval.compare_runs $DATA_ROOT/runs/reg1 [--ref ref] [--w-vol 1]
    python -m cfm.eval.compare_runs RUN_DIR RUN_DIR ...

Every run must have `eval/val_points.npz` (`eval.save_points: true`). For each metric
(CE per return horizon, QLIKE per vol horizon, and the total val loss
sum_h CE_h + w_vol sum_h QLIKE_h) the mean per run with a 95% day-block-bootstrap CI and
the paired gain over the reference run (positive = better than the reference), on the
points where every run's target is valid. Writes `compare.csv` next to the runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.data.config import load_data_config
from cfm.eval import metrics as mt

POINTS = Path("eval") / "val_points.npz"


def load_runs(dirs: list[Path]) -> tuple[dict[str, dict], np.ndarray]:
    runs, idx0, day = {}, None, None
    for d in dirs:
        z = np.load(d / POINTS)
        if idx0 is None:
            idx0, day = z["idx"], z["day"]
        elif not np.array_equal(z["idx"], idx0):
            raise ValueError(f"{d.name}: different validation points than {next(iter(runs))}")
        runs[d.name] = {"ce": z["ce"].astype(np.float64), "qlike": z["qlike"].astype(np.float64)}
    return runs, day


def compare_runs(runs: dict[str, dict], day: np.ndarray, ref: str, ret_h: list[int], vol_h: list[int],
                 w_vol: float = 1.0) -> pd.DataFrame:
    metrics = {f"ce_h{h}": lambda r, j=j: r["ce"][:, j] for j, h in enumerate(ret_h)}
    metrics |= {f"qlike_h{h}": lambda r, j=j: r["qlike"][:, j] for j, h in enumerate(vol_h)}
    metrics["val_loss"] = lambda r: r["ce"].sum(1) + w_vol * r["qlike"].sum(1)
    rows = []
    for m, get in metrics.items():
        losses = {n: get(r) for n, r in runs.items()}
        ok = np.logical_and.reduce([np.isfinite(v) for v in losses.values()])
        tb = mt.day_bootstrap(day[ok], {n: v[ok] for n, v in losses.items()}, ref)
        tb.insert(0, "metric", m)
        rows.append(tb.rename(columns={"model": "run"}))
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", type=Path, nargs="+", help="run dirs, or one sweep dir containing them")
    p.add_argument("--ref", help="reference run (default: 'ref' if present, else the first)")
    p.add_argument("--w-vol", type=float, default=1.0)
    args = p.parse_args(argv)

    dirs = args.dirs
    if len(dirs) == 1 and not (dirs[0] / POINTS).exists():
        dirs = sorted(d for d in dirs[0].iterdir() if (d / POINTS).exists())
    runs, day = load_runs(dirs)
    ref = args.ref or ("ref" if "ref" in runs else next(iter(runs)))
    t = load_data_config()["targets"]
    table = compare_runs(runs, day, ref, t["return_horizons"], t["vol_horizons"], args.w_vol)
    steps = {d.name: json.loads((d / "eval" / "val_summary.json").read_text()) for d in dirs}
    table["best_step"] = table["run"].map(lambda n: steps[n].get("best_step"))
    table["last_step"] = table["run"].map(lambda n: steps[n].get("last_step"))
    out = dirs[0].parent / "compare.csv"
    table.to_csv(out, index=False)
    print(f"[compare] reference {ref}; gain = ref - run (positive = better than {ref}); {out}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
