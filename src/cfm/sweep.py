"""Train several backbone configurations on one data build (hyperparameter sweeps).

    python -m cfm.sweep configs/sweeps/reg1.yaml [--part 1 --parts 2]

The sweep file names a base config, overrides applied to every run (`base`) and one
override dict per run (`runs`); keys are dotted config paths as for `cfm.train --set`.
Runs are split round-robin into `--parts` (one LSF array element each) and trained one
after another in this process; the data is built once. Run dirs:
`$DATA_ROOT/runs/<sweep>/<run>`; W&B group = sweep name. COMPLETED runs are skipped, an
interrupted run resumes from its last checkpoint when the job is resubmitted. Checkpoints
of completed runs are deleted (the evaluation files stay) unless `--keep-checkpoints`.
Compare finished runs with `python -m cfm.eval.compare_runs $DATA_ROOT/runs/<sweep>`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

from cfm.data.binance import data_root
from cfm.data.config import PROJECT_ROOT, load_data_config
from cfm.data.download import universe_symbols
from cfm.train import apply_overrides, prepare, train_run
from cfm.utils.run import GracefulStop

# Overrides of these change the data build or the windows, which all runs of a sweep share.
SHARED_PREFIXES = ("data.", "eval.stride")


def select_runs(names: list[str], part: int, parts: int) -> list[str]:
    return [n for i, n in enumerate(names) if i % parts == part - 1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", type=Path)
    p.add_argument("--part", type=int, default=1)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--max-minutes", type=float, help="stop (and checkpoint) after this long")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--symbols", nargs="+", help="subset of the universe (smoke tests)")
    args = p.parse_args(argv)

    sw = yaml.safe_load(open(args.sweep))
    name = sw["name"]
    base = apply_overrides(yaml.safe_load(open(PROJECT_ROOT / sw.get("config", "configs/backbone.yaml"))),
                           sw.get("base") or {})
    runs = {r: apply_overrides(base, o or {}) for r, o in sw["runs"].items()}   # validates every key now
    for r, o in sw["runs"].items():
        bad = [k for k in (o or {}) if k.startswith(SHARED_PREFIXES)]
        assert not bad, f"run {r}: {bad} would change the shared data build"
    mine = select_runs(list(runs), args.part, args.parts)
    print(f"[sweep] {name} part {args.part}/{args.parts}: {mine}", flush=True)

    dcfg = load_data_config()
    stopper = GracefulStop(args.max_minutes)
    prep = prepare(base, dcfg, args.symbols or universe_symbols())
    for r in mine:
        run_dir = data_root() / "runs" / name / r
        print(f"[sweep] === {r}: {sw['runs'][r] or 'reference'}", flush=True)
        status = train_run(runs[r], dcfg, prep, run_dir, stopper=stopper, name=f"{name}_{r}", group=name,
                           tags=[name, *sw.get("tags", [])], control={"sweep": name, "run": r,
                                                                      "overrides": sw["runs"][r] or {}})
        if status == "interrupted":
            print("[sweep] stopping: resubmit the job to continue", flush=True)
            return 0
        if not args.keep_checkpoints:
            shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
