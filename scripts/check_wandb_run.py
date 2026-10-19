"""Check that a chained run shows up as ONE continuous W&B run (M-1 acceptance).

Reads the W&B run ID from the run directory, fetches the run's history through the
W&B API and checks that `trainer/step` covers every logging step from start to end.
Light (one API call), so it may run on the login node:

    python scripts/check_wandb_run.py "$RUNS_DIR/<job name>" --total-steps 600 --log-every 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import wandb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    run_id = (args.run_dir / "wandb_id.txt").read_text().strip()
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    project = os.environ.get("WANDB_PROJECT", "crypto-fm")
    run = api.run(f"{entity}/{project}/{run_id}")

    steps = [int(r["trainer/step"]) for r in run.scan_history(keys=["trainer/step", "train/loss"])]
    expected = set(range(args.log_every, args.total_steps + 1, args.log_every))
    missing = sorted(expected - set(steps))
    jobs = [json.loads(line) for line in open(args.run_dir / "jobs.jsonl")]

    print(f"W&B run   {run.url}  state={run.state}")
    print(f"jobs      {[(j['job_id'], j['node'], j['start_step'], j['end_step'], j['status'], j['stop_reason']) for j in jobs]}")
    print(f"history   {len(steps)} rows, steps {min(steps, default=None)}..{max(steps, default=None)}, "
          f"{len(steps) - len(set(steps))} duplicate steps, missing {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"summary   lsf_jobs={run.summary.get('lsf_jobs')} n_jobs={run.summary.get('n_jobs')} "
          f"verify/identical={run.summary.get('verify/identical')}")

    ok = (not missing and run.state == "finished" and len(jobs) >= 2
          and any(j["status"] == "interrupted" for j in jobs)
          and bool(run.summary.get("verify/identical")))
    print("W&B CONTINUITY CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
