"""Run bookkeeping shared by all jobs: run metadata, W&B init/resume, graceful stop."""

from __future__ import annotations

import json
import os
import platform
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

WANDB_ID_FILE = "wandb_id.txt"
COMPLETED_FILE = "COMPLETED"


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_run_info() -> dict[str, Any]:
    """Git commit, LSF job, node, GPU and library versions for reproducibility."""
    import numpy

    dirty = _git("status", "--porcelain", "--untracked-files=no")
    info: dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) and dirty != "unknown",
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "lsf_job_name": os.environ.get("LSB_JOBNAME"),
        "lsf_job_index": os.environ.get("LSB_JOBINDEX"),
        "lsf_queue": os.environ.get("LSB_QUEUE"),
        "node": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": numpy.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "argv": sys.argv,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import wandb
        info["wandb"] = wandb.__version__
    except ImportError:
        pass
    return info


def write_run_info(run_dir: Path, config: dict[str, Any], info: dict[str, Any]) -> None:
    """One file per job, since a run spans several chained jobs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = info.get("lsf_job_id") or f"local-{os.getpid()}"
    with open(run_dir / f"run_info_{tag}.json", "w") as f:
        json.dump({"config": config, "info": info}, f, indent=2, default=str)


def init_wandb(run_dir: Path, config: dict[str, Any], info: dict[str, Any], *,
               name: str | None = None, group: str | None = None,
               tags: list[str] | None = None, job_type: str = "train"):
    """Start or resume this run's W&B run.

    The run ID is generated on the first job and stored in the run directory, so every
    chained job resumes the same W&B run. All metrics use `trainer/step` as x-axis
    (log it with every call): a job killed between a log and a checkpoint then just
    re-logs a few steps instead of tripping W&B's monotonic-step check.
    """
    import wandb

    id_file = run_dir / WANDB_ID_FILE
    if id_file.exists():
        run_id = id_file.read_text().strip()
    else:
        run_id = secrets.token_hex(4)  # W&B run IDs: [a-z0-9]
        run_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run_id + "\n")

    run = wandb.init(
        id=run_id, resume="allow", name=name or run_dir.name, group=group, tags=tags,
        job_type=job_type, dir=os.environ.get("WANDB_DIR"), config=config,
    )
    run.define_metric("trainer/step")
    run.define_metric("*", step_metric="trainer/step")
    # Record every LSF job that contributed to this run.
    jobs = list(run.summary.get("lsf_jobs", []) or [])
    jobs.append(info.get("lsf_job_id") or "local")
    run.summary["lsf_jobs"] = jobs
    run.config.update({f"run_info_job{len(jobs)}": info}, allow_val_change=True)
    return run


class GracefulStop:
    """Flag set by SIGUSR2 (LSF walltime warning via `#BSUB -wa USR2 -wt <min>`) or
    SIGTERM, or when `max_minutes` have elapsed. The training loop polls `should_stop()`
    and then checkpoints and exits 0, so `bsub -w "done(...)"` chains continue."""

    def __init__(self, max_minutes: float | None = None):
        self.reason: str | None = None
        self._deadline = time.monotonic() + 60 * max_minutes if max_minutes else None
        for sig in (signal.SIGUSR2, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        if self.reason is None:
            self.reason = signal.Signals(signum).name
            print(f"[stop] received {self.reason}; will checkpoint and exit", flush=True)

    def should_stop(self) -> bool:
        if self.reason is None and self._deadline is not None and time.monotonic() > self._deadline:
            self.reason = "max_minutes"
            print("[stop] time budget used up; will checkpoint and exit", flush=True)
        return self.reason is not None
