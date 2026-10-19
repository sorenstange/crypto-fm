"""Checkpoint-and-resume smoke test (M-1).

Trains a tiny model on synthetic data with everything a real run has to restore:
model, AdamW, LR schedule, grad scaler, data-sampler generator, dropout RNG and step.
Run it as a chain of LSF jobs sharing one run directory. A job stops early on the LSF
walltime warning (SIGUSR2), on SIGTERM, after `--max-minutes` or after
`--stop-after-steps`, checkpoints and exits 0. The next job resumes from the checkpoint
and the same W&B run.

With `--verify`, the job that finishes the run then retrains the same config from
scratch without interruption in-process and checks that the per-step losses and final
parameters are identical to the resumed run.

    python -m cfm.smoke.resume_test --run-dir "$RUNS_DIR/$LSB_JOBNAME" --verify
"""

from __future__ import annotations

import os

# Deterministic cuBLAS; must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
import socket
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfm.utils.checkpoint import (capture_rng_state, latest_checkpoint, load_checkpoint,
                                  restore_rng_state, save_checkpoint)
from cfm.utils.run import COMPLETED_FILE, GracefulStop, collect_run_info, init_wandb, write_run_info


@dataclass
class Config:
    total_steps: int = 600
    seed: int = 0
    batch_size: int = 64
    d_in: int = 32
    d_hidden: int = 256
    n_classes: int = 66
    dataset_size: int = 20_000
    dropout: float = 0.1
    lr: float = 1e-3
    warmup_steps: int = 50
    precision: str = "bf16"  # bf16 | fp16 | fp32
    log_every: int = 10
    ckpt_every: int = 100
    step_sleep: float = 0.0  # seconds per step; stretches the run so LSF signals land mid-run


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wsd_lambda(cfg: Config):
    """Warmup-stable-decay: linear warmup, constant, linear decay over the last 20%."""
    decay_start = int(0.8 * cfg.total_steps)

    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if step < decay_start:
            return 1.0
        return max(0.0, (cfg.total_steps - step) / (cfg.total_steps - decay_start))

    return f


def build(cfg: Config, device: torch.device):
    # Fixed synthetic dataset: noisy labels from a random linear map.
    g_data = torch.Generator().manual_seed(cfg.seed + 1)
    x = torch.randn(cfg.dataset_size, cfg.d_in, generator=g_data)
    w = torch.randn(cfg.d_in, cfg.n_classes, generator=g_data)
    y = (x @ w + 2.0 * torch.randn(cfg.dataset_size, cfg.n_classes, generator=g_data)).argmax(1)

    model = nn.Sequential(
        nn.Linear(cfg.d_in, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
        nn.Linear(cfg.d_hidden, cfg.n_classes),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda(cfg))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.precision == "fp16")
    sampler = torch.Generator().manual_seed(cfg.seed + 2)  # stands in for the data sampler
    return model, opt, sched, scaler, sampler, x.to(device), y.to(device)


def train(cfg: Config, device: torch.device, run_dir: Path | None = None,
          stopper: GracefulStop | None = None, wandb_run=None,
          stop_after_steps: int | None = None) -> dict:
    """Train to cfg.total_steps, resuming from run_dir/checkpoints if present.

    Returns {"status": "completed"|"interrupted", "start_step", "step", "losses", "params", ...}.
    """
    seed_all(cfg.seed)
    model, opt, sched, scaler, sampler, x, y = build(cfg, device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.precision)
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if amp_dtype else nullcontext
    ckpt_dir = run_dir / "checkpoints" if run_dir else None

    step, losses = 0, []
    ckpt = latest_checkpoint(ckpt_dir) if ckpt_dir else None
    if ckpt is not None:
        state = load_checkpoint(ckpt)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        sampler.set_state(state["sampler"])
        restore_rng_state(state["rng"])
        step, losses = state["step"], state["losses"]
        print(f"[resume] loaded {ckpt} at step {step}", flush=True)
    start_step = step

    def save() -> None:
        if ckpt_dir is None:
            return
        save_checkpoint(ckpt_dir, {
            "step": step, "config": asdict(cfg), "losses": losses,
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "sampler": sampler.get_state(), "rng": capture_rng_state(),
        })
        print(f"[ckpt] saved step {step}", flush=True)

    model.train()
    t_last = time.time()
    while step < cfg.total_steps:
        if (stopper is not None and stopper.should_stop()) or \
                (stop_after_steps is not None and step - start_step >= stop_after_steps):
            save()
            return {"status": "interrupted", "start_step": start_step, "step": step, "losses": losses}

        idx = torch.randint(cfg.dataset_size, (cfg.batch_size,), generator=sampler).to(device)
        with autocast():
            loss = F.cross_entropy(model(x[idx]).float(), y[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        losses.append(loss.item())

        if wandb_run is not None and step % cfg.log_every == 0:
            now = time.time()
            wandb_run.log({
                "trainer/step": step,
                "train/loss": float(np.mean(losses[-cfg.log_every:])),
                "train/lr": sched.get_last_lr()[0],
                "train/grad_norm": gnorm.item(),
                "train/steps_per_s": cfg.log_every / (now - t_last),
            })
            t_last = now
        if step % cfg.ckpt_every == 0:
            save()
        if cfg.step_sleep:
            time.sleep(cfg.step_sleep)

    save()
    params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"status": "completed", "start_step": start_step, "step": step, "losses": losses, "params": params}


def verify(cfg: Config, device: torch.device, resumed: dict) -> dict:
    """Retrain without interruption and compare with the resumed run."""
    ref = train(Config(**{**asdict(cfg), "step_sleep": 0.0}), device)
    loss_diff = float(np.max(np.abs(np.array(ref["losses"]) - np.array(resumed["losses"]))))
    param_diff = max((ref["params"][k].float() - resumed["params"][k].float()).abs().max().item()
                     for k in ref["params"])
    return {"n_steps": len(resumed["losses"]), "max_abs_loss_diff": loss_diff,
            "max_abs_param_diff": param_diff, "identical": loss_diff == 0.0 and param_diff == 0.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--total-steps", type=int, default=Config.total_steps)
    p.add_argument("--step-sleep", type=float, default=Config.step_sleep)
    p.add_argument("--precision", default=Config.precision, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--ckpt-every", type=int, default=Config.ckpt_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-minutes", type=float, default=None, help="stop and checkpoint after this long")
    p.add_argument("--stop-after-steps", type=int, default=None, help="stop and checkpoint after N steps in this job")
    p.add_argument("--verify", action="store_true", help="after completion, compare with an uninterrupted run")
    p.add_argument("--wandb-group", default="m-1")
    args = p.parse_args(argv)

    run_dir: Path = args.run_dir
    if (run_dir / COMPLETED_FILE).exists():
        print(f"[resume_test] {run_dir} already completed; nothing to do")
        return 0

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    cfg = Config(total_steps=args.total_steps, step_sleep=args.step_sleep, precision=args.precision,
                 ckpt_every=args.ckpt_every, seed=args.seed)
    info = collect_run_info()
    write_run_info(run_dir, asdict(cfg), info)
    stopper = GracefulStop(max_minutes=args.max_minutes)
    run = init_wandb(run_dir, asdict(cfg), info, group=args.wandb_group,
                     tags=["m-1", "resume-test"], job_type="smoke")

    t0 = time.time()
    result = train(cfg, device, run_dir, stopper, run, args.stop_after_steps)
    job_record = {"job_id": info["lsf_job_id"], "node": socket.gethostname(), "gpu": info["gpu"],
                  "start_step": result["start_step"], "end_step": result["step"], "status": result["status"],
                  "stop_reason": stopper.reason, "seconds": round(time.time() - t0, 1),
                  "wandb_run_id": run.id, "wandb_mode": os.environ.get("WANDB_MODE")}
    with open(run_dir / "jobs.jsonl", "a") as f:
        f.write(json.dumps(job_record) + "\n")
    print(f"[resume_test] {json.dumps(job_record)}", flush=True)

    rc = 0
    if result["status"] == "interrupted":
        run.mark_preempting()
    else:
        n_jobs = sum(1 for _ in open(run_dir / "jobs.jsonl"))
        run.summary["n_jobs"] = n_jobs
        if args.verify:
            report = verify(cfg, device, result)
            report["n_jobs"] = n_jobs
            (run_dir / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"[verify] {json.dumps(report)}", flush=True)
            run.summary.update({f"verify/{k}": v for k, v in report.items()})
            rc = 0 if report["identical"] else 1
        (run_dir / COMPLETED_FILE).write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    run.finish(exit_code=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
