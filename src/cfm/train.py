"""Backbone pretraining (plan.md §9) with checkpoint/resume across chained LSF jobs.

    python -m cfm.train --config configs/backbone.yaml --run-dir "$RUNS_DIR/<name>" \
        [--set model.dropout=0.2 --set loss.ret_weights=[1,1,0.5,0.25] ...]

- Data: `dataset.build_patch_data` (features in RAM, then on the GPU); train windows
  sampled with sqrt-size symbol weights; a fixed subset of validation windows for the
  periodic evaluation, all validation windows for the final one. `prepare` builds this
  once, so `cfm.sweep` can train several configurations on the same data.
- Loss: sum_h w_h CE_h + w_vol * sum_h QLIKE_h + z_loss * z-loss (`losses.py`); CE_h can use
  HL-Gauss soft targets (`loss.hl_gauss_sigma`, in z units). Logged and evaluated CE is
  always the hard (one-hot) CE.
- AdamW, WSD schedule, grad clipping, bf16 autocast, torch.compile.
- Every `eval_every` steps: validation subset -> CE gain over the train marginal, QLIKE vs.
  EWMA/HAR, rank IC; the best checkpoint (lowest unweighted validation loss
  sum_h CE_h + w_vol sum_h QLIKE_h) is kept; early stop after `early_stop_evals`
  evaluations without improvement.
- Checkpoint (model, optimizer, scheduler, sampler, RNG, step, best) every
  `ckpt_every_minutes`, on SIGUSR2/SIGTERM and at the end; resumed jobs continue the
  same W&B run.
- Diagnostic controls (M3): `--shuffle-train-targets` permutes the return targets across
  train patches (validation gains must then vanish); `--input-lag k` shows the model only
  the inputs up to k patches before each prediction point.
- When training ends: the best checkpoint is evaluated on all validation points with
  day-block-bootstrap CIs (`<run-dir>/eval/`; with `eval.save_points` also the per-point
  losses for `cfm.eval.compare_runs`), and the run is marked COMPLETED.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from cfm.data.config import PROJECT_ROOT, load_data_config
from cfm.data.dataset import (PatchData, WindowSampler, build_patch_data, eval_windows, n_features,
                              start_weights, train_starts)
from cfm.data.download import universe_symbols
from cfm.data.features import load_stats, stats_dir
from cfm.data.targets import bin_edges
from cfm.eval.model_eval import GPUData, compare, predict_points
from cfm.losses import qlike_losses, return_losses
from cfm.model.backbone import build_backbone
from cfm.utils.checkpoint import (capture_rng_state, latest_checkpoint, load_checkpoint,
                                  restore_rng_state, save_checkpoint, seed_all)
from cfm.utils.run import COMPLETED_FILE, GracefulStop, collect_run_info, init_wandb, write_run_info

BEST = "best.pt"
POINTS_FILE = "val_points.npz"


class InputLag(torch.nn.Module):
    """Shift inputs [B, L, D] by `lag` patches along L (zeros in front): position p sees patches <= p - lag."""

    def __init__(self, model, lag: int):
        super().__init__()
        self.model, self.lag = model, lag

    def forward(self, x: torch.Tensor):
        return self.model(torch.cat([torch.zeros_like(x[:, :self.lag]), x[:, :-self.lag]], dim=1))


def wsd(step: int, total: int, warmup: int, decay_frac: float) -> float:
    decay_start = int((1 - decay_frac) * total)
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))


def val_loss(summary: dict, cfg: dict, w_vol: float) -> float:
    t = cfg["targets"]
    return (sum(summary[f"ce_h{h}"] for h in t["return_horizons"])
            + w_vol * sum(summary[f"qlike_h{h}"] for h in t["vol_horizons"]))


def parse_sets(sets: list[str]) -> dict:
    """["section.key=value", ...] (values parsed as YAML) -> {"section.key": value}."""
    out = {}
    for s in sets:
        path, sep, raw = s.partition("=")
        if not sep:
            raise ValueError(f"override {s!r} is not KEY=VALUE")
        out[path.strip()] = yaml.safe_load(raw)
    return out


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """{"section.key": value} -> a copy of cfg with those values; every key must already exist."""
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = cfg
        for k in parents:
            node = node.get(k) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[leaf] = value
    return cfg


@dataclass
class Prepared:
    """Everything a training run needs that does not depend on the run's hyperparameters."""
    data: PatchData
    gpu: GPUData
    symbols: list[str]
    starts: np.ndarray
    v_starts: np.ndarray
    v_from: np.ndarray
    v_to: np.ndarray
    baseline_params: dict
    device: torch.device


def prepare(bcfg: dict, dcfg: dict, symbols: list[str], shuffle_train_targets: bool = False) -> Prepared:
    L, W = bcfg["data"]["context_patches"], bcfg["data"]["warmup_patches"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_patch_data(symbols, load_stats(), dcfg, bcfg)
    if shuffle_train_targets:
        tr = np.flatnonzero(data.split == 0)
        data.z[tr] = data.z[np.random.default_rng(bcfg["train"]["seed"] + 2).permutation(tr)]
        print(f"[train] CONTROL: return targets of {len(tr):,} train patches shuffled", flush=True)
    starts = train_starts(data, L, W)
    v_starts, v_from, v_to = eval_windows(data, L, W, bcfg["eval"]["stride"])
    print(f"[train] {len(starts):,} train window starts, {len(v_starts):,} val windows, device {device}",
          flush=True)
    return Prepared(data, GPUData(data, device), symbols, starts, v_starts, v_from, v_to,
                    json.loads((stats_dir() / "baselines.json").read_text()), device)


def train_run(bcfg: dict, dcfg: dict, prep: Prepared, run_dir: Path, *, stopper: GracefulStop,
              name: str | None = None, group: str = "backbone", tags: list[str] | None = None,
              max_steps: int | None = None, input_lag: int = 0, control: dict | None = None) -> str:
    """Train (or resume) one run on prepared data -> status: completed | interrupted | diverged."""
    tc, lc = bcfg["train"], bcfg["loss"]
    L = bcfg["data"]["context_patches"]
    ckpt_dir = run_dir / "checkpoints"
    if (run_dir / COMPLETED_FILE).exists():
        print(f"[train] {run_dir} is already COMPLETED; nothing to do", flush=True)
        return "completed"

    seed_all(tc["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device, data, gpu = prep.device, prep.data, prep.gpu
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tc["precision"])
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if amp_dtype else nullcontext
    sampler = WindowSampler(prep.starts, start_weights(data, prep.starts, bcfg["data"]["symbol_weight_power"]),
                            seed=tc["seed"])
    sub = np.sort(np.random.default_rng(tc["seed"] + 1).choice(
        len(prep.v_starts), size=min(tc["eval_windows"], len(prep.v_starts)), replace=False))
    edges = bin_edges(dcfg)
    edges_t = torch.as_tensor(edges, device=device, dtype=torch.float32)
    ret_h, vol_h = dcfg["targets"]["return_horizons"], dcfg["targets"]["vol_horizons"]
    ret_w = torch.as_tensor(lc["ret_weights"], device=device, dtype=torch.float32)
    assert len(ret_w) == len(ret_h), "loss.ret_weights needs one weight per return horizon"

    # --- model ---
    torch._dynamo.reset()            # several runs per process (cfm.sweep): don't hit the recompile limit
    model = build_backbone(bcfg, dcfg, n_features()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.param_groups(tc["weight_decay"]), lr=tc["lr"], betas=tuple(tc["betas"]),
                            eps=tc["eps"], fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: wsd(s, tc["total_steps"], tc["warmup_steps"], tc["decay_frac"]))
    scaler = torch.amp.GradScaler(device.type, enabled=tc["precision"] == "fp16")
    fwd = torch.compile(model) if tc["compile"] and device.type == "cuda" else model
    if input_lag:
        fwd = InputLag(fwd, input_lag)
        print(f"[train] CONTROL: inputs lagged by {input_lag} patch(es)", flush=True)
    last_step = min(tc["total_steps"], max_steps or tc["total_steps"])

    step, best, since_best = 0, math.inf, 0
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is not None:
        st = load_checkpoint(ckpt)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        sampler.load_state_dict(st["sampler"])
        restore_rng_state(st["rng"])
        step, best, since_best = st["step"], st["best"], st["since_best"]
        print(f"[resume] {ckpt} at step {step} (best val {best:.4f})", flush=True)

    config = {"backbone": bcfg, "data": dcfg, "n_params": n_params, "symbols": len(prep.symbols),
              "control": control or {}}
    info = collect_run_info()
    write_run_info(run_dir, config, info)
    run = init_wandb(run_dir, config, info, name=name, group=group,
                     tags=tags or ["M3", f"d{bcfg['model']['d_model']}", f"L{bcfg['model']['n_layers']}"])
    print(f"[train] {n_params / 1e6:.2f}M params; W&B {run.url}", flush=True)

    def state(s: int) -> dict:
        return {"step": s, "best": best, "since_best": since_best, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "sampler": sampler.state_dict(),
                "rng": capture_rng_state(), "config": config}

    def loss_on(starts_b):
        _, x, z, rv, s2 = gpu.batch(starts_b, L)
        with autocast():
            logits, u = fwd(x.to(amp_dtype or torch.float32))
        B = x.shape[0] * x.shape[1]
        ones = torch.ones(B, dtype=torch.bool, device=device)
        obj_h, zl, ce_pt = return_losses(logits.reshape(B, *logits.shape[2:]), z.reshape(B, -1), edges_t, ones,
                                         soft_sigma=lc["hl_gauss_sigma"])
        ql_h, _ = qlike_losses(u.reshape(B, -1), rv.reshape(B, -1), s2.reshape(B), vol_h, ones)
        loss = (ret_w * obj_h).sum() + lc["w_vol"] * ql_h.sum() + lc["z_loss"] * zl
        return loss, ce_pt.detach().nanmean(0), ql_h, zl

    def evaluate(windows, bootstrap: bool):
        pred = predict_points(fwd, gpu, prep.v_starts[windows], prep.v_from[windows], prep.v_to[windows], L,
                              edges_t, vol_h, bcfg["eval"]["batch_size"], amp_dtype)
        return (pred, *compare(pred, data, prep.baseline_params, dcfg, edges, bootstrap=bootstrap))

    if step == 0:   # sanity: initial CE should be ~ ln(66)
        with torch.no_grad():
            model.eval()
            _, ce_h, ql_h, _ = loss_on(sampler.sample(tc["batch_size"]))
            model.train()
        init = {f"init/ce_h{h}": float(v) for h, v in zip(ret_h, ce_h)}
        init |= {f"init/qlike_h{h}": float(v) for h, v in zip(vol_h, ql_h)}
        print(f"[train] initial losses {init} (ln 66 = {math.log(66):.3f})", flush=True)
        run.log({"trainer/step": 0, **init})

    model.train()
    t_ckpt = t_log = time.time()
    tokens = 0
    status = "completed"
    while step < last_step:
        if stopper.should_stop():
            status = "interrupted"
            break
        loss, ce_h, ql_h, zl = loss_on(sampler.sample(tc["batch_size"]))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        tokens += tc["batch_size"] * L

        if step % tc["log_every"] == 0:
            dt = time.time() - t_log
            log = {"trainer/step": step, "train/loss": loss.item(), "train/z_loss": zl.item(),
                   "train/grad_norm": gnorm.item(), "train/lr": sched.get_last_lr()[0],
                   "train/patches_per_s": tokens / dt,
                   "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0}
            log |= {f"train/ce_h{h}": v.item() for h, v in zip(ret_h, ce_h)}
            log |= {f"train/qlike_h{h}": v.item() for h, v in zip(vol_h, ql_h)}
            run.log(log)
            if not math.isfinite(log["train/loss"]):
                print(f"[train] non-finite loss at step {step}; stopping", flush=True)
                run.alert(title="cfm: non-finite loss", text=f"{run_dir.name} step {step}")
                status = "diverged"
                break
            t_log, tokens = time.time(), 0

        if step % tc["eval_every"] == 0 or step == last_step:
            _, _, summ = evaluate(sub, bootstrap=False)
            vl = val_loss(summ, dcfg, lc["w_vol"])
            if vl < best:
                best, since_best = vl, 0
                save_checkpoint(ckpt_dir, state(step), name=BEST)
            else:
                since_best += 1
            run.log({"trainer/step": step, "val/loss": vl, "val/best": best,
                     **{f"val/{k}": v for k, v in summ.items()}})
            print(f"[eval] step {step} val {vl:.4f} (best {best:.4f}, {since_best} since) "
                  + " ".join(f"{k}={v:.4f}" for k, v in summ.items() if k.startswith(("ce_gain", "qlike_gain"))),
                  flush=True)
            if since_best >= tc["early_stop_evals"]:
                print(f"[train] early stop at step {step}", flush=True)
                break
        if time.time() - t_ckpt > 60 * tc["ckpt_every_minutes"]:
            save_checkpoint(ckpt_dir, state(step))
            t_ckpt = time.time()

    save_checkpoint(ckpt_dir, state(step))
    if status == "interrupted":
        print(f"[train] stopped ({stopper.reason}) at step {step}; checkpoint saved", flush=True)
        run.finish()
        return status

    # --- final evaluation of the best checkpoint on all validation points ---
    best_path = ckpt_dir / BEST
    best_step = step
    if best_path.exists():
        st = load_checkpoint(best_path)
        model.load_state_dict(st["model"])
        best_step = st["step"]
    pred, table, summ = evaluate(np.arange(len(prep.v_starts)), bootstrap=True)
    summ = {"best_step": int(best_step), "last_step": int(step), "val_loss": val_loss(summ, dcfg, lc["w_vol"])} | summ
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "val_vs_baselines.csv", index=False)
    (out / "val_summary.json").write_text(json.dumps(summ, indent=1))
    if bcfg["eval"]["save_points"]:
        np.savez(out / POINTS_FILE, idx=pred["idx"].astype(np.int32),
                 day=(data.minute[pred["idx"]] // 1440).astype(np.int32),
                 ce=pred["ce"].astype(np.float16), qlike=pred["qlike"].astype(np.float16))
    import wandb
    run.log({"trainer/step": step, "final/table": wandb.Table(dataframe=table),
             **{f"final/{k}": v for k, v in summ.items()}})
    print("[final] best checkpoint on all validation points:\n"
          + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print("[final] " + json.dumps(summ), flush=True)
    (run_dir / COMPLETED_FILE).write_text(f"{status} at step {step}\n")
    run.finish()
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "backbone.yaml")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. model.dropout=0.2 (repeatable)")
    p.add_argument("--symbols", nargs="+", help="subset of the universe (smoke tests)")
    p.add_argument("--total-steps", type=int, help="override train.total_steps")
    p.add_argument("--max-steps", type=int, help="stop after this many steps (keeps the LR schedule of total_steps)")
    p.add_argument("--max-minutes", type=float, help="stop (and checkpoint) after this long")
    p.add_argument("--shuffle-train-targets", action="store_true", help="control: permute train return targets")
    p.add_argument("--input-lag", type=int, default=0, help="control: inputs end this many patches early")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--name", help="W&B run name (default: run dir name)")
    p.add_argument("--group", default="backbone")
    args = p.parse_args(argv)

    bcfg = apply_overrides(yaml.safe_load(open(args.config)), parse_sets(args.sets))
    if args.total_steps:
        bcfg["train"]["total_steps"] = args.total_steps
    if args.no_compile:
        bcfg["train"]["compile"] = False
    dcfg = load_data_config()
    if (args.run_dir / COMPLETED_FILE).exists():
        print(f"[train] {args.run_dir} is already COMPLETED; nothing to do", flush=True)
        return 0
    stopper = GracefulStop(args.max_minutes)
    prep = prepare(bcfg, dcfg, args.symbols or universe_symbols(), args.shuffle_train_targets)
    train_run(bcfg, dcfg, prep, args.run_dir, stopper=stopper, name=args.name, group=args.group,
              max_steps=args.max_steps, input_lag=args.input_lag,
              control={"shuffle_train_targets": args.shuffle_train_targets, "input_lag": args.input_lag,
                       "max_steps": args.max_steps, "sets": args.sets})
    return 0


if __name__ == "__main__":
    sys.exit(main())
