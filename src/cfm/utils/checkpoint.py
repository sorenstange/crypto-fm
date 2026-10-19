"""Checkpointing for resumable training across chained LSF jobs.

A checkpoint is a plain dict saved with torch.save. Training code puts in everything
needed to continue bit-for-bit: model, optimizer, scheduler, grad-scaler and sampler
state, the step counter, and the RNG states from `capture_rng_state()`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAST = "last.pt"


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    # RNG states must be CPU ByteTensors, whatever map_location the checkpoint used.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(ckpt_dir: str | Path, state: dict[str, Any], name: str = LAST) -> Path:
    """Write atomically: a job killed mid-write never leaves a truncated checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    path = Path(ckpt_dir) / LAST
    return path if path.exists() else None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # Keep the default "cpu": model/optimizer load_state_dict copy to the parameters'
    # device, and RNG/generator states must stay on the CPU anyway.
    # weights_only=False: our own files, and RNG states are not plain tensors.
    return torch.load(path, map_location=map_location, weights_only=False)
