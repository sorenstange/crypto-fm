"""configs/data.yaml access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


def load_data_config(path: Path | None = None) -> dict:
    with open(path or DATA_CONFIG) as f:
        return yaml.safe_load(f)


def split_bounds(cfg: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """{'train': (start, end), 'val': ..., 'test': ...}, inclusive, UTC; no embargo applied."""
    s = cfg["splits"]
    ts = lambda x: pd.Timestamp(x, tz="UTC")  # noqa: E731
    return {"train": (pd.Timestamp.min.tz_localize("UTC"), ts(s["train_end"])),
            "val": (ts(s["val_start"]), ts(s["val_end"])),
            "test": (ts(s["test_start"]), ts(s["test_end"]))}
