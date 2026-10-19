"""CPU tests for checkpoint/resume (the GPU version runs as chained LSF jobs)."""

import json
import os
import signal

import pytest
import torch

from cfm.smoke.resume_test import Config, main, train
from cfm.utils.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from cfm.utils.run import COMPLETED_FILE, WANDB_ID_FILE, GracefulStop

CPU = torch.device("cpu")


def test_rng_roundtrip():
    state = capture_rng_state()
    a = (torch.rand(3), torch.randn(2))
    restore_rng_state(state)
    b = (torch.rand(3), torch.randn(2))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_save_is_atomic_and_loadable(tmp_path):
    path = save_checkpoint(tmp_path, {"step": 7, "rng": capture_rng_state()})
    assert path.exists() and not list(tmp_path.glob("*.tmp"))
    assert load_checkpoint(path)["step"] == 7


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_resume_is_bitwise_identical(tmp_path, precision):
    # Interrupt off a checkpoint boundary, twice, so the stop-time checkpoint is exercised.
    cfg = Config(total_steps=120, ckpt_every=25, precision=precision, dataset_size=2000)
    r1 = train(cfg, CPU, tmp_path, stop_after_steps=37)
    assert r1["status"] == "interrupted" and r1["step"] == 37
    r2 = train(cfg, CPU, tmp_path, stop_after_steps=40)
    assert r2["status"] == "interrupted" and (r2["start_step"], r2["step"]) == (37, 77)
    r3 = train(cfg, CPU, tmp_path)
    assert r3["status"] == "completed" and r3["start_step"] == 77

    ref = train(cfg, CPU)
    assert ref["losses"] == r3["losses"]
    assert all(torch.equal(ref["params"][k], r3["params"][k]) for k in ref["params"])


def test_graceful_stop_on_sigusr2():
    stopper = GracefulStop()
    assert not stopper.should_stop()
    os.kill(os.getpid(), signal.SIGUSR2)
    assert stopper.should_stop() and stopper.reason == "SIGUSR2"
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_main_two_jobs_one_wandb_id(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    run_dir = tmp_path / "run"
    common = ["--run-dir", str(run_dir), "--device", "cpu", "--total-steps", "60", "--ckpt-every", "20"]
    assert main(common + ["--stop-after-steps", "30"]) == 0
    wandb_id = (run_dir / WANDB_ID_FILE).read_text()
    assert main(common + ["--verify"]) == 0
    assert (run_dir / WANDB_ID_FILE).read_text() == wandb_id
    assert (run_dir / COMPLETED_FILE).exists()
    jobs = [json.loads(line) for line in open(run_dir / "jobs.jsonl")]
    assert [(j["start_step"], j["end_step"], j["status"]) for j in jobs] == \
        [(0, 30, "interrupted"), (30, 60, "completed")]
    assert json.loads((run_dir / "verify.json").read_text())["identical"]
    assert main(common) == 0  # a further chained job is a no-op
