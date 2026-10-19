#!/bin/bash
# CPU smoke test of cfm.sweep + cfm.eval.compare_runs (5 symbols, 3 tiny runs):
#     bsub < jobs/sweep_smoke.sh
#BSUB -q hpc
#BSUB -J cfm_sweep_smoke
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:45
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
export WANDB_MODE=offline
rm -rf "$RUNS_DIR/sweep_smoke"
cfm_run python -m cfm.sweep configs/sweeps/smoke.yaml --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT HYPERUSDT
python -m cfm.eval.compare_runs "$RUNS_DIR/sweep_smoke"
