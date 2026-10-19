#!/bin/bash
# M3 control runs, same config and LR schedule as cfm_bb_v1 but stopped after 3,000 steps:
#   [1] diag_shuffle: train return targets permuted across patches -> validation CE gain must vanish
#   [2] diag_lag1:    inputs end one patch (15 min) before each prediction point
#     bsub < jobs/controls.sh
#BSUB -q gpul40s
#BSUB -J "cfm_controls[1-2]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

set -euo pipefail
source jobs/common.sh
case "$LSB_JOBINDEX" in
  1) name=diag_shuffle; extra=(--shuffle-train-targets) ;;
  2) name=diag_lag1;    extra=(--input-lag 1) ;;
esac
cfm_run python -m cfm.train --config configs/backbone.yaml --run-dir "$RUNS_DIR/$name" \
    --name "$name" --group diagnostics --max-steps 3000 "${extra[@]}"
