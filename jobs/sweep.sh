#!/bin/bash
# Backbone hyperparameter sweep (cfm.sweep): each array element trains its share of the runs
# in configs/sweeps/$SWEEP.yaml one after another on a single data build. Change SWEEP and
# the array size (= --parts) for another sweep:
#     bsub < jobs/sweep.sh
#BSUB -q gpul40s
#BSUB -J "cfm_reg1[1-2]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -wa USR2
#BSUB -wt 15
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

set -euo pipefail
SWEEP=reg1
PARTS=2
source jobs/common.sh
cfm_run python -m cfm.sweep "configs/sweeps/$SWEEP.yaml" --part "$LSB_JOBINDEX" --parts "$PARTS"
