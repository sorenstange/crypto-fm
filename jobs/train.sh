#!/bin/bash
# Backbone pretraining (plan.md §9). The job name is the run name: run dir
# $RUNS_DIR/<name>, one W&B run, resumed by later jobs with the same name:
#     bsub -J cfm_bb_v1 < jobs/train.sh
#     jobs/submit_chain.sh 2 jobs/train.sh cfm_bb_v1      # chained, if one job is not enough
#BSUB -q gpua100
#BSUB -J cfm_bb_v1
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -wa USR2
#BSUB -wt 15
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
cfm_run python -m cfm.train --config configs/backbone.yaml --run-dir "$RUNS_DIR/$LSB_JOBNAME"
