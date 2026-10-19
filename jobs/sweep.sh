#!/bin/bash
# Backbone hyperparameter sweep (cfm.sweep): each array element trains its share of the runs
# in configs/sweeps/<sweep>.yaml one after another on a single data build. The job name
# cfm_<sweep>[1-N] selects the sweep and the number of parts:
#     bsub -J "cfm_reg2[1-2]" < jobs/sweep.sh
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
SWEEP=${LSB_JOBNAME%%\[*}
SWEEP=${SWEEP#cfm_}
PARTS=${LSB_JOBINDEX_END:-$(sed -n 's/.*\[1-\([0-9]*\)\].*/\1/p' <<< "$LSB_JOBNAME")}
PARTS=${PARTS:-1}
echo "[sweep.sh] sweep $SWEEP part ${LSB_JOBINDEX:-1}/$PARTS"
source jobs/common.sh
cfm_run python -m cfm.sweep "configs/sweeps/$SWEEP.yaml" --part "${LSB_JOBINDEX:-1}" --parts "$PARTS"
