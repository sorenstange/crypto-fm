#!/bin/bash
# M2: fit the baselines on train, evaluate on validation, log to W&B (group `baselines`).
#     bsub < jobs/baselines.sh
#BSUB -q hpc
#BSUB -J cfm_baselines
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -W 3:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
export OMP_NUM_THREADS=1      # one process per symbol; keep BLAS single-threaded
python -m cfm.eval.evaluate baselines --workers 8
