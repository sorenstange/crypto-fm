#!/bin/bash
# M3 diagnostics of a finished backbone run (vol-scale vs. directional CE gain, zero returns,
# reversal ICs) on all validation points:
#     bsub < jobs/diagnostics.sh                  # run dir $RUNS_DIR/cfm_bb_v1, best.pt
#BSUB -q gpul40s
#BSUB -J cfm_diag
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
cfm_run python -m cfm.eval.diagnostics --run-dir "$RUNS_DIR/cfm_bb_v1"
