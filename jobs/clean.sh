#!/bin/bash
# Phase 1 cleaning: segment index + cleaning report for all universe symbols.
#     bsub < jobs/clean.sh
#BSUB -q hpc
#BSUB -J cfm_clean
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 1:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
python -m cfm.data.clean --workers 8
