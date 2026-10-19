#!/bin/bash
# Download + verify + convert raw klines and funding, one array element per symbol of
# configs/universe.yaml (plan.md §2.3-2.4). Zips stay in memory; already finished
# symbols are skipped, so the same array can simply be resubmitted after failures.
#     bsub < jobs/download.sh                              # all 371 symbols
#     bsub -J "cfm_dl[1,2,7,339,347]" < jobs/download.sh   # selected universe ranks
#BSUB -q hpc
#BSUB -J "cfm_dl[1-371]%16"
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

set -euo pipefail
source jobs/common.sh
# --end: last complete month, fixed so reruns produce the same dataset.
python -m cfm.data.download --index "$LSB_JOBINDEX" --end 2026-09 --workers 4
