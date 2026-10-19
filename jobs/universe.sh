#!/bin/bash
# Universe selection (plan.md §2.2): list + classify all USD-M symbols, fetch their 1d
# klines, rank by median daily quote volume, write configs/universe.yaml (no cut-off yet;
# the candle budget is chosen after the pilot download).
#     bsub < jobs/universe.sh
#BSUB -q hpc
#BSUB -J cfm_universe
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 2:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
python -m cfm.data.universe scan --workers 16
python -m cfm.data.universe select
