#!/bin/bash
# Phase 2: fit the train-only statistics (seasonality, scalers), then build every symbol's
# features in RAM and summarise them (NaN/inf, per-split moments, timing).
#     bsub < jobs/features.sh
#BSUB -q hpc
#BSUB -J cfm_features
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
python -m cfm.data.features fit-stats --workers 8
python -m cfm.data.features check --workers 8
