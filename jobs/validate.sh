#!/bin/bash
# Validate the raw store: merged manifest, per-symbol report, REST spot-checks.
#     bsub < jobs/validate.sh
#BSUB -q hpc
#BSUB -J cfm_validate
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
python -m cfm.data.validate --workers 4
