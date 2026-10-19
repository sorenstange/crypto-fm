#!/bin/bash
# Run the pytest suite on a CPU node.
#     bsub < jobs/tests.sh
#BSUB -q hpc
#BSUB -J cfm_tests
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:30
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
export WANDB_MODE=disabled
python -m pytest -v "$@"
