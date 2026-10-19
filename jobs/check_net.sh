#!/bin/bash
# Does a CPU compute node have outbound internet? (GPU nodes: see jobs/test_gpu.sh.)
#     bsub < jobs/check_net.sh
#BSUB -q hpc
#BSUB -J cfm_check_net
#BSUB -n 1
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 0:10
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh

echo "--- network from $(hostname) ---"
cfm_net_report
echo "--- DATA_ROOT writable? ---"
touch "$DATA_ROOT/.write_test_$LSB_JOBID" && rm "$DATA_ROOT/.write_test_$LSB_JOBID" && echo "yes: $DATA_ROOT"
