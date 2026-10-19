#!/bin/bash
# M-1 GPU smoke test: GPU info, tiny bf16 fwd/bwd, flash SDPA, torch.compile, network probe.
#     bsub < jobs/test_gpu.sh
#BSUB -q gpua100
#BSUB -J cfm_test_gpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:20
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh

nvidia-smi
echo "--- network from $(hostname) ---"
cfm_net_report
echo "--- DATA_ROOT writable? ---"
touch "$DATA_ROOT/.write_test_$LSB_JOBID" && rm "$DATA_ROOT/.write_test_$LSB_JOBID" && echo "yes: $DATA_ROOT"
echo "--- gpu check ---"
python -m cfm.smoke.gpu_check
