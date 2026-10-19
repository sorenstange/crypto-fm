#!/bin/bash
# M-1 checkpoint-and-resume test. Submit as a chain of jobs sharing one job name
# (= one run directory = one W&B run):
#     jobs/submit_chain.sh 2 jobs/resume_test.sh cfm_resume_$(date +%m%d%H%M)
# 600 steps x 0.5 s sleep outlast the first job: LSF sends SIGUSR2 2 min before the
# 6 min run limit (-wa/-wt), the job checkpoints and exits 0, the next job resumes,
# finishes and verifies against an uninterrupted rerun.
#BSUB -q gpua100
#BSUB -J cfm_resume_test
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:06
#BSUB -wa USR2
#BSUB -wt 2
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
source jobs/common.sh
cfm_run python -m cfm.smoke.resume_test --run-dir "$RUNS_DIR/$LSB_JOBNAME" \
    --total-steps 600 --step-sleep 0.5 --verify
