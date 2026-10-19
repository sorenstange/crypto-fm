#!/bin/bash
# Offline fallback: upload W&B runs written with WANDB_MODE=offline. Run on the login
# node (has internet), periodically or after each job:
#     bash scripts/wandb_sync.sh
# Already-synced runs are skipped (wandb marks them with a .synced file).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../jobs/common.sh"
shopt -s nullglob
runs=("$WANDB_DIR"/wandb/offline-run-*)
(( ${#runs[@]} )) || { echo "no offline runs in $WANDB_DIR/wandb"; exit 0; }
wandb sync --include-offline --no-include-synced "${runs[@]}"
