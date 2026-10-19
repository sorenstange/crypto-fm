#!/bin/bash
# Submit N copies of a job script, each waiting for the previous one to finish.
# All copies share one job name, so they share one run directory ($RUNS_DIR/<name>).
#     jobs/submit_chain.sh <n_jobs> <job_script> [job_name]
# DEP=ended also continues after a failed/killed job (resume picks up the last
# checkpoint); the default DEP=done stops the chain on failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

n="$1"; script="$2"; name="${3:-}"
dep="${DEP:-done}"
prev=""
for ((i = 1; i <= n; i++)); do
    args=()
    [[ -n "$name" ]] && args+=(-J "$name")
    [[ -n "$prev" ]] && args+=(-w "$dep($prev)")
    out="$(bsub "${args[@]}" < "$script")"
    echo "$out"
    prev="$(sed -nE 's/^Job <([0-9]+)>.*/\1/p' <<<"$out")"
    [[ -n "$prev" ]] || { echo "submit_chain: could not parse job id" >&2; exit 1; }
done
