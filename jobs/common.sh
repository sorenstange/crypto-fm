# Shared setup for every LSF job. Submit jobs from the project root and source it:
#     set -euo pipefail
#     source jobs/common.sh
# Loads modules, activates the venv, defines paths, picks the W&B mode and prints
# node/GPU info. Also safe to source on the login node (e.g. for `wandb sync`).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

# --- storage --------------------------------------------------------------------------
# DATA_ROOT holds raw/processed data, runs (checkpoints) and W&B files. It lives in
# the home quota (30 GB, shared with the venv), so delete intermediates and caches
# that are no longer needed. Check usage with `getquota_zhome.sh`.
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export RUNS_DIR="$DATA_ROOT/runs"
mkdir -p "$DATA_ROOT" "$RUNS_DIR" "$DATA_ROOT/wandb" "$DATA_ROOT/cache"

# --- Python environment ---------------------------------------------------------------
# shellcheck source=env/modules.sh
source "$PROJECT_ROOT/env/modules.sh"
cfm_load_modules
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[common] ERROR: no venv at $VENV_DIR -- run env/setup_env.sh first" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# One thread per requested core (LSF sets LSB_DJOB_NUMPROC inside jobs).
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-${OMP_NUM_THREADS:-1}}"
# Compiler caches are node-local: fast, and keeps file locks off the network FS.
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/cfm-$USER/triton"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/cfm-$USER/inductor"

# --- network --------------------------------------------------------------------------
# cfm_can_reach URL: succeeds if an HTTP(S) response arrives (any status code).
cfm_can_reach() {
    curl -sS -o /dev/null --max-time "${2:-8}" "$1" 2>/dev/null
}

# cfm_net_report: probe the endpoints the project needs and print a table.
cfm_net_report() {
    local url out
    printf '%-6s %-7s %s\n' "HTTP" "time_s" "url"
    for url in \
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip.CHECKSUM" \
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/futures/um/monthly/klines/&delimiter=/" \
        "https://fapi.binance.com/fapi/v1/ping" \
        "https://api.wandb.ai/healthz" \
        "https://pypi.org/simple/pip/" \
        "https://download.pytorch.org/whl/"; do
        # HTTP code 000 = no connection.
        out="$(curl -sS -o /dev/null --max-time 10 -w '%{http_code} %{time_total}' "$url" 2>/dev/null)" || true
        out="${out:-000 -}"
        printf '%-6s %-7s %s\n' "${out%% *}" "${out##* }" "$url"
    done
    env | grep -i '_proxy=' || echo "(no proxy variables set)"
}

# --- Weights & Biases -----------------------------------------------------------------
# WANDB_ENTITY is left unset: W&B uses the default entity of the logged-in account.
# The API key lives in ~/.netrc (`wandb login`); never put it in scripts or configs.
export WANDB_PROJECT="${WANDB_PROJECT:-crypto-fm}"
export WANDB_DIR="$DATA_ROOT/wandb"
export WANDB_CACHE_DIR="$DATA_ROOT/cache/wandb"
export WANDB_DATA_DIR="$DATA_ROOT/cache/wandb-data"
export WANDB_ARTIFACT_DIR="$DATA_ROOT/cache/wandb-artifacts"
export WANDB_CONFIG_DIR="$DATA_ROOT/cache/wandb-config"
if [[ -z "${WANDB_MODE:-}" ]]; then
    if cfm_can_reach "https://api.wandb.ai/healthz"; then
        export WANDB_MODE=online
    else
        export WANDB_MODE=offline   # sync later from the login node: scripts/wandb_sync.sh
    fi
fi

# --- signal forwarding ----------------------------------------------------------------
# cfm_run CMD...: run CMD, forwarding SIGUSR2 (LSF walltime warning, `#BSUB -wa USR2`)
# and SIGTERM to it, so Python can checkpoint and exit cleanly. Returns CMD's status.
cfm_run() {
    "$@" &
    local pid=$! rc=0
    # shellcheck disable=SC2064
    trap "echo '[common] SIGUSR2 -> $pid'; kill -USR2 $pid 2>/dev/null || true" USR2
    # shellcheck disable=SC2064
    trap "echo '[common] SIGTERM -> $pid'; kill -TERM $pid 2>/dev/null || true" TERM
    while true; do
        wait "$pid" && rc=0 || rc=$?
        # A trapped signal interrupts `wait` (rc > 128) while the child keeps running.
        kill -0 "$pid" 2>/dev/null || break
    done
    trap - USR2 TERM
    return "$rc"
}

# --- job info -------------------------------------------------------------------------
cfm_print_info() {
    local commit dirty=""
    commit="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    git -C "$PROJECT_ROOT" diff --quiet HEAD 2>/dev/null || dirty="+dirty"
    echo "================ cfm job ================"
    echo "date        $(date -Is)"
    echo "host        $(hostname)"
    echo "job         ${LSB_JOBID:-none} name=${LSB_JOBNAME:-none} index=${LSB_JOBINDEX:-0} queue=${LSB_QUEUE:-none}"
    echo "cores       ${LSB_DJOB_NUMPROC:-n/a} (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
    echo "git         $commit$dirty"
    echo "python      $(python -V 2>&1) at $(command -v python)"
    echo "DATA_ROOT   $DATA_ROOT"
    echo "WANDB_MODE  $WANDB_MODE"
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "gpu         $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
    fi
    echo "========================================="
}

if [[ -n "${LSB_JOBID:-}" ]]; then
    cfm_print_info
fi
