#!/bin/bash
# One-time environment setup. Light (downloads + installs only), so it may run on the
# login node:
#     bash env/setup_env.sh
# Creates the venv if missing (default: <project>/.venv) and installs the pinned
# requirements. Re-running is safe; it only installs what is missing or differs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

# shellcheck source=env/modules.sh
source "$PROJECT_ROOT/env/modules.sh"
cfm_load_modules

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[setup] creating venv at $VENV_DIR with $(python3 -V)"
    python3 -m venv "$VENV_DIR"
else
    echo "[setup] using existing venv at $VENV_DIR"
fi

module_py="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
venv_py="$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$module_py" != "$venv_py" ]]; then
    echo "[setup] ERROR: venv uses Python $venv_py but module $CFM_PYTHON_MODULE is $module_py" >&2
    exit 1
fi

# --no-cache-dir: the pip cache would land in the (small) home quota.
"$VENV_DIR/bin/python" -m pip install --no-cache-dir --upgrade pip
"$VENV_DIR/bin/python" -m pip install --no-cache-dir -r "$PROJECT_ROOT/env/requirements.txt"
"$VENV_DIR/bin/python" -m pip freeze --exclude-editable > "$PROJECT_ROOT/env/requirements.lock.txt"

"$VENV_DIR/bin/python" - <<'EOF'
import sys, torch, numpy, polars, pandas, wandb
print(f"[setup] python {sys.version.split()[0]}  torch {torch.__version__} (CUDA {torch.version.cuda})"
      f"  numpy {numpy.__version__}  polars {polars.__version__}  pandas {pandas.__version__}"
      f"  wandb {wandb.__version__}")
EOF

echo "[setup] venv size: $(du -sh "$VENV_DIR" | cut -f1)"
echo "[setup] done. If not done yet, log in to W&B once on the login node:"
echo "        $VENV_DIR/bin/wandb login"
