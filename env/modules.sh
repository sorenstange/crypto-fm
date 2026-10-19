# Module set shared by env/setup_env.sh and jobs/common.sh (sourced, not executed),
# so the venv is built and used with the same interpreter.
#
# No CUDA module is loaded on purpose: the torch wheel ships its own CUDA runtime,
# and a cuda module on LD_LIBRARY_PATH could shadow those libraries.

CFM_PYTHON_MODULE="${CFM_PYTHON_MODULE:-python3/3.11.9}"

cfm_load_modules() {
    # Batch shells may not have the `module` function defined.
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /appl/Modules/4.8.0/init/bash
    fi
    # modulecmd output is not `set -u` safe.
    local had_u=0
    [[ $- == *u* ]] && had_u=1
    set +u
    module load "$CFM_PYTHON_MODULE"
    local rc=$?
    (( had_u )) && set -u
    # The module prepends its own site-packages (setuptools, pip, filelock, ...) to
    # PYTHONPATH, which would shadow the venv's packages. The venv finds the stdlib
    # through pyvenv.cfg, so both variables can go.
    unset PYTHONPATH PYTHONHOME
    return $rc
}
