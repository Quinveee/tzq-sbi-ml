#!/bin/bash

# To be run on Snellius compute nodes.
# Uses shared filesystem paths directly and executes the worker with a config yaml.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_yaml>" >&2
    exit 1
fi

CFG_PATH="$1"
if [[ "${CFG_PATH}" != /* ]]; then
    CFG_PATH="$(realpath "${CFG_PATH}")"
fi

if [[ ! -f "${CFG_PATH}" ]]; then
    echo "Configuration file not found: ${CFG_PATH}" >&2
    exit 1
fi

# Resolve project root from config location first.
# The config is written to <repo>/inputs/snellius_*.yaml by the launcher,
# while the Slurm job script itself may run from a spooled temp path.
PROJECT_ROOT="$(cd "$(dirname "${CFG_PATH}")/.." && pwd)"

# Fallback to submit directory when available.
if [[ ! -d "${PROJECT_ROOT}/workers" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}/workers" ]]; then
        PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
    fi
fi

if [[ ! -d "${PROJECT_ROOT}/workers" ]]; then
    echo "Could not locate project root with workers/ directory." >&2
    echo "Resolved PROJECT_ROOT=${PROJECT_ROOT}" >&2
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Optional: preserve the same ENV convention used by HTCondor launcher.
if [[ -n "${ENV:-}" ]]; then
    ENV_ROOT="$(dirname "${ENV}")"
    export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
    export PATH="${ENV}:${PATH}"
fi

python -m workers.worker "${CFG_PATH}"
