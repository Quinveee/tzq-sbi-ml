#!/bin/bash

# To be run on the working node
# Sets up environment, extracts tarball with needed files and calls 
# the worker

set -euo pipefail

# The $ENV variable contains the bin/ directory of the Python interpreter we want to use
# in the Working Node. The parent of bin/ is the conda env root.
setup_env() {
    ENV_ROOT="$(dirname "$ENV")"
    # Prepend the conda env's lib/ so CUDA runtime libraries (libcudart, etc.) are found
    export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$ENV:$PATH"
}

setup_env


TARBALL="$1"
tar -xzf "$TARBALL"

python -m workers.worker *.yaml
