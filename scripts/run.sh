#!/bin/bash

# To be run on the working node
# Sets up environment, extracts tarball with needed files and calls 
# the worker

set -euo pipefail

# The $ENV variable contains the root folder of the Python interpreter we want to use
# in the Working Node
setup_env() {
    PATH="$ENV:$PATH"
}

setup_env


TARBALL="$1"
tar -xzf "$TARBALL"

python -m workers.worker *.yaml

