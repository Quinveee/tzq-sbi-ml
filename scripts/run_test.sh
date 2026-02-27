#!/bin/bash

# To be run on the working node
# Sets up environment, extracts tarball with needed files and calls 
# the worker

set -euo pipefail

# Path to your Python interpreter bin/
ENV=/data/atlas/users/qvanenge/miniforge/envs/ml/bin

# The $ENV variable contains the bin/ directory of the Python interpreter we want to use
# in the Working Node. The parent of bin/ is the conda env root.
setup_env() {
    ENV_ROOT="$(dirname "$ENV")"
    # Prepend the conda env's lib/ so CUDA runtime libraries (libcudart, etc.) are found
    export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$ENV:$PATH"
}

setup_env

# Optional: debug check for Torch & CUDA
python - <<'EOF'
import torch
import os

print("Torch:", torch.__version__)
print("CUDA compiled:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA not available")
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
EOF

