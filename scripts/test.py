import torch
import os

print("Torch:", torch.__version__)
print("CUDA compiled:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))