"""
Base model wrapper class
"""

from typing import Optional

import torch.nn as nn


class BaseWrapper(nn.Module):
    def __init__(self, net, key: Optional[str] = None) -> None:
        super().__init__()
        self.net = net
        self._key = key

    def __str__(self):
        return self._key if self._key else ""
