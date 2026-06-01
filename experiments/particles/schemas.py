"""
Useful containers for the 'particles' approach
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..utils import to_fields


@dataclass(slots=True)
class ParticlesEvent:
    fourmomenta: np.ndarray
    length: int
    score: np.ndarray
    preprocessed: np.ndarray
    met: np.ndarray


@dataclass(slots=True)
class ParametrizedParticlesEvent(ParticlesEvent):
    theta: np.ndarray
    ratio: np.ndarray
    label: np.ndarray


@dataclass(slots=True)
class ParticleBatch:
    # `particles` is padded (B, L_max, 4). `valid_mask` (B, L_max) marks real
    # tokens; False entries are padding. `ptr` is CSR-style (B+1,) and kept as
    # a derived/compat field for downstream code that still expects the
    # per-event boundary layout.
    particles: torch.Tensor
    ptr: torch.Tensor
    valid_mask: torch.Tensor
    score: torch.Tensor
    preprocessed: torch.Tensor
    met: torch.Tensor

    def to_(self, **kwargs):
        to_fields(self, **kwargs)


@dataclass(slots=True)
class ParametrizedParticleBatch(ParticleBatch):
    theta: torch.Tensor
    ratio: torch.Tensor
    label: torch.Tensor
