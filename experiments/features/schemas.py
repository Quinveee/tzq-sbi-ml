"""
Useful containers specific to the 'features' case
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..utils import to_fields


@dataclass(slots=True)
class FeaturesEvent:
    x: np.ndarray
    score: np.ndarray


@dataclass(slots=True)
class ParametrizedFeaturesEvent(FeaturesEvent):
    theta: np.ndarray
    ratio: np.ndarray
    label: np.ndarray


@dataclass(slots=True)
class FeaturesBatch:
    x: torch.Tensor
    score: torch.Tensor

    def to_(self, **kwargs):
        to_fields(self, **kwargs)


@dataclass(slots=True)
class ParametrizedFeaturesBatch(FeaturesBatch):
    theta: torch.Tensor
    ratio: torch.Tensor
    label: torch.Tensor
