"""
Useful data containers
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(slots=True)
class RawData:
    x_train: np.ndarray
    score_train: np.ndarray
    x_test: np.ndarray
    score_test: np.ndarray


@dataclass(slots=True)
class ParametrizedRawData(RawData):
    theta_train: np.ndarray
    ratio_train: np.ndarray
    labels_train: np.ndarray
    theta_test: np.ndarray
    ratio_test: np.ndarray
    labels_test: np.ndarray


@dataclass(slots=True)
class Limits:
    param_names: List[str]
    grid: np.ndarray
    p_values: np.ndarray
    mle: int
    llr_kin: np.ndarray
    rate_ll: np.ndarray

    @property
    def llr(self) -> np.ndarray:
        llr = self.llr_kin + self.rate_ll
        return llr - llr.max()

    @property
    def ranges(self) -> List[Tuple[float, float]]:
        mins = np.min(self.grid, axis=0)
        maxs = np.max(self.grid, axis=0)
        return list(zip(mins, maxs))

    @property
    def resolutions(self) -> List[int]:
        # NOTE: I assume resolutions are the same
        # across all dimensions
        n, d = self.grid.shape
        res = np.pow(n, (1 / d)).round().astype(int)
        return [res for _ in range(d)]


@dataclass(slots=True)
class Losses:
    val: np.ndarray
    train: np.ndarray

    def __post_init__(self):
        self.val = np.asarray(self.val).reshape(-1)
        self.train = np.asarray(self.train).reshape(-1)


@dataclass
class Chekcpoints:
    state_dict: Optional[Mapping[str, Any]] = None
    losses: Optional[Losses] = None
    limits: Optional[Limits] = None

    def __post_init__(self):
        if isinstance(self.losses, dict):
            self.losses = Losses(**self.losses)

        if isinstance(self.limits, dict):
            self.limits = Limits(**self.limits)


@dataclass(slots=True)
class PredictionOutput:
    score: torch.Tensor


@dataclass(slots=True)
class ParametrizedPredictionOutput(PredictionOutput):
    log_ratio: torch.Tensor


@dataclass(slots=True)
class TargetOutput:
    score: torch.Tensor


@dataclass(slots=True)
class ParametrizedTargetOutput(TargetOutput):
    ratio: torch.Tensor
    label: torch.Tensor


@dataclass(slots=True)
class ModelOutput:
    pred: Union[PredictionOutput, ParametrizedPredictionOutput]
    target: Union[TargetOutput, ParametrizedTargetOutput]
