"""
Datasets for the features experiments
"""

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from .schemas import FeaturesEvent, ParametrizedFeaturesEvent


class FeaturesDataset(Dataset):
    """
    Non-parametrized features dataset
    """

    def __init__(self, x: np.ndarray, score: Optional[np.ndarray] = None, **kwargs):
        score = score if score is not None else np.zeros((len(x), 1))
        assert len(score) == len(x)
        self._x = x
        self._score = score

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index):
        return FeaturesEvent(x=self._x[index], score=self._score[index])


class ParametrizedFeaturesDataset(FeaturesDataset):
    """
    Parametrized features dataset
    """

    def __init__(self, x, theta, score=None, ratio=None, labels=None, **kwargs):
        super().__init__(x, score, **kwargs)
        ratio = ratio if ratio is not None else np.zeros((len(x), 1))
        labels = labels if labels is not None else np.zeros((len(x), 1))

        assert len(x) == len(theta), "x and theta differ in length"

        self._thetas = theta
        self._ratios = ratio
        self._labels = labels

    def __getitem__(self, index):
        return ParametrizedFeaturesEvent(
            x=self._x[index],
            theta=self._thetas[index],
            score=self._score[index],
            ratio=self._ratios[index],
            label=self._labels[index],
        )
