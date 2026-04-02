"""
Datasets for the 'particles' approach
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from .schemas import ParametrizedParticlesEvent, ParticlesEvent


def _reshape_particles(
    x: np.ndarray, met: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Reshape numpy array in which each event contains
    (max_num_particles * 4) features, optionally extracting trailing
    MET features (pt, phi).

    :param x: Flat array of shape (samples, n_particles*4 [+ 2])
    :type x: np.ndarray
    :param met: Whether the data contains trailing MET features
    :type met: bool
    :return: Tuple of (particles, met_features) where met_features is
        None when met=False or shape (samples, 2) when met=True
    :rtype: tuple[np.ndarray, Optional[np.ndarray]]
    """
    samples, features = x.shape

    # Some datasets may include trailing MET features (pt, phi).
    # These should not be interpreted as particle four-momenta.
    met_features = None
    if features % 4 == 2:
        if met:
            met_features = x[:, -2:].astype(np.float32)
        x = x[:, :-2]
        features -= 2

    max_particles = features // 4

    # Each particle is represnted by its 4-momenta
    assert not features % 4, "invalid number of features"

    # (samples, max_particles, 4)
    return x.reshape(samples, max_particles, 4).astype(np.float32), met_features


def _sample_lengths(x: np.ndarray):
    """
    Get event lenghts (number of particles per event)

    :param x: Numpy array of events
    :type x: np.ndarray
    """
    # This will mask null values for missing particles in an event
    mask = np.abs(x).sum(axis=-1) > 0  # (samples, max_particles)
    return mask.sum(axis=-1)


class ParticlesDataset(Dataset):
    """
    Dataset container for the non-parametrized 'particles' approach
    """

    # Used when preprocessed features are enabled but no explicit array is
    # passed (e.g. some evaluation loaders built from sampled events).
    default_preprocessed_dim: int = 0

    def __init__(
        self,
        *,
        x: np.ndarray,
        score: Optional[np.ndarray] = None,
        preprocessed: Optional[np.ndarray] = None,
        preprocessed_dim: Optional[int] = None,
        met: bool = False,
        **kwds,
    ) -> None:
        # allow for unlabelled data
        score = score if score is not None else np.zeros((len(x), 1))
        assert len(x) == len(score), f"x and y differ in length"

        if preprocessed_dim is None:
            preprocessed_dim = self.default_preprocessed_dim

        if preprocessed is None:
            preprocessed = np.zeros((len(x), preprocessed_dim), dtype=np.float32)
        else:
            preprocessed = np.asarray(preprocessed, dtype=np.float32)
            if preprocessed.ndim == 1:
                preprocessed = preprocessed[:, None]

        assert len(x) == len(preprocessed), "x and preprocessed differ in length"

        self._x, met_features = _reshape_particles(x, met=met)
        self._lengths = _sample_lengths(self._x)
        self._score = score  # (nsamples, ??) depends on dim(theta)
        self._preprocessed = preprocessed.astype(np.float32)
        self._met = (
            met_features
            if met_features is not None
            else np.zeros((len(x), 0), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index) -> ParticlesEvent:
        """
        `self._x[index]` contains all particles in event #index,
        including `null`s for non-available particles
        `self._lenghts[index]` contains the lenght of the event,
        this is, the number of non-null particles

        ::note:: For an event 'i', we assume the first `lenghts[i]` particles
        are available (particles[i][:length[i]]) and the rest are `null`s

        :param index: Description
        :return: Container for single event data
        :rtype: ParticlesEvent
        """
        return ParticlesEvent(
            fourmomenta=self._x[index],
            length=int(self._lengths[index]),
            score=self._score[index],
            preprocessed=self._preprocessed[index],
            met=self._met[index],
        )


class ParametrizedParticleDataset(ParticlesDataset):
    """
    Dataset container for the parametrized 'particles' approach
    """

    def __init__(
        self,
        *,
        x: np.ndarray,
        theta: np.ndarray,
        score: Optional[np.ndarray] = None,
        ratio: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        preprocessed: Optional[np.ndarray] = None,
        preprocessed_dim: Optional[int] = None,
        met: bool = False,
        **kwds,
    ):
        super().__init__(
            x=x,
            score=score,
            preprocessed=preprocessed,
            preprocessed_dim=preprocessed_dim,
            met=met,
        )
        # allow for unllabeled data
        ratio = ratio if ratio is not None else np.zeros((len(x), 1))
        labels = labels if labels is not None else np.zeros((len(x), 1))

        assert len(x) == len(theta), "x and theta differ in length"

        self._thetas = theta
        self._ratios = ratio
        self._labels = labels

    def __getitem__(self, index) -> ParametrizedParticlesEvent:
        """
        `self._x[index]` contains all particles in event #index,
        including `null`s for non-available particles
        `self._lenghts[index]` contains the lenght of the event,
        this is, the number of non-null particles

        ::note:: For an event 'i', we assume the first `lenghts[i]` particles
        are available (particles[i][:length[i]]) and the rest are `null`s

        :param index: Description
        :return: Container for single event data
        :rtype: ParametrizedParticlesEvent
        """
        return ParametrizedParticlesEvent(
            fourmomenta=self._x[index],
            theta=self._thetas[index],
            length=int(self._lengths[index]),
            score=self._score[index],
            preprocessed=self._preprocessed[index],
            met=self._met[index],
            ratio=self._ratios[index],
            label=self._labels[index],
        )
