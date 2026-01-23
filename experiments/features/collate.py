"""
Collate functions for the features experiments
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Iterable

from torch.utils.data import default_collate

from .schemas import FeaturesBatch, ParametrizedFeaturesBatch

if TYPE_CHECKING:
    from .schemas import FeaturesEvent, ParametrizedFeaturesEvent


def collate_features_fn(batch: Iterable[FeaturesEvent]) -> FeaturesBatch:
    """
    Turn event dataclasses into dicts and create a container
    for the batch data

    :param batch: Iterable over event samples
    :type batch: Iterable[FeaturesEvent]
    :return: Container with batched tensors
    :rtype: FeaturesBatch
    """
    return FeaturesBatch(**default_collate([asdict(e) for e in batch]))


def parametrized_collate_features_fn(
    batch: Iterable[ParametrizedFeaturesEvent],
) -> ParametrizedFeaturesBatch:
    """
    Turn event dataclasses into dicts and create a container
    for the batch data

    :param batch: Iterable over event samples
    :type batch: Iterable[ParametrizedFeaturesEvent]
    :return: Container with batched tensors
    :rtype: ParametrizedFeaturesBatch
    """
    return ParametrizedFeaturesBatch(**default_collate([asdict(e) for e in batch]))
