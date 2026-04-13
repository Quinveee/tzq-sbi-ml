"""Experiment class for likelihood ratio regression in the 'features' case"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..base.base_experiment_ratios import BaseExperimentRatios
from .collate import parametrized_collate_features_fn
from .datasets import ParametrizedFeaturesDataset

if TYPE_CHECKING:
    from .schemas import ParametrizedFeaturesBatch


class ExperimentRatiosFeatures(BaseExperimentRatios):
    dataset_cls = ParametrizedFeaturesDataset
    collate_fn = staticmethod(parametrized_collate_features_fn)

    @torch.enable_grad()
    def _preds(self, batch: ParametrizedFeaturesBatch):
        batch.theta.requires_grad_(self.loss_fn.REQUIRES_SCORE)
        log_ratio_pred = self.model(batch.x, theta=batch.theta)
        return self.pack_output(
            theta=batch.theta,
            log_ratio_pred=log_ratio_pred,
            score=batch.score,
            ratio=batch.ratio,
            label=batch.label,
        )
