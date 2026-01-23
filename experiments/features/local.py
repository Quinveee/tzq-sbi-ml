"""
Local (score regression) experiment class for the 'features' case
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base.base_experiment_local import BaseExperimentLocal
from .collate import collate_features_fn
from .datasets import FeaturesDataset

if TYPE_CHECKING:
    from .schemas import FeaturesBatch


class ExperimentLocalFeatures(BaseExperimentLocal):
    dataset_cls = FeaturesDataset
    collate_fn = staticmethod(collate_features_fn)

    def _preds(self, batch: FeaturesBatch):
        score_pred = self.model(batch.x)
        return self.pack_output(score_pred, batch.score)
