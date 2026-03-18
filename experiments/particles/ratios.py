"""Experiment class for likelihood ratio regression in the 'particles' case"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ..base.base_experiment_ratios import BaseExperimentRatios
from .collate import parametrized_collate_particles_fn
from .datasets import ParametrizedParticleDataset

if TYPE_CHECKING:
    from .schemas import ParametrizedParticleBatch


class ExperimentRatiosParticles(BaseExperimentRatios):
    dataset_cls = ParametrizedParticleDataset
    collate_fn = staticmethod(parametrized_collate_particles_fn)

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        lloca_cfg = self.cfg.model.get("LLoCa", {})
        self.collate_fn = partial(
            parametrized_collate_particles_fn,
            lloca=lloca_cfg.get("active", False),
        )

    def _preds(self, batch: ParametrizedParticleBatch):
        """
        Return model predictions

        :param self: Description
        :param batch: Description
        :type batch: ParametrizedParticleBatch
        """
        # To later compute score based on regressed log-likelihood ratio
        batch.theta.requires_grad_(self.loss_fn.REQUIRES_SCORE)
        embedding_kwargs = {
            "theta": batch.theta,
            "ptr": batch.ptr,
            "mode": self.cfg.model.get("mode", "channels"),
        }

        # Regress log-likelihood ratio
        log_ratio_pred = self.model(
            batch.particles,
            batch.ptr,
            force_math=self.loss_fn.REQUIRES_SCORE,
            embedding_kwargs=embedding_kwargs,
        )
        return self.pack_output(
            theta=batch.theta,
            log_ratio_pred=log_ratio_pred,
            score=batch.score,
            ratio=batch.ratio,
            label=batch.label,
        )
