"""
Local (score regression) experiment class for the 'particles' case
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base.base_experiment_local import BaseExperimentLocal
from .collate import collate_particles_fn
from .datasets import ParticlesDataset

if TYPE_CHECKING:
    from .schemas import ParticleBatch


class ExperimentLocalParticles(BaseExperimentLocal):
    dataset_cls = ParticlesDataset
    collate_fn = staticmethod(collate_particles_fn)

    def _preds(self, batch: ParticleBatch):
        embedding_kwargs = {"theta_dim": self.cfg.dataset.theta_dim}
        score_pred = self.model(
            batch.particles, batch.ptr, embedding_kwargs=embedding_kwargs
        )
        return self.pack_output(score_pred, batch.score)
