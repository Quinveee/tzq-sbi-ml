"""
Local (score regression) experiment class for the 'particles' case
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ..base.base_experiment_local import BaseExperimentLocal
from .collate import collate_particles_fn
from .datasets import ParticlesDataset

if TYPE_CHECKING:
    from .schemas import ParticleBatch


class ExperimentLocalParticles(BaseExperimentLocal):
    dataset_cls = ParticlesDataset
    collate_fn = staticmethod(collate_particles_fn)

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        lloca_cfg = self.cfg.model.get("LLoCa", {})
        self.collate_fn = partial(
            collate_particles_fn,
            lloca=lloca_cfg.get("active", False),
        )

    def _preds(self, batch: ParticleBatch):
        embedding_kwargs = {"theta_dim": self.cfg.dataset.theta_dim}
        score_pred = self.model(
            batch.particles, batch.ptr, embedding_kwargs=embedding_kwargs
        )
        return self.pack_output(score_pred, batch.score)
