"""
Local (score regression) experiment class for the 'particles' case
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import numpy as np

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
        self._use_preprocessed = bool(self.cfg.data.get("preprocessed", False))
        self._preprocessed_dim = (
            int(self.cfg.data.get("n_preprocessed_features", 0))
            if self._use_preprocessed
            else 0
        )

        # Some evaluation loaders are built from sampled events and do not pass
        # preprocessed arrays explicitly. Keep the expected dimensionality there.
        self.dataset_cls.default_preprocessed_dim = self._preprocessed_dim

        self.collate_fn = partial(
            collate_particles_fn,
            lloca=lloca_cfg.get("active", False),
        )

    def _load_raw_data(self, source: str):
        source = Path(source)
        max_samples = self.cfg.train.get("clamp_samples", None)

        x_train = np.load(source / "x_train_score.npy")[:max_samples]
        x_test = np.load(source / "x_test.npy")

        score_train = np.load(source / "t_xz_train_score.npy")[:max_samples]
        score_test = np.load(source / "t_xz_test_score.npy")

        preprocessed_train = preprocessed_test = None
        if self._use_preprocessed:
            preprocessed_train = np.load(source / "x_train_score_hlvl.npy")[:max_samples]
            preprocessed_test = np.load(source / "x_test_hlvl.npy")
            if preprocessed_train.shape[-1] != self._preprocessed_dim:
                raise ValueError(
                    "x_train_score_hlvl.npy has "
                    f"{preprocessed_train.shape[-1]} features, expected {self._preprocessed_dim}"
                )
            if preprocessed_test.shape[-1] != self._preprocessed_dim:
                raise ValueError(
                    "x_test_hlvl.npy has "
                    f"{preprocessed_test.shape[-1]} features, expected {self._preprocessed_dim}"
                )

        return SimpleNamespace(
            x_train=x_train,
            score_train=score_train,
            x_test=x_test,
            score_test=score_test,
            preprocessed_train=preprocessed_train,
            preprocessed_test=preprocessed_test,
        )

    def _load_dataset(self, raw, mode: Literal["train", "test"] = "train"):
        if mode == "train":
            return self.dataset_cls(
                x=raw.x_train,
                score=raw.score_train,
                preprocessed=raw.preprocessed_train,
                preprocessed_dim=self._preprocessed_dim,
            )
        if mode == "test":
            return self.dataset_cls(
                x=raw.x_test,
                score=raw.score_test,
                preprocessed=raw.preprocessed_test,
                preprocessed_dim=self._preprocessed_dim,
            )
        raise ValueError(f"Invalid mode {mode}")

    def _preds(self, batch: ParticleBatch):
        embedding_kwargs = {
            "theta_dim": self.cfg.dataset.theta_dim,
            "ptr": batch.ptr,
        }
        if self._use_preprocessed:
            embedding_kwargs["preprocessed"] = batch.preprocessed

        if self.cfg.model.key == "lgatr":
            score_pred = self.model(
                batch.particles,
                batch.ptr,
                scalars=batch.preprocessed if self._use_preprocessed else None,
                embedding_kwargs=embedding_kwargs,
            )
        else:
            score_pred = self.model(
                batch.particles,
                batch.ptr,
                embedding_kwargs=embedding_kwargs,
            )

        return self.pack_output(score_pred, batch.score)
