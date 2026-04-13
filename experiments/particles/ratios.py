"""Experiment class for likelihood ratio regression in the 'particles' case"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

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
        self._use_preprocessed = bool(self.cfg.data.get("preprocessed", False))
        self._use_met = bool(self.cfg.data.get("met", False)) and not self._use_preprocessed
        self._preprocessed_dim = (
            int(self.cfg.data.get("n_preprocessed_features", 0))
            if self._use_preprocessed
            else 0
        )

        # Some evaluation loaders are built from sampled events and do not pass
        # preprocessed arrays explicitly. Keep the expected dimensionality there.
        self.dataset_cls.default_preprocessed_dim = self._preprocessed_dim

        self.collate_fn = partial(
            parametrized_collate_particles_fn,
            lloca=lloca_cfg.get("active", False),
        )

    def _load_raw_data(self, source: str):
        source = Path(source)
        max_samples = self.cfg.train.get("clamp_samples", None)

        x_train = np.load(source / "x_train_ratio.npy")[:max_samples]
        theta_train = np.load(source / "theta0_train_ratio.npy")[:max_samples]

        ratio_train = np.load(source / "r_xz_train_ratio.npy")[:max_samples]
        score_train = np.load(source / "t_xz_train_ratio.npy")[:max_samples]
        labels_train = np.load(source / "y_train_ratio.npy")[:max_samples]

        x_test = np.load(source / "x_test.npy")
        theta_test = np.load(source / "theta_test.npy")

        ratio_test = np.load(source / "r_xz_test_ratio.npy")
        score_test = np.load(source / "t_xz_test_ratio.npy")
        labels_test = np.load(source / "y_test_ratio.npy")

        preprocessed_train = preprocessed_test = None
        if self._use_preprocessed:
            preprocessed_train = np.load(source / "x_train_ratio_hlvl.npy")[:max_samples]
            preprocessed_test = np.load(source / "x_test_hlvl.npy")
            if preprocessed_train.shape[-1] != self._preprocessed_dim:
                raise ValueError(
                    "x_train_ratio_hlvl.npy has "
                    f"{preprocessed_train.shape[-1]} features, expected {self._preprocessed_dim}"
                )
            if preprocessed_test.shape[-1] != self._preprocessed_dim:
                raise ValueError(
                    "x_test_hlvl.npy has "
                    f"{preprocessed_test.shape[-1]} features, expected {self._preprocessed_dim}"
                )

        return SimpleNamespace(
            x_train=x_train,
            theta_train=theta_train,
            ratio_train=ratio_train,
            score_train=score_train,
            labels_train=labels_train,
            x_test=x_test,
            theta_test=theta_test,
            ratio_test=ratio_test,
            score_test=score_test,
            labels_test=labels_test,
            preprocessed_train=preprocessed_train,
            preprocessed_test=preprocessed_test,
        )

    def _load_dataset(self, raw, mode: Literal["train", "test"] = "train"):
        if mode == "train":
            return self.dataset_cls(
                x=raw.x_train,
                theta=raw.theta_train,
                score=raw.score_train,
                ratio=raw.ratio_train,
                labels=raw.labels_train,
                preprocessed=raw.preprocessed_train,
                preprocessed_dim=self._preprocessed_dim,
                met=self._use_met,
            )
        if mode == "test":
            return self.dataset_cls(
                x=raw.x_test,
                theta=raw.theta_test,
                score=raw.score_test,
                ratio=raw.ratio_test,
                labels=raw.labels_test,
                preprocessed=raw.preprocessed_test,
                preprocessed_dim=self._preprocessed_dim,
                met=self._use_met,
            )
        raise ValueError(f"Invalid mode {mode}")

    @torch.enable_grad()
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
            "theta_dim": batch.theta.shape[-1],
        }
        if self._use_preprocessed:
            embedding_kwargs["preprocessed"] = batch.preprocessed
        if self._use_met:
            embedding_kwargs["met"] = batch.met

        # Regress log-likelihood ratio
        model_kwds = {
            "particles": batch.particles,
            "ptr": batch.ptr,
            "force_math": self.loss_fn.REQUIRES_SCORE,
            "embedding_kwargs": embedding_kwargs,
        }
        if self.cfg.model.key == "lgatr":
            model_kwds["scalars"] = (
                batch.preprocessed if self._use_preprocessed else None
            )
            model_kwds["met"] = batch.met if self._use_met else None

        log_ratio_pred = self.model(**model_kwds)
        return self.pack_output(
            theta=batch.theta,
            log_ratio_pred=log_ratio_pred,
            score=batch.score,
            ratio=batch.ratio,
            label=batch.label,
        )
