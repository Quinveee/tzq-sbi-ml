"""Experiment class for likelihood ratio regression in the 'features' case"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from ..base.base_experiment_ratios import BaseExperimentRatios
from ..logger import LOGGER as _LOGGER
from ..utils import device
from .collate import parametrized_collate_features_fn
from .datasets import ParametrizedFeaturesDataset

if TYPE_CHECKING:
    from .schemas import ParametrizedFeaturesBatch

LOGGER = _LOGGER.getChild(__name__)


class ExperimentRatiosFeatures(BaseExperimentRatios):
    dataset_cls = ParametrizedFeaturesDataset
    collate_fn = staticmethod(parametrized_collate_features_fn)

    @torch.no_grad()
    def eval_grid(
        self,
        x: np.ndarray,
        theta_grid: np.ndarray,
        chunk_size: int = 128,
    ) -> np.ndarray:
        """Vectorized grid evaluation for feature-level inputs.

        The base implementation tiles x/theta into a fresh Dataset + DataLoader
        per chunk. With num_workers=0 that rebuilds chunk_size * n_events event
        dataclasses and deep-copies each one via ``dataclasses.asdict`` inside
        the collate fn, which dominates runtime and does NOT scale with batch
        size (hence the flat ~40s/chunk regardless of eval_batch_size).

        Here we build the tiled (x, theta) tensors directly on the eval device
        and call the model in large blocks, skipping the per-sample
        Dataset/collate path entirely. The grid sweep never consumes the score,
        so this runs under a real ``no_grad`` (no autograd graph). Output shape
        is (n_grid, n_events), identical to the base method.
        """
        self.model.eval()
        device_kwds = self.device_kwds.copy()
        device_kwds["device"] = device(self.cfg.devices.eval)
        self.model = self.model.to(**device_kwds)
        dev = device_kwds["device"]
        dt = device_kwds["dtype"]
        LOGGER.info(f"Model moved to {dev}")

        x_norm = self.normalizer.transform(x)
        n_events = len(x_norm)
        n_grid = len(theta_grid)

        x_t = torch.as_tensor(np.ascontiguousarray(x_norm)).to(device=dev, dtype=dt)
        theta_t = torch.as_tensor(np.ascontiguousarray(theta_grid)).to(
            device=dev, dtype=dt
        )
        if theta_t.ndim == 1:
            theta_t = theta_t.unsqueeze(-1)

        # Evaluate `block` grid points per forward so the flattened
        # (block * n_events) batch stays near the configured row budget.
        eval_batch = self._eval_batch_size(default=131072)
        block = max(1, eval_batch // max(1, n_events))

        out = np.empty((n_grid, n_events), dtype=np.float32)
        LOGGER.info(
            f"Evaluating {n_grid} grid points x {n_events} events "
            f"({block} grid points/forward)"
        )
        for start in tqdm(range(0, n_grid, block), desc="Evaluating grid"):
            th = theta_t[start : start + block]
            m = th.shape[0]
            x_b = x_t.repeat(m, 1)
            th_b = th.repeat_interleave(n_events, dim=0)
            log_ratio = self.model(x_b, theta=th_b).reshape(m, n_events)
            out[start : start + m] = log_ratio.float().cpu().numpy()

        return out

    def _preds(self, batch: ParametrizedFeaturesBatch):
        # Grad enabling is left to the caller (see particles.ratios._preds).
        batch.theta.requires_grad_(self.loss_fn.REQUIRES_SCORE and torch.is_grad_enabled())
        log_ratio_pred = self.model(batch.x, theta=batch.theta)
        return self.pack_output(
            theta=batch.theta,
            log_ratio_pred=log_ratio_pred,
            score=batch.score,
            ratio=batch.ratio,
            label=batch.label,
        )
