from typing import List

import numpy as np
from madminer.limits import AsymptoticLimits as _AsymptoticLimits
from madminer.utils.histo import Histo

from .asymptotic_limits import AsymptoticLimits


class AsymptoticLimitsHistos(AsymptoticLimits):

    NEEDS_HISTOS = True

    def hist_bins(self, dim_theta: int):
        hist_bins_map = {1: (25,), 2: (8, 8)}
        return hist_bins_map.get(dim_theta, (5,) * dim_theta)

    def histos(self, scores: np.ndarray, weights) -> List[Histo]:
        hist_bins = self.hist_bins(scores.shape[1] if scores.ndim > 1 else 1)
        histo_edges = Histo(
            scores, weights.mean(axis=0), hist_bins, epsilon=1e-12
        ).edges
        return [Histo(scores, weight, histo_edges, epsilon=1e-12) for weight in weights]

    def log_r_kin(
        self,
        *,
        predictions,
        theta_grid: np.ndarray,
        histos: List[Histo],
        **kwds,
    ) -> np.ndarray:
        # ``predictions`` is the per-event summary statistic on the asimov
        # events: shape (n_events, dim). Accept either the bare array or the
        # legacy single-element list wrapper.
        summary = predictions[-1] if isinstance(predictions, list) else predictions
        log_r_kin, *_ = _AsymptoticLimits._calculate_log_likelihood_histo(
            summary_stats=summary, theta_grid=theta_grid, histos=histos
        )
        return log_r_kin
