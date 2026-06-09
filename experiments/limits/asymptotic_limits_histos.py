from typing import List

import numpy as np
from madminer.utils.histo import Histo

from .asymptotic_limits import AsymptoticLimits


class _StackedHistos:
    """Lightweight replacement for a ``list[Histo]`` that shares bin edges.

    Holds the bin edges (common to every grid point) and a single dense array of
    per-grid-point bin densities, ``densities`` with shape ``(n_grid, *n_bins)``.
    This lets the histogram template likelihood be built and evaluated with two
    vectorized passes instead of an ``O(n_grid)`` Python loop of ``Histo``
    objects (see ``AsymptoticLimitsHistos.histos`` / ``log_r_kin``).
    """

    def __init__(self, edges, n_bins, densities):
        self.edges = edges
        self.n_bins = n_bins
        self.densities = densities


def _bin_indices(x, edges, n_bins, drop_out_of_range):
    """Per-event bin index along each observable.

    Mirrors ``np.histogramdd`` binning (half-open bins, closed last bin). When
    ``drop_out_of_range`` is True (template filling) out-of-range events are
    flagged in ``in_range`` so they can be dropped, matching ``histogramdd``.
    When False (likelihood evaluation) indices are clipped into the edge bins,
    matching ``Histo.log_likelihood``.
    """
    idx_per_dim = []
    in_range = np.ones(x.shape[0], dtype=bool)
    for i in range(x.shape[1]):
        e = edges[i]
        ind = np.searchsorted(e, x[:, i], side="right") - 1
        # histogramdd's last bin is closed: x == last edge -> last bin.
        ind[x[:, i] == e[-1]] = n_bins[i] - 1
        in_range &= (ind >= 0) & (ind < n_bins[i])
        np.clip(ind, 0, n_bins[i] - 1, out=ind)
        idx_per_dim.append(ind)
    if drop_out_of_range:
        return idx_per_dim, in_range
    return idx_per_dim, None


def _cell_volumes(edges, n_bins):
    """Bin cell volumes, identical to ``Histo._fit`` (shared across grid points,
    so computed once instead of per template)."""
    bin_widths = []
    for ax in edges:
        ax = np.copy(ax)
        if len(ax) > 2:
            # First/last bins treated as at most twice the neighbouring width.
            ax[0] = max(ax[0], ax[1] - 2.0 * (ax[2] - ax[1]))
            ax[-1] = min(ax[-1], ax[-1] + 2.0 * (ax[-1] - ax[-2]))
        bin_widths.append(ax[1:] - ax[:-1])
    shape = tuple(n_bins)
    volumes = np.ones(shape)
    for obs, w in enumerate(bin_widths):
        bshape = [1] * len(shape)
        bshape[obs] = len(w)
        volumes = volumes * w.reshape(bshape)
    return volumes


class AsymptoticLimitsHistos(AsymptoticLimits):

    NEEDS_HISTOS = True

    def hist_bins(self, dim_theta: int):
        hist_bins_map = {1: (25,), 2: (8, 8)}
        return hist_bins_map.get(dim_theta, (5,) * dim_theta)

    def histos(self, scores: np.ndarray, weights) -> _StackedHistos:
        """Build per-grid-point template densities in two vectorized passes.

        Every grid point histograms the *same* template events ``scores`` with
        the *same* (shared, adaptive) bin edges; only the morphed ``weights``
        (shape ``(n_grid, n_toys)``) differ. So the events' bin assignment is
        computed once and the per-grid-point weighted bin counts are obtained
        with a single matmul against a one-hot bin-membership matrix — replacing
        ``n_grid`` separate ``Histo`` constructions (each of which redundantly
        re-binned the events, recomputed identical cell volumes via a Python
        ``ndindex`` loop, and computed unused uncertainties).

        The normalization reproduces ``Histo._fit`` exactly.
        """
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        weights = np.asarray(weights, dtype=np.float64)

        hist_bins = self.hist_bins(scores.shape[1])
        # Shared adaptive edges from the mean template weight (as before).
        ref = Histo(scores, weights.mean(axis=0), hist_bins, epsilon=1e-12)
        edges, n_bins = ref.edges, list(ref.n_bins)

        # Bin assignment of the template events (drop out-of-range like histogramdd)
        idx_per_dim, in_range = _bin_indices(
            scores, edges, n_bins, drop_out_of_range=True
        )
        total_bins = int(np.prod(n_bins))
        flat = np.ravel_multi_index(tuple(idx_per_dim), tuple(n_bins))

        # Weighted bin counts for ALL grid points at once via one-hot matmul:
        #   counts[g, b] = sum_e weights[g, e] * onehot[e, b]
        onehot = np.zeros((scores.shape[0], total_bins), dtype=np.float64)
        onehot[np.arange(scores.shape[0]), flat] = 1.0
        if in_range is not None:
            onehot[~in_range] = 0.0  # drop out-of-range template events
        counts = (weights @ onehot).reshape((-1,) + tuple(n_bins))

        # Reproduce Histo._fit normalization, vectorized over the grid axis (0):
        # normalize to unit sum, add epsilon, renormalize, divide by cell volume.
        eps = 1e-12
        axes = tuple(range(1, counts.ndim))
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = counts / counts.sum(axis=axes, keepdims=True)
            dens = dens + eps
            dens = dens / dens.sum(axis=axes, keepdims=True)
            dens = dens / _cell_volumes(edges, n_bins)[None, ...]
        dens[~np.isfinite(dens)] = 0.0

        return _StackedHistos(edges, n_bins, dens)

    def log_r_kin(
        self,
        *,
        predictions,
        theta_grid: np.ndarray,
        histos,
        **kwds,
    ) -> np.ndarray:
        """Per-event, per-grid-point log density of the asimov summary stats.

        Equivalent to looping ``histo.log_likelihood(asimov)`` over the grid, but
        vectorized: every template shares bin edges, so the asimov's bin
        assignment is identical for all grid points. Compute it once, then gather
        the per-grid-point densities in a single advanced-indexing op — turning
        an ``O(n_grid)`` Python loop of ``searchsorted`` calls into one
        ``searchsorted`` plus one gather.

        :return: array of shape ``(n_grid, n_events)``
        """
        summary = predictions[-1] if isinstance(predictions, list) else predictions
        x = np.asarray(summary)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if isinstance(histos, _StackedHistos):
            edges, n_bins, densities = histos.edges, histos.n_bins, histos.densities
        else:  # list[Histo] fallback (shared edges still assumed)
            edges, n_bins = histos[0].edges, list(histos[0].n_bins)
            densities = np.stack([h.histo for h in histos], axis=0)

        # Asimov bin assignment: clip into edge bins (matches Histo.log_likelihood).
        idx_per_dim, _ = _bin_indices(x, edges, n_bins, drop_out_of_range=False)

        gathered = densities[(slice(None),) + tuple(idx_per_dim)]
        with np.errstate(divide="ignore"):
            log_r_kin = np.log(gathered)
        return log_r_kin
