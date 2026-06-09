"""Estimating LL ratios using histogrammed event observables.

Unlike the neural estimators, the histogram has no trained model: the summary
statistic is just the chosen raw observables (``cfg.limits.hist_vars``). Because
there is nothing to overfit, it can build both its templates and its Asimov data
from the full train+test dataset (``histo_full_partition=true``), giving it the
same total MC budget the neural pipeline uses (train partition for the network +
test partition for the Asimov). The Asimov is capped at ``n_asimov`` and the
templates at ``n_toys`` so the dense ``(n_grid x n_events)`` likelihood arrays
stay within memory regardless of grid resolution.
"""

import numpy as np

from .base.base_experiment import BaseExperiment
from .base.schemas import Limits
from .limits import AsymptoticLimitsHistos


class ExperimentHistos(BaseExperiment):
    """
    Experiment class for the histograms of observables approach
    """

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "histos"
        super().__init__(*args, **kwds)

    def _init(self):
        pass

    def _seed_rng(self) -> None:
        """Seed numpy from ``data.run`` so MadMiner's global-RNG resampling is
        reproducible and identical between eval-only re-runs."""
        raw = self.cfg.data.get("run", 0)
        try:
            seed = int(raw)
        except (TypeError, ValueError):
            seed = abs(hash(str(raw))) % (2**32)
        np.random.seed(seed)

    def eval_lims(self) -> Limits:
        """
        Compute asymptotic limits from histograms of raw observables, using the
        codebase histogram machinery (same back-end as the score/ratio
        estimators) so the only difference from the neural paths is the choice
        of summary statistic.

        :return: Limits object with grid and estimated LLR information
        :rtype: Limits
        """
        cfg = self.cfg
        alims = AsymptoticLimitsHistos(cfg.dataset.events_file)

        theta_true = np.asarray(cfg.limits.asimov.theta_true, dtype=float)
        test_split = cfg.limits.test_split
        soc = cfg.limits.asimov.sample_only_from_closest_benchmark
        # Two modes:
        #   histo_full_partition=true  -> templates AND Asimov from the full
        #       train+test set (max statistics; the histogram has no model to
        #       overfit, matches the neural pipeline's total MC budget).
        #   histo_full_partition=false -> disjoint split, mirroring MadMiner's
        #       partition choice: templates from the "train" partition
        #       (1 - test_split), Asimov from the "test" partition. No overlap.
        if cfg.limits.get("histo_full_partition", True):
            template_partition = asimov_partition = "all"
        else:
            template_partition, asimov_partition = "train", "test"

        theta_grid = alims.theta_grid(
            theta_ranges=cfg.limits.theta_ranges,
            resolutions=cfg.limits.resolutions,
        )

        # Column indices of the chosen observables in the event array. The histo
        # runs on the feature-level file (model=noop -> dataset.path=features),
        # whose observable names include pt_j1, pt_l1, ...
        obs_names = list(alims.observables.keys())
        missing = [v for v in cfg.limits.hist_vars if v not in obs_names]
        assert not missing, (
            f"hist_vars {missing} not found in events file "
            f"{cfg.dataset.events_file}. Available e.g.: {obs_names[:8]}"
        )
        obs_idx = [obs_names.index(v) for v in cfg.limits.hist_vars]

        # --- Asimov data (the observed summary stats), capped at n_asimov ---
        self._seed_rng()
        x_asimov, w_asimov = alims.asimov_data(
            theta_true,
            soc,
            test_split,
            cfg.limits.asimov.n_asimov,
            partition=asimov_partition,
        )
        s_asimov = np.asarray(x_asimov)[:, obs_idx]

        # Expected event count for the rate term, at theta_true (cross section is
        # partition-invariant via the correction factor; use the Asimov pool).
        n_events = (
            cfg.limits.luminosity
            * alims.calculate_xsecs(
                [theta_true], test_split, partition=asimov_partition
            )[0]
        )

        # --- Template events, capped at n_toys; morphed weights give the
        # per-grid-point templates ---
        self._seed_rng()
        x_histo, w_histo, _ = alims.weighted_events_from_partition(
            n_draws=cfg.limits.n_toys,
            partition=template_partition,
            test_split=test_split,
            thetas=theta_grid,
            generated_close_to=theta_true if soc else None,
        )
        s_histo = np.asarray(x_histo)[:, obs_idx]
        histos = alims.histos(s_histo, w_histo)

        return alims.limits(
            predictions=s_asimov,
            n_events=n_events,
            x_weights=w_asimov,
            theta_grid=theta_grid,
            luminosity=cfg.limits.luminosity,
            test_split=test_split,
            histos=histos,
            partition=asimov_partition,
        )

    def _run(self) -> None:
        if self.cfg.modes.eval:
            self.checkpoints.limits = self.eval_lims()
        if self.cfg.modes.plot:
            self.plot("Histogram")
