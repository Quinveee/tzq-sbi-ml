import numpy as np

from .asymptotic_limits import AsymptoticLimits
from .asymptotic_limits_histos import AsymptoticLimitsHistos


class AsymptoticLimitsRatios(AsymptoticLimits):
    """Direct per-event summation of the learned log-ratio.

    NOTE: this estimator is calibration-fragile. The kinematic LLR is
    ``n_events * <log r>_w``, so a sub-percent per-event calibration error is
    amplified by ``n_events`` (~1e5) into hundreds of spurious LLR units, and
    the theta-independent background (which the network only approximately
    zeroes, yet carries ~87% of the asimov weight) dominates the sum. Prefer
    ``AsymptoticLimitsRatiosHistos`` for SMEFT limits; keep this for
    comparison via ``limits.ratio_histo=false``.
    """

    NEEDS_HISTOS = False

    def log_r_kin(self, **kwds):
        # predictions: list[n_thetas] of arrays shape (n_events,) or (n_events,1)
        arr = np.asarray(kwds["predictions"])
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return arr  # (n_thetas, n_events)


class AsymptoticLimitsRatiosHistos(AsymptoticLimitsHistos):
    """Use the learned ratio as a (multi-dimensional) summary statistic and
    feed it through the same histogram-template likelihood as the score path.

    For each EFT direction ``k`` we evaluate ``log r_hat(x | theta_ref_k, SM)``
    at a fixed reference point, stack the results into a ``dim_theta``-component
    summary statistic, and histogram it exactly like the score. This (a)
    averages per-event noise over histogram bins and (b) cancels the
    theta-independent background, which appears identically in the asimov data
    histogram and in the morphed templates. Both fixes are absent from the
    direct-summation estimator above.

    All histogram machinery (``hist_bins``, ``histos``, ``log_r_kin``) is
    inherited unchanged from ``AsymptoticLimitsHistos`` -- only the summary
    statistic fed in differs, and that is computed in the experiment
    (``BaseExperimentRatios._histo_summary_stats``).
    """

    NEEDS_HISTOS = True
