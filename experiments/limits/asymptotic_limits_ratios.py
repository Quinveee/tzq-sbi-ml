import numpy as np

from .asymptotic_limits import AsymptoticLimits


class AsymptoticLimitsRatios(AsymptoticLimits):

    NEEDS_HISTOS = False

    def log_r_kin(self, **kwds):
        # predictions: list[n_thetas] of arrays shape (n_events,) or (n_events,1)
        arr = np.asarray(kwds["predictions"])
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        return arr  # (n_thetas, n_events)
