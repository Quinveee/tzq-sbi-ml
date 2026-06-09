from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from madminer.analysis import DataAnalyzer
from madminer.limits import AsymptoticLimits as _AsymptoticLimits
from madminer.utils.various import mdot
from scipy.stats import chi2, poisson

from ..base.schemas import Limits


# NOTE: Test split is hardcoded in madminer augmentation step (0.2)
# since the events file (h5 file) is unchanged here (see `cfg.limits.test_split`)
# , is safe to use that split now to calculate the 'training' partition
# for the calculation of histograms (Madminer calculates this partitions
# deterministically)
#
# In an ideal world both augmentation and this code should grab the
# test split value from the same place

__all__ = ("AsymptoticLimitsHistos", "AsymptoticLimitsRatio")


class AsymptoticLimits(DataAnalyzer, ABC):
    """Most of the code here has been copied
    almost verbatim from madminer code, with simplifications"""

    NEEDS_HISTOS: bool

    def __init__(self, h5_filename):
        super().__init__(h5_filename, False, False)

    def asimov_data(
        self,
        theta,
        sample_only_from_closest_benchmark,
        test_split,
        n_asimov,
        partition: Literal["train", "test", "all"] = "test",
    ):
        x, weights_benchmarks, correction_factor = self.weighted_events_from_partition(
            n_draws=n_asimov,
            partition=partition,
            test_split=test_split,
            thetas=None,
            generated_close_to=theta if sample_only_from_closest_benchmark else None,
        )
        weights_benchmarks *= correction_factor

        theta_matrix = self._get_theta_benchmark_matrix(theta)
        weights_theta = mdot(theta_matrix, weights_benchmarks)
        weights_theta /= np.sum(weights_theta)

        return x, weights_theta

    def weighted_events_from_partition(
        self,
        n_draws,
        partition: Literal["train", "test"],
        test_split: float,
        thetas=None,
        generated_close_to=None,
    ):
        """Returns all events with benchmark weights for partition plus correction
        factor. Only train or test available"""

        assert partition in (
            "train",
            "test",
            "all",
        ), f"Invalid partition key: {partition}"

        # Weighted histo data
        if partition == "all":
            # Entire dataset (train + test). Used by the histogram baseline,
            # which has no trained model and so cannot overfit the events it
            # also reuses as Asimov data; this gives it the same total MC budget
            # the neural pipeline gets (train for the network + test for Asimov).
            start_event, end_envent, cf = 0, None, 1.0
        else:
            start_event, end_envent, cf = self._calculate_partition_bounds(
                partition, test_split, 0.0
            )

        x, weights_benchmarks = self.weighted_events(
            start_event=start_event,
            end_event=end_envent,
            n_draws=n_draws,
            generated_close_to=generated_close_to,
        )

        if thetas is None:
            return x, weights_benchmarks, cf

        weights = np.asarray(self._weights(thetas, None, weights_benchmarks))
        return x, weights, cf

    def calculate_xsecs(self, thetas, test_split, partition="test"):

        # Total xsecs for benchmarks
        _, weights, correction_factor = self.weighted_events_from_partition(
            n_draws=None, partition=partition, test_split=test_split
        )
        xsecs_benchmarks = np.sum(weights, axis=0)

        # xsecs at thetas
        xsecs = []
        for theta in thetas:
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            xsecs.append(mdot(theta_matrix, xsecs_benchmarks) * correction_factor)
        return np.asarray(xsecs)

    def calculate_log_likelihood_xsec(
        self, n_events, theta_grid, luminosity, test_split, partition="test"
    ):
        n_events_rounded = int(np.round(n_events, 0))
        n_predicted = (
            self.calculate_xsecs(theta_grid, test_split=test_split, partition=partition)
            * luminosity
        )
        log_p = poisson.logpmf(k=n_events_rounded, mu=n_predicted)
        return log_p

    def asymptotic_p_value(self, llr):
        return chi2.sf(x=-2.0 * llr, df=self.n_parameters)

    @staticmethod
    def theta_grid(theta_ranges, resolutions):
        """Returns the grid of parameters so that the column index represents the
        different parameters and the rows the different values"""
        theta_grid, _ = _AsymptoticLimits._make_theta_grid(
            theta_ranges=theta_ranges, resolutions=resolutions
        )
        return theta_grid

    @abstractmethod
    def log_r_kin(self, *args, **kwds) -> np.ndarray:
        pass

    def limits(
        self,
        *,
        predictions,
        n_events,
        x_weights,
        theta_grid,
        luminosity,
        test_split,
        histos=None,
        partition="test",
    ):
        x_weights /= np.sum(x_weights)
        x_weights.astype(np.float64)

        llr_kwds = {
            "predictions": predictions,
            "theta_grid": theta_grid,
            "histos": histos,
        }

        # Subclass specific computation
        log_r_kin = self.log_r_kin(**llr_kwds)

        # Common post processing
        log_r_kin = log_r_kin.astype(np.float64)
        log_r_kin = _AsymptoticLimits._clean_nans(log_r_kin)
        log_r_kin = n_events * np.sum(log_r_kin * x_weights[None, :], axis=1)

        # Include xsecs
        log_p_xsec = self.calculate_log_likelihood_xsec(
            n_events, theta_grid, luminosity, test_split=test_split, partition=partition
        )

        # Combine and get p-values
        log_r = log_r_kin + log_p_xsec
        log_r, i_ml = _AsymptoticLimits._subtract_mle(log_r)
        p_values = self.asymptotic_p_value(log_r)

        return Limits(
            param_names=list(self.parameters.keys()),
            grid=theta_grid,
            p_values=p_values,
            mle=i_ml,
            llr_kin=log_r_kin,
            rate_ll=log_p_xsec,
        )
