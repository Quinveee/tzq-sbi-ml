"""Enseble class to aggregate results from different experiments and runs"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .base.schemas import Chekcpoints
from .logger import LOGGER as _LOGGER
from .plotting import plot_intervals, plot_llr

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = _LOGGER.getChild(__name__)

# Translate model strings into labels appropriate to display on a plot
# Asume empty string to be histograms (no model)
MODEL2LABEL = {
    "mlp": "MLP",
    "lgatr": "LGATr",
    "transformer": "Transformer",
    "": "2D Histogram",
}

# For ploting LLR contours
COLORS = ["#20908C", "#D4BB36", "#A43FA0", "#4C72B0"]


class Ensemble:
    """
    Class to merge results from different expeirments and runs
    """

    def __init__(
        self, *, cfg: DictConfig, checkpoints_from: Dict, key: str = "ensemble"
    ) -> None:
        self.cfg = cfg
        self.key = key
        self.checkpoints_from = checkpoints_from

    def run(self) -> None:
        """
        Iterate over: experiment, model and run to create merged plots

        ::note:: In the future, grid, parameter names, ranges and resolutions
            will be defined in a file and loaded from there to avoid confusion

        """
        # Containers for the log-likelihood ratio arrays of predictions
        llr_all, std_all = [], []

        # Translate model string into a label appropriate for plotting
        labels_all = []

        # These should be loaded from a file ...
        grid = names = ranges = resolutions = None

        # Iterate over all combinations of exp, model and run possible
        for exp, model, runs in zip(*self.checkpoints_from.values()):

            # Container for the LLR of one combination of model and exp
            llr_this = []

            # Iterate over all runs
            for run in runs:
                ckpts_path = Path(
                    self.cfg.data.run_dir_base,
                    self.cfg.dataset.key,
                    exp,
                    model,
                    str(run),
                    self.cfg.data.ckpts,
                )

                # Load checkpoints
                checkpoints = Chekcpoints(
                    **torch.load(ckpts_path, map_location="cpu", weights_only=False)
                )

                llr_this.append(checkpoints.limits.llr)
                print(
                    f"Exp: {exp} model {model} resolutions {checkpoints.limits.resolutions}"
                )

                # Since we are not storing these in a file, we have to check for
                # consistency. FIXME
                assert names == checkpoints.limits.param_names or names is None
                assert np.all(ranges == checkpoints.limits.ranges) or ranges is None
                assert (
                    np.all(resolutions == checkpoints.limits.resolutions)
                    or resolutions is None
                )

                # Reasing to keep checking
                grid = checkpoints.limits.grid
                names = checkpoints.limits.param_names
                ranges = checkpoints.limits.ranges
                resolutions = checkpoints.limits.resolutions

            # Average over all runs
            mean_llr = np.asarray(llr_this).mean(axis=0)

            # Rescale the averaged LLR
            mean_llr = mean_llr - mean_llr.max()

            # Append results to global containers
            llr_all.append(mean_llr)
            std_all.append(np.asarray(llr_this).std(axis=0))

            # Keep track of which model produced which results
            labels_all.append(MODEL2LABEL[model])

        plot_llr(
            llr_list=llr_all,
            std_list=std_all,
            param_names=names,
            grid=grid,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels_all,
            to="fig.png",
            conf_levels=(0.68,),
            colors=COLORS,
        )
        plot_intervals(llr_all, grid, labels_all, to="limits.png", colors=COLORS)

    def __call__(self):
        self.run()
