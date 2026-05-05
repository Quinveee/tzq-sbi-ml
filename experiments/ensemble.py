"""Enseble class to aggregate results from different experiments and runs"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .base.schemas import Chekcpoints
from .logger import LOGGER as _LOGGER
from .plotting import (
    plot_intervals,
    plot_llr,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = _LOGGER.getChild(__name__)

# Translate model strings into labels appropriate to display on a plot
# Asume empty string to be histograms (no model)
MODEL2LABEL = {
    "mlp": "MLP",
    "lgatr": "LGATr",
    "transformer": "Transformer",
    "lloca": "Transformer (LLoCa)",
    "lorentznet": "LorentzNet",
    "": "2D Histogram",
}

# Mapping from ensemble model key → directory name under run_dirs/<dataset>/<exp>/
_MODEL2RUNDIR = {
    "lloca": "transformer_lloca",
}

# Model keys that have a `<name>_preprocessed` run-dir variant
_PREPROCESSED_CAPABLE = {"transformer", "lloca", "lgatr", "lorentznet"}

# Translate experiment keys into human-readable method labels
EXP2LABEL = {
    "local": "Score Regression (SALLY)",
    "ratio": "Likelihood Ratio Regression (ROLR)",
    "histo": "Histogram",
}

# Filesystem-friendly directory names for each experiment key
EXP2DIRNAME = {
    "local": "score_regression",
    "ratio": "ratio_regression",
    "histo": "histogram",
}

# For ploting LLR contours
COLORS = ["#20908C", "#D4BB36", "#A43FA0", "#4C72B0", "#E07A3C", "#7A4A2B"]


class Ensemble:
    """
    Class to merge results from different expeirments and runs
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        checkpoints_from: Dict,
        key: str = "ensemble",
        processed: bool = False,
    ) -> None:
        self.cfg = cfg
        self.key = key
        self.checkpoints_from = checkpoints_from
        self.processed = processed

    def _resolve_run_model_dir(self, model_key: str) -> str:
        """Resolve model directory name used under run_dirs for checkpoints."""
        if model_key in _MODEL2RUNDIR:
            base = _MODEL2RUNDIR[model_key]
        elif model_key == "transformer":
            lloca_active = False
            if self.cfg.model.get("key", None) == "transformer":
                lloca_cfg = self.cfg.model.get("LLoCa", {})
                lloca_active = lloca_cfg.get("active", False)
            base = "transformer_lloca" if lloca_active else "transformer"
        else:
            base = model_key

        if self.processed and model_key in _PREPROCESSED_CAPABLE:
            base = f"{base}_preprocessed"

        return base

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

        # Derive a human-readable method label from the experiment keys
        exp_keys = set(self.checkpoints_from["exp"])
        # Use the ML method (non-histo) if available, otherwise "Histogram"
        ml_keys = exp_keys - {"histo"}
        method = ", ".join(EXP2LABEL.get(k, k) for k in sorted(ml_keys)) if ml_keys else EXP2LABEL["histo"]
        if self.processed:
            method = f"{method} Preprocessed"

        # Build output directory: images/<dataset>/<exp_key>/<method_dirname>/
        exp_dir_key = sorted(ml_keys)[0] if ml_keys else "histo"
        method_dirname = EXP2DIRNAME.get(exp_dir_key, exp_dir_key)
        out_dir = Path("images", self.cfg.dataset.key, exp_dir_key, method_dirname)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Suffix for output filenames so processed/raw runs don't overwrite each other
        plot_stem = f"{method_dirname}_preprocessed" if self.processed else method_dirname

        # These should be loaded from a file ...
        grid = names = ranges = resolutions = None

        # Iterate over all combinations of exp, model and run possible
        for exp, model, runs in zip(*self.checkpoints_from.values()):

            # Container for the LLR of one combination of model and exp
            llr_this = []

            # Iterate over all runs
            for run in runs:
                run_model_dir = self._resolve_run_model_dir(model)
                ckpts_path = Path(
                    self.cfg.data.run_dir_base,
                    self.cfg.dataset.key,
                    exp,
                    run_model_dir,
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
                new_names = checkpoints.limits.param_names
                new_ranges = checkpoints.limits.ranges
                new_resolutions = checkpoints.limits.resolutions

                # Treat empty lists as unset (some checkpoints lack param_names)
                if names and new_names:
                    assert names == new_names, f"param_names mismatch: {names} vs {new_names}"
                if ranges is not None and new_ranges is not None:
                    assert np.all(ranges == new_ranges)
                if resolutions is not None and new_resolutions is not None:
                    assert np.all(resolutions == new_resolutions)

                # Keep the most informative value
                grid = checkpoints.limits.grid
                names = new_names if new_names else names
                ranges = new_ranges if new_ranges is not None else ranges
                resolutions = new_resolutions if new_resolutions is not None else resolutions

            # Average over all runs
            mean_llr = np.asarray(llr_this).mean(axis=0)

            # Rescale the averaged LLR
            mean_llr = mean_llr - mean_llr.max()

            # Append results to global containers
            llr_all.append(mean_llr)
            std_all.append(np.asarray(llr_this).std(axis=0))

            # Keep track of which model produced which results
            labels_all.append(MODEL2LABEL[model])

        # Fallback: if no checkpoint stored param_names, generate generic ones
        if names is None and grid is not None:
            names = [f"param_{i}" for i in range(grid.shape[1])]

        plot_llr(
            llr_list=llr_all,
            std_list=std_all,
            param_names=names,
            grid=grid,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels_all,
            to=str(out_dir / f"{plot_stem}.png"),
            conf_levels=(0.68,),
            colors=COLORS,
            method=method,
        )
        plot_intervals(llr_all, grid, labels_all, to=str(out_dir / f"{plot_stem}_limits.png"), colors=COLORS, method=method)

    def __call__(self):
        self.run()
