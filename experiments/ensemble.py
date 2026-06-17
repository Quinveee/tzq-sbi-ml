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
    plot_ratio_calibration,
    plot_score_calibration,
)
from .plotting.utils import PARAM2LABEL

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .base.base_experiment_ml import BaseExperimentML

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

# Mapping from ensemble model key → (model config to compose, extra Hydra
# overrides) used when *rebuilding* an experiment to re-run inference for the
# correlation study. `lloca` is the transformer config with equivariant frames.
_MODEL2COMPOSE = {
    "lloca": ("transformer", ["model.LLoCa.active=true"]),
}

# Plain-text Wilson-coefficient labels for the markdown correlation table
# (PARAM2LABEL holds their LaTeX equivalents, used in the .tex export).
PARAM2PLAIN = {"cHt": "c_Ht", "ctWRe": "c_tW", "ctBRe": "c_tB"}

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
        correlations: bool = True,
    ) -> None:
        self.cfg = cfg
        self.key = key
        self.checkpoints_from = checkpoints_from
        self.processed = processed
        self.correlations = correlations

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
        # Containers for the log-likelihood ratio arrays of predictions.
        # `runs_all` keeps the per-run LLR stacks so the marginal uncertainty
        # band can be computed from the marginalised per-run curves.
        llr_all, std_all, runs_all = [], [], []

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

                if checkpoints.limits is None:
                    raise ValueError(
                        f"Checkpoint {ckpts_path} has no `limits` (asymptotic scan "
                        f"was never computed for exp={exp} model={model!r} run={run}). "
                        "Recompute it with:\n"
                        f"    python main.py exp={exp} model={model or 'mlp'} "
                        f"dataset={self.cfg.dataset.key} data.run={run} "
                        "modes.train=false modes.eval=true modes.plot=false\n"
                        "(for the LLoCa transformer add `model.LLoCa.active=true`), "
                        "or point this entry at a run that already has limits."
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
            runs_all.append(np.asarray(llr_this))

            # Keep track of which model produced which results
            labels_all.append(MODEL2LABEL[model])

        # Fallback: if no checkpoint stored param_names, generate generic ones
        if names is None and grid is not None:
            names = [f"param_{i}" for i in range(grid.shape[1])]

        plot_llr(
            llr_list=llr_all,
            std_list=std_all,
            runs_list=runs_all,
            param_names=names,
            grid=grid,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels_all,
            to=str(out_dir / f"{plot_stem}.png"),
            conf_levels=(0.68,),
            colors=COLORS,
            mode="average",
            method=method,
        )
        # The 1D and 3D plots already include the limits in their combined
        # figures, so only produce a standalone limits plot for the 2D case.
        if grid is not None and grid.shape[1] == 2:
            plot_intervals(
                llr_all,
                grid,
                labels_all,
                to=str(out_dir / f"{plot_stem}_limits.png"),
                colors=COLORS,
                resolutions=resolutions,
                param_names=names,
                mode="average",
                method=method,
            )

        # Truth-vs-prediction correlation study over the full test set. Runs
        # after the LLR plots so those are always produced even if the (heavier)
        # correlation pass fails or is skipped.
        if self.correlations:
            self.eval_correlations(out_dir, plot_stem)

    # ------------------------------------------------------------------ #
    # Truth-vs-prediction correlation study
    # ------------------------------------------------------------------ #

    def _build_experiment(self, exp: str, model: str, run: int) -> "BaseExperimentML":
        """Rebuild a trained experiment so we can re-run inference on the test set.

        The ensemble only stores precomputed ``limits``; the event-level
        correlation needs the model itself. We recompose the exact Hydra config
        for ``exp``/``model``/``dataset``/``run`` (mirroring ``main.py``), load
        the saved weights via ``modes.recycle``, and build the test loader —
        without re-running training, eval or the asymptotic scan.
        """
        # Local imports: keep the module importable without a full Hydra/torch
        # environment and avoid paying the cost unless correlations are run.
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        from helpers.derive_config import derive_config

        compose_model, extra = _MODEL2COMPOSE.get(model, (model, []))
        overrides = [
            f"exp={exp}",
            f"model={compose_model}",
            f"dataset={self.cfg.dataset.key}",
            f"data.run={run}",
            "modes.train=false",
            "modes.eval=false",
            "modes.plot=false",
            "modes.recycle=true",
            *extra,
        ]
        if self.processed and model in _PREPROCESSED_CAPABLE:
            overrides.append("data.preprocessed=true")

        # hydra.run.dir is `.` and output_subdir is null (see conf/hydra.yaml),
        # so the working directory stays the project root and conf/ lives here.
        conf_dir = str((Path.cwd() / "conf").resolve())
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="config", overrides=overrides)

        cfg = derive_config(cfg)
        experiment = instantiate(cfg.exp)(cfg=cfg)
        experiment.init()    # run dir + load checkpoints (weights, limits)
        experiment._init()   # model (recycled weights), datasets, loaders
        return experiment

    @staticmethod
    def _test_truth_pred(experiment: "BaseExperimentML", exp: str):
        """Return ``(y_true, y_pred)`` over the full test set for one experiment.

        Mirrors the per-experiment ``plot_diagnostics``: score regressors compare
        the (possibly multi-component) score, ratio regressors compare ``log r``.
        """
        y_pred = experiment.eval(experiment.test_loader)
        if exp == "ratio":
            y_true = np.log(experiment.test_dataset._ratios.ravel())
            y_pred = y_pred.ravel()
        else:
            y_true = np.asarray(experiment.test_dataset._score)
        return np.asarray(y_true), np.asarray(y_pred)

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson ρ, guarding against degenerate (constant / tiny) inputs."""
        if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    @classmethod
    def _corr_metrics(cls, exp: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Per-run correlation(s) and MSE for one (exp, model, run).

        Score regression returns one ρ per component (Wilson coefficient); ratio
        regression returns a single ρ on ``log r``. NaNs/infs are dropped.
        """
        if exp == "ratio":
            yt, yp = y_true.ravel(), y_pred.ravel()
            valid = np.isfinite(yt) & np.isfinite(yp)
            yt, yp = yt[valid], yp[valid]
            return {
                "kind": "logr",
                "rho": [cls._pearson(yt, yp)],
                "mse": float(np.mean((yp - yt) ** 2)) if yt.size else float("nan"),
            }

        # Score regression (1+ components).
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        rhos = []
        for d in range(y_true.shape[1]):
            yt, yp = y_true[:, d], y_pred[:, d]
            valid = np.isfinite(yt) & np.isfinite(yp)
            rhos.append(cls._pearson(yt[valid], yp[valid]))
        valid_all = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
        mse = (
            float(np.mean((y_pred[valid_all] - y_true[valid_all]) ** 2))
            if valid_all.any()
            else float("nan")
        )
        return {"kind": "score", "rho": rhos, "mse": mse}

    def _mle_bias_from_limits(self, limits) -> np.ndarray:
        """Signed per-coefficient MLE bias ``θ̂ - θ_true`` (one value per coeff).

        ``θ̂`` is read off the *marginalised* 1D likelihood per coefficient — the
        same reduction the contour/marginal plots use (see
        :meth:`_marginal_mle`) — so the table agrees with the plots. ``θ_true``
        is the Asimov truth from the config.
        """
        theta_true = np.asarray(self.cfg.limits.asimov.theta_true, dtype=float)
        if limits is None:
            return np.full(theta_true.shape, np.nan)
        theta_hat, _ = self._marginal_mle(limits.grid, limits.llr, limits.resolutions)
        return theta_hat - theta_true

    @staticmethod
    def _marginal_mle(grid, llr, resolutions, dchi2: float = 1.0):
        """Per-coefficient MLE and constraint flag from the averaged marginal.

        The ensemble plots reduce the N-D ``-2lnL`` to a 1D curve per coefficient
        by **averaging** over the other coefficients (``mode="average"`` in
        :func:`plot_llr` / ``_reduce_llr``), then read the minimum. We replicate
        exactly that reduction here so the tabulated MLE bias and the plotted
        marginal coincide — rather than the global argmin / profile, which rails
        to grid corners on flat directions even when the marginal is a clean
        parabola at zero.

        For each coefficient the marginal minimum is refined sub-grid by a
        parabolic fit (clamped to the bracketing cell; grid node kept at a
        boundary or non-convex point). The constraint flag is ``True`` when the
        marginal's Δχ² < ``dchi2`` (≈1σ) region reaches the grid edge.

        :return: ``(theta_hat, unconstrained)`` arrays of length ``d``.
        """
        grid = np.asarray(grid, dtype=float)
        res = [int(r) for r in resolutions]
        d = grid.shape[1]
        values_nd = (-2.0 * np.asarray(llr, dtype=float)).reshape(res)

        theta_hat = np.empty(d, dtype=float)
        unconstrained = np.empty(d, dtype=bool)
        for a in range(d):
            others = tuple(k for k in range(d) if k != a)
            marg = values_nd.mean(axis=others) if others else values_nd.copy()
            marg = marg - marg.min()
            vals = np.unique(grid[:, a])

            i = int(np.argmin(marg))
            theta_hat[a] = vals[i]
            if 0 < i < vals.size - 1:
                y1, y2, y3 = marg[i - 1], marg[i], marg[i + 1]
                curv = y1 - 2.0 * y2 + y3
                if curv > 0.0:
                    h = float(vals[i + 1] - vals[i])  # uniform per axis
                    delta = float(np.clip(0.5 * h * (y1 - y3) / curv, -h, h))
                    theta_hat[a] = vals[i] + delta

            region = vals[marg < dchi2]
            unconstrained[a] = (
                region.size == 0
                or np.isclose(region.min(), vals.min())
                or np.isclose(region.max(), vals.max())
            )
        return theta_hat, unconstrained

    def _mle_bias_aggregate(self, limits_list):
        """Aggregate the MLE bias across seeds, consistent with the LLR plots.

        Returns ``(central, std, unconstrained)`` per coefficient. The central
        value and constraint flag come from the *seed-averaged* LLR surface,
        reduced to per-coefficient marginals exactly as the plots do (see
        :meth:`_marginal_mle`) — so the table and plots agree. ``std`` is the
        per-seed spread of the marginal MLEs (a stability indicator).
        """
        theta_true = np.asarray(self.cfg.limits.asimov.theta_true, dtype=float)
        valid = [L for L in limits_list if L is not None]
        if not valid:
            nan = np.full(theta_true.shape, np.nan)
            return nan, nan, np.zeros(theta_true.shape, dtype=bool)

        per_seed = np.asarray(
            [self._mle_bias_from_limits(L) for L in valid], dtype=float
        )
        std = np.nanstd(per_seed, axis=0)

        # Seed-averaged surface — only valid if every seed shares the same grid.
        base = valid[0]
        same_grid = all(
            L.grid.shape == base.grid.shape and np.allclose(L.grid, base.grid)
            for L in valid
        )
        if not same_grid:
            LOGGER.warning(
                "MLE bias: seeds have mismatched θ grids; falling back to "
                "mean-of-per-seed MLE (will not match the averaged-surface plot)."
            )
            return np.nanmean(per_seed, axis=0), std, np.zeros(theta_true.shape, bool)

        avg_llr = np.mean([np.asarray(L.llr, dtype=float) for L in valid], axis=0)
        theta_hat, unconstrained = self._marginal_mle(
            base.grid, avg_llr, base.resolutions
        )
        return theta_hat - theta_true, std, unconstrained

    def _load_checkpoint_limits(self, exp: str, model: str, run: int):
        """Load just the ``limits`` from a run's checkpoint (no model rebuild).

        Used for the histogram baseline, which has no per-event prediction to
        regress but does store an asymptotic scan we can read the MLE from.
        Mirrors the path convention used in :meth:`run`.
        """
        ckpts_path = Path(
            self.cfg.data.run_dir_base,
            self.cfg.dataset.key,
            exp,
            self._resolve_run_model_dir(model),
            str(run),
            self.cfg.data.ckpts,
        )
        if not ckpts_path.exists():
            return None
        return Chekcpoints(
            **torch.load(ckpts_path, map_location="cpu", weights_only=False)
        ).limits

    def _histo_row(self, exp: str, model: str, runs) -> Dict:
        """Aggregate the histogram baseline: only MLE bias is defined.

        The histogram method bins observables; it never predicts ``t`` or
        ``log r`` per event, so the correlation and MSE columns are left empty.
        """
        limits_list = [self._load_checkpoint_limits(exp, model, run) for run in runs]
        param_names = next(
            (list(L.param_names) for L in limits_list if L is not None), None
        )
        central, std, unconstrained = self._mle_bias_aggregate(limits_list)
        return {
            "label": MODEL2LABEL[model],
            "kind": "histo",
            "param_names": param_names,
            "n_runs": len(runs),
            "rho_mean": np.array([]),
            "rho_std": np.array([]),
            "mse_mean": float("nan"),
            "mse_std": float("nan"),
            "mle_mean": central,
            "mle_std": std,
            "mle_unconstrained": unconstrained,
        }

    def eval_correlations(self, out_dir: Path, plot_stem: str) -> None:
        """Truth-vs-prediction correlation study over the full test set.

        For every ML entry in ``checkpoints_from`` (histograms are skipped — they
        have no event-level prediction) this rebuilds the model for each seed,
        evaluates it on the *whole* test set (not the Asimov/test partition), and
        reports ρ = mean ± std over seeds together with the MSE and MLE bias.
        Per-model scatter plots (pooled over seeds) are written alongside a
        markdown + LaTeX summary table.
        """
        LOGGER.info("Starting truth-vs-prediction correlation study over test set")
        rows = []

        for exp, model, runs in zip(*self.checkpoints_from.values()):
            # Histograms (empty model key) have no per-event regression target,
            # so only the MLE bias is defined — read it from the stored scan
            # without rebuilding a model.
            if not model:
                rows.append(self._histo_row(exp, model, runs))
                continue

            per_run = []
            limits_list = []
            pooled_true, pooled_pred = [], []
            param_names = None

            for run in runs:
                experiment = self._build_experiment(exp, model, run)
                y_true, y_pred = self._test_truth_pred(experiment, exp)

                per_run.append(self._corr_metrics(exp, y_true, y_pred))
                limits_list.append(experiment.checkpoints.limits)

                pooled_true.append(y_true)
                pooled_pred.append(y_pred)

                if param_names is None and experiment.checkpoints.limits is not None:
                    param_names = list(experiment.checkpoints.limits.param_names)

                # Free GPU/host memory before rebuilding the next seed's model.
                del experiment

            row = self._aggregate_runs(model, exp, per_run, param_names, limits_list)
            rows.append(row)

            # Pooled scatter plot (all seeds) for this model.
            self._plot_correlation(
                exp=exp,
                model=model,
                param_names=param_names,
                y_true=np.concatenate(pooled_true, axis=0),
                y_pred=np.concatenate(pooled_pred, axis=0),
                to=out_dir / f"{plot_stem}_corr_{model}.png",
            )

        if rows:
            self._write_correlation_table(rows, out_dir, plot_stem)

    def _aggregate_runs(self, model: str, exp: str, per_run, param_names, limits_list) -> Dict:
        """Reduce per-seed metrics to mean ± std for one model.

        ρ and MSE are averaged over seeds; the MLE bias is taken from the
        seed-averaged LLR surface (see :meth:`_mle_bias_aggregate`) so it matches
        the contour plots.
        """
        rho = np.asarray([r["rho"] for r in per_run], dtype=float)  # (n_runs, n_comp)
        mse = np.asarray([r["mse"] for r in per_run], dtype=float)
        central, std, unconstrained = self._mle_bias_aggregate(limits_list)
        return {
            "label": MODEL2LABEL[model],
            "kind": per_run[0]["kind"],
            "param_names": param_names,
            "n_runs": len(per_run),
            "rho_mean": np.nanmean(rho, axis=0),
            "rho_std": np.nanstd(rho, axis=0),
            "mse_mean": float(np.nanmean(mse)),
            "mse_std": float(np.nanstd(mse)),
            "mle_mean": central,
            "mle_std": std,
            "mle_unconstrained": unconstrained,
        }

    @staticmethod
    def _plot_correlation(*, exp, model, param_names, y_true, y_pred, to) -> None:
        """Save the pooled truth-vs-prediction scatter for one model."""
        if exp == "ratio":
            plot_ratio_calibration(y_true.ravel(), y_pred.ravel(), to=str(to))
        else:
            plot_score_calibration(
                y_true, y_pred, param_names=param_names, to=str(to)
            )

    def _write_correlation_table(self, rows, out_dir: Path, plot_stem: str) -> None:
        """Write the correlation summary as markdown (logged) and LaTeX."""

        def fmt(mean: float, std: float, math: bool = False, signed: bool = False) -> str:
            # `signed` forces an explicit +/- on the mean — used for the MLE bias
            # so the direction of the offset is unambiguous.
            m = f"{mean:+.3f}" if signed else f"{mean:.3f}"
            return f"${m} \\pm {std:.3f}$" if math else f"{m}±{std:.3f}"

        has_score = any(r["kind"] == "score" for r in rows)
        has_logr = any(r["kind"] == "logr" for r in rows)
        # Wilson-coefficient order is shared across score models (one ρ each).
        score_params = next(
            (r["param_names"] for r in rows if r["kind"] == "score" and r["param_names"]),
            None,
        )
        # MLE bias is reported per coefficient for every method (incl. histogram),
        # so its column order comes from whichever row carries param names.
        bias_params = next((r["param_names"] for r in rows if r["param_names"]), None)
        if bias_params is None:
            bias_params = [
                f"param_{i}" for i in range(len(self.cfg.limits.asimov.theta_true))
            ]
        n_seeds = max(r["n_runs"] for r in rows)

        def score_cells(row, math: bool):
            cells = []
            if row["kind"] == "score" and score_params is not None:
                for i in range(len(score_params)):
                    cells.append(fmt(row["rho_mean"][i], row["rho_std"][i], math))
            elif score_params is not None:
                cells = ["—"] * len(score_params)
            return cells

        def logr_cell(row, math: bool):
            if row["kind"] == "logr":
                return [fmt(row["rho_mean"][0], row["rho_std"][0], math)]
            return ["—"]

        def mse_cell(row, math: bool):
            # No per-event prediction for the histogram → MSE undefined.
            if row["kind"] == "histo" or not np.isfinite(row["mse_mean"]):
                return "—"
            return fmt(row["mse_mean"], row["mse_std"], math)

        def mle_cells(row, math: bool):
            # Directions the data doesn't bound get flagged rather than printed
            # as a number — their MLE rails to the grid edge and is not a bias.
            flags = row.get("mle_unconstrained")
            cells = []
            for i in range(len(bias_params)):
                if flags is not None and i < len(flags) and bool(flags[i]):
                    cells.append("unconstr.")
                else:
                    cells.append(
                        fmt(row["mle_mean"][i], row["mle_std"][i], math, signed=True)
                    )
            return cells

        # --- Markdown ---
        header = ["Model"]
        if has_score and score_params is not None:
            header += [f"ρ({PARAM2PLAIN.get(p, p)})" for p in score_params]
        if has_logr:
            header += ["log r corr."]
        header += ["MSE"]
        header += [f"bias({PARAM2PLAIN.get(p, p)})" for p in bias_params]

        md_lines = [
            f"# Truth-vs-prediction correlations (ρ = mean ± std over {n_seeds} seeds)",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for r in rows:
            cells = [r["label"]]
            if has_score:
                cells += score_cells(r, math=False)
            if has_logr:
                cells += logr_cell(r, math=False)
            cells += [mse_cell(r, math=False)]
            cells += mle_cells(r, math=False)
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines += [
            "",
            "_MLE bias = θ̂ − θ_true, where θ̂ is the minimum of each coefficient's "
            "averaged 1D marginal of the seed-averaged LLR (the same marginal the "
            "plots show), parabola-refined sub-grid; ± is the per-seed spread. "
            "`unconstr.` marks coefficients whose 68% marginal reaches the grid "
            "edge — the data does not bound them within the scan range._",
        ]
        md = "\n".join(md_lines) + "\n"

        md_path = out_dir / f"{plot_stem}_correlations.md"
        md_path.write_text(md)
        # Emit the table exactly once. (Logging *and* printing it, with logger
        # propagation to the root handler, is what produced duplicate tables.)
        print(md)
        LOGGER.info("Wrote correlation summary to %s", md_path)

        # --- LaTeX ---
        # PARAM2LABEL values are already $-wrapped (e.g. "$c_{Ht}$"); strip the
        # delimiters so they nest cleanly inside $\rho(...)$ / bias headers.
        def tex_label(p):
            return PARAM2LABEL.get(p, p).strip("$")

        tex_header = ["Model"]
        if has_score and score_params is not None:
            tex_header += [rf"$\rho({tex_label(p)})$" for p in score_params]
        if has_logr:
            tex_header += [r"$\rho(\log r)$"]
        tex_header += ["MSE"]
        tex_header += [
            rf"$\hat{{\theta}}_{{{tex_label(p)}}} - \theta^{{\rm true}}$"
            for p in bias_params
        ]

        tex_lines = [
            r"\begin{tabular}{l" + "c" * (len(tex_header) - 1) + "}",
            r"\toprule",
            " & ".join(tex_header) + r" \\",
            r"\midrule",
        ]
        for r in rows:
            cells = [r["label"]]
            if has_score:
                cells += score_cells(r, math=True)
            if has_logr:
                cells += logr_cell(r, math=True)
            cells += [mse_cell(r, math=True)]
            cells += mle_cells(r, math=True)
            tex_lines.append(" & ".join(cells) + r" \\")
        tex_lines += [r"\bottomrule", r"\end{tabular}", ""]
        (out_dir / f"{plot_stem}_correlations.tex").write_text("\n".join(tex_lines))

    def __call__(self):
        self.run()
