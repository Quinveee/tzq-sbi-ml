from itertools import combinations
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ..logger import LOGGER as _LOGGER

mh.style.use("ATLAS")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15

PARAM2LABEL = {"cHt": r"$c_{Ht}$", "ctWRe": r"$c_{tW}$", "ctBRe": r"$c_{tB}$"}

# Per-coefficient x-range for the limits panels (falls back to ±0.4)
PARAM2LIMITXLIM = {
    "cHt": (-0.2, 0.2),
    "ctWRe": (-0.7, 0.7),
    "ctBRe": (-1, 1),
}

LOGGER = _LOGGER.getChild(__name__)


def plot_llr(
    *,
    llr_list: List[np.ndarray],
    std_list: Optional[List[np.ndarray]] = None,
    runs_list: Optional[List[np.ndarray]] = None,
    grid: np.ndarray,
    param_names: List[str],
    ranges: List[Tuple[float, float]],
    resolutions: List[int],
    labels: List[str],
    colors=None,
    linestyles=None,
    conf_levels=(0.68, 0.95),
    to=None,
    mode: Literal["average", "slice", "mle"] = "average",
    plot_marginals: bool = True,
    method: Optional[str] = None,
) -> None:
    """
    Plot N-dimensional contours of the LLR

    :param llr_list: Description
    :type llr_list: List[np.ndarray]
    :param std_list: Description
    :type std_list: List[np.ndarray]
    :param grid: Description
    :type grid: np.ndarray
    :param param_names: Description
    :type param_names: List[str]
    :param ranges: Description
    :type ranges: List[Tuple[float, float]]
    :param resolutions: Description
    :type resolutions: List[int]
    :param labels: Description
    :type labels: List[str]
    :param colors: Description
    :param linestyles: Description
    :param conf_levels: Description
    :param to: Description
    :param mode: Description
    :type mode: Literal["average", "slice", "mle"]
    """
    assert len(llr_list) == len(labels), f"Number of LLR arrays and labels don't match"

    # `grid` has shape (n_grid_points, n_dimensions)
    N = grid.shape[1]
    if N < 2:
        return _plot_llr_1d(
            llr_list=llr_list,
            std_list=std_list,
            x_range=ranges[-1],
            grid=grid,
            param_name=param_names[0],
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            to=to,
            method=method,
        )
    if N == 3:
        # Single combined 3x3 figure: 2D projections, marginals, and limits.
        return _plot_llr_combined_3d(
            llr_list=llr_list,
            std_list=std_list,
            runs_list=runs_list,
            grid=grid,
            param_names=param_names,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            conf_levels=conf_levels,
            mode=mode,
            to=to,
            method=method,
        )
    D = resolutions

    pairs = list(combinations(range(N), 2))
    ncols = len(pairs)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Default styles
    if colors is None:
        colors = plt.cm.tab10.colors[: len(llr_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(llr_list) // 4) + 1)

    # Precompute χ² levels for each confidence probability (2D case)
    chi2_levels = [stats.chi2.ppf(p, 2) for p in conf_levels]

    # Build legend handles for all methods
    handles = [
        mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lab)
        for c, ls, lab in zip(colors, linestyles, labels)
    ]

    # Iterate over all possible 2d slices
    for ax, (i, j) in zip(axes, pairs):

        # For each slice, plot one by one all available LLR arrays
        for llr, color, ls in zip(llr_list, colors, linestyles):
            values_nd = -2 * llr.reshape(resolutions)
            others = [k for k in range(N) if k not in (i, j)]

            # Select what to do with the disregarded dimensions
            if mode == "average":
                data_2d = values_nd.mean(axis=tuple(others))
            elif mode == "slice":
                fixed_index = D // 2
                slicer = [slice(None)] * N
                for k in others:
                    slicer[k] = fixed_index
                data_2d = values_nd[tuple(slicer)]
            elif mode == "mle":
                data_2d = values_nd.min(axis=tuple(others))
            else:
                raise ValueError("mode must be 'average' or 'slice'")

            # Rescale the resulting LLR slice
            data_2d -= data_2d.min()
            xi = np.unique(grid[:, i])
            yj = np.unique(grid[:, j])
            X, Y = np.meshgrid(xi, yj, indexing="ij")

            # Plot contours for different confidence levels
            for lvl, alpha in zip(chi2_levels, np.linspace(1.0, 0.4, len(chi2_levels))):
                ax.contour(
                    X,
                    Y,
                    data_2d,
                    levels=[lvl],
                    colors=[color],
                    linestyles=[ls],
                    linewidths=1.75,
                    alpha=alpha,
                )

        # Set labels for the 2d slice with the considered parameters
        ax.set_xlabel(PARAM2LABEL.get(param_names[i], param_names[i]))
        ax.set_ylabel(PARAM2LABEL.get(param_names[j], param_names[j]))
        ax.set_title(
            f"({PARAM2LABEL.get(param_names[i], param_names[i])}, {PARAM2LABEL.get(param_names[j], param_names[j])}) projection",
            fontsize=20,
        )

    # Shared legend on top — wrap to multiple rows so wide label sets
    # don't stretch the figure
    legend_ncol = min(len(labels), 3)
    legend_nrows = int(np.ceil(len(labels) / legend_ncol))

    legend_height = 0.06 * legend_nrows
    legend_top = 1.0 - 0.01
    axes_top = legend_top - legend_height

    if method:
        fig.suptitle(method, fontsize=20, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, axes_top - 0.03])
    else:
        fig.tight_layout(rect=[0, 0, 1, axes_top])

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top),
        ncol=legend_ncol,
        frameon=False,
        fontsize=20,
    )

    fig.text(
        0.5,
        0.0,
        r"$\sqrt{s}=13.6$ TeV~~$L=300\,\mathrm{fb}^{-1}$",
        ha="center",
        va="bottom",
        fontsize=20,
    )

    # Hide any unused axes (only present if len(axes) > len(pairs))
    for unused in axes[len(pairs):]:
        unused.set_visible(False)

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_" + "_".join(param_names)),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    if plot_marginals and N >= 2:
        _plot_marginals(
            llr_list=llr_list,
            grid=grid,
            param_names=param_names,
            ranges=ranges,
            resolutions=resolutions,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            conf_levels=conf_levels,
            mode=mode,
            to=to,
            method=method,
        )


def _plot_marginals(
    *,
    llr_list: List[np.ndarray],
    grid: np.ndarray,
    param_names: List[str],
    ranges: List[Tuple[float, float]],
    resolutions: List[int],
    labels: List[str],
    colors=None,
    linestyles=None,
    conf_levels=(0.68, 0.95),
    mode: Literal["average", "slice", "mle"] = "average",
    to=None,
    method: Optional[str] = None,
) -> None:
    """
    Plot 1D marginal profile LLR curves for each parameter, profiling/averaging
    over all other dimensions.
    """
    N = grid.shape[1]

    if colors is None:
        colors = plt.cm.tab10.colors[: len(llr_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(llr_list) // 4) + 1)

    chi2_levels_1d = [stats.chi2.ppf(p, 1) for p in conf_levels]

    ncols = min(3, N)
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    handles = [
        mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lab)
        for c, ls, lab in zip(colors, linestyles, labels)
    ]

    for ax, i in zip(axes, range(N)):
        others = tuple(k for k in range(N) if k != i)
        xi = np.unique(grid[:, i])
        x_range = ranges[i]

        for llr, color, ls in zip(llr_list, colors, linestyles):
            values_nd = -2 * llr.reshape(resolutions)

            if mode == "mle":
                profile = values_nd.min(axis=others)
            elif mode == "average":
                profile = values_nd.mean(axis=others)
            elif mode == "slice":
                slicer = [resolutions[k] // 2 if k in others else slice(None) for k in range(N)]
                profile = values_nd[tuple(slicer)]
            else:
                raise ValueError("mode must be 'average', 'slice', or 'mle'")

            profile = profile - profile.min()
            ax.plot(xi, profile, color=color, linestyle=ls, linewidth=2)

        # Confidence level lines
        for lvl, chi2_val in zip(conf_levels, chi2_levels_1d):
            ax.axhline(chi2_val, color="grey", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                x_range[-1] - 0.05 * (x_range[-1] - x_range[0]),
                chi2_val + 0.3,
                f"{int(lvl * 100)}\\% CI",
                color="grey",
                ha="right",
                fontsize=12,
            )

        ax.set_xlabel(PARAM2LABEL.get(param_names[i], param_names[i]))
        ax.set_ylabel(r"$-2\log\Lambda$")
        ax.set_xlim(x_range)
        ax.set_ylim(0, max(chi2_levels_1d[-1] * 1.5, 5))

    # Hide unused axes
    for ax in axes[N:]:
        ax.set_visible(False)

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=16,
    )
    _title = "Marginal profile likelihood"
    if method:
        _title = f"{method} — {_title}"
    fig.suptitle(_title, fontsize=18, y=1.01)
    fig.tight_layout()

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_marginals"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _reduce_llr(values_nd, others, mode, resolutions, N):
    """Reduce an N-d ``-2logΛ`` array over the ``others`` axes to a lower-d
    profile, matching the convention used across the LLR plots."""
    others = tuple(others)
    if mode == "mle":
        return values_nd.min(axis=others)
    if mode == "average":
        return values_nd.mean(axis=others)
    if mode == "slice":
        slicer = [
            resolutions[k] // 2 if k in others else slice(None) for k in range(N)
        ]
        return values_nd[tuple(slicer)]
    raise ValueError("mode must be 'average', 'slice', or 'mle'")


def _plot_llr_combined_3d(
    *,
    llr_list: List[np.ndarray],
    std_list: Optional[List[np.ndarray]] = None,
    runs_list: Optional[List[np.ndarray]] = None,
    grid: np.ndarray,
    param_names: List[str],
    ranges: List[Tuple[float, float]],
    resolutions: List[int],
    labels: List[str],
    colors=None,
    linestyles=None,
    conf_levels=(0.68, 0.95),
    mode: Literal["average", "slice", "mle"] = "average",
    to=None,
    method: Optional[str] = None,
) -> None:
    """
    Combined 3D figure laid out as a 3x3 grid sharing one title and model
    legend:
      - top row    : the three 2D contour projections (pairs of coefficients)
      - middle row : the marginal profile-likelihood curve per coefficient
      - bottom row : the confidence intervals beneath each marginal
    """
    N = grid.shape[1]

    if colors is None:
        colors = plt.cm.tab10.colors[: len(llr_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(llr_list) // 4) + 1)

    pairs = list(combinations(range(N), 2))
    # `conf_levels` sets the 2D contour levels (top row); the marginals and
    # limits always show the 68% (solid) and 95% (dashed) CIs like the 1D plot.
    ci_levels = (0.68, 0.95)
    chi2_levels_2d = [stats.chi2.ppf(p, 2) for p in conf_levels]
    chi2_levels_1d = [stats.chi2.ppf(p, 1) for p in ci_levels]

    grid_axes = [np.unique(grid[:, d]) for d in range(N)]

    fig = plt.figure(figsize=(16, 16))
    # A small outer gap puts the header right above the plots; the plot rows
    # keep their own (larger) spacing via the inner grid.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.2, 10.2], hspace=0.05)

    # --- Shared header: title + model legend ---
    ax_head = fig.add_subplot(outer[0])
    ax_head.axis("off")
    if method:
        ax_head.set_title(method, fontsize=30, pad=14)

    handles = [
        mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lab)
        for c, ls, lab in zip(colors, linestyles, labels)
    ]
    ax_head.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        fontsize=22,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.5,
    )

    gs = outer[1].subgridspec(
        3,
        len(pairs),
        height_ratios=[4, 4, 2.2],
        hspace=0.45,
        wspace=0.25,
    )

    # --- Top row: 2D contour projections ---
    for col, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, col])
        others = [k for k in range(N) if k not in (i, j)]
        xi = grid_axes[i]
        yj = grid_axes[j]
        X, Y = np.meshgrid(xi, yj, indexing="ij")

        for llr, color, ls in zip(llr_list, colors, linestyles):
            values_nd = -2 * llr.reshape(resolutions)
            data_2d = _reduce_llr(values_nd, others, mode, resolutions, N)
            data_2d = data_2d - data_2d.min()
            for lvl, alpha in zip(
                chi2_levels_2d, np.linspace(1.0, 0.4, len(chi2_levels_2d))
            ):
                ax.contour(
                    X,
                    Y,
                    data_2d,
                    levels=[lvl],
                    colors=[color],
                    linestyles=[ls],
                    linewidths=1.75,
                    alpha=alpha,
                )

        li = PARAM2LABEL.get(param_names[i], param_names[i])
        lj = PARAM2LABEL.get(param_names[j], param_names[j])
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"({li}, {lj}) projection", fontsize=18)

    # --- Middle row: marginal profile-likelihood curves ---
    for col in range(N):
        ax = fig.add_subplot(gs[1, col])
        others = tuple(k for k in range(N) if k != col)
        xi = grid_axes[col]
        x_range = ranges[col]

        for llr, color, ls in zip(llr_list, colors, linestyles):
            values_nd = -2 * llr.reshape(resolutions)
            profile = _reduce_llr(values_nd, others, mode, resolutions, N)
            profile = profile - profile.min()
            ax.plot(xi, profile, color=color, linestyle=ls, linewidth=2)

        for lvl, chi2_val in zip(ci_levels, chi2_levels_1d):
            ax.axhline(chi2_val, color="grey", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                x_range[-1] - 0.05 * (x_range[-1] - x_range[0]),
                chi2_val + 0.3,
                f"{int(lvl * 100)}\\% CI",
                color="grey",
                ha="right",
                fontsize=12,
            )

        ax.set_xlabel(PARAM2LABEL.get(param_names[col], param_names[col]))
        ax.set_ylabel(r"$-2\log\Lambda$")
        ax.set_xlim(x_range)
        ax.set_ylim(0, max(chi2_levels_1d[-1] * 1.5, 5))

    # --- Bottom row: confidence intervals beneath each marginal ---
    n_models = len(llr_list)
    row_height = 0.1
    for col in range(N):
        ax = fig.add_subplot(gs[2, col])
        others = tuple(k for k in range(N) if k != col)
        xi = grid_axes[col]

        for m, (llr, color) in enumerate(zip(llr_list, colors)):
            values_nd = -2 * llr.reshape(resolutions)
            profile = _reduce_llr(values_nd, others, mode, resolutions, N)
            profile = profile - profile.min()
            theta_mle, intervals = _find_ci_intervals(xi, profile, levels=ci_levels)
            y = (n_models - 1 - m) * row_height
            ax.scatter(theta_mle, y, color=color)
            for interval, style in zip(intervals.values(), ["solid", "dashed"]):
                ax.hlines(y, interval[0], interval[1], linestyles=style, color=color)

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_yticks([(n_models - 1) * row_height / 2])
        ax.set_yticklabels([PARAM2LABEL.get(param_names[col], param_names[col])])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(*PARAM2LIMITXLIM.get(param_names[col], (-0.4, 0.4)))
        ax.set_ylim(-0.1, (n_models - 1) * row_height + 0.1)
        for spine in ("right", "top", "left"):
            ax.spines[spine].set_visible(False)

    fig.text(
        1.0,
        0.5,
        r"$\sqrt{s}=13.6$ TeV~~$L=300\,\mathrm{fb}^{-1}$",
        rotation=270,
        ha="left",
        va="center",
        fontsize=22,
    )

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_" + "_".join(param_names)),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_llr_1d(
    *,
    llr_list: List[np.ndarray],
    std_list: List[np.ndarray] = None,
    grid: np.ndarray,
    param_name: str,
    labels: List[str],
    x_range: Tuple[float, float],
    colors=None,
    linestyles=None,
    to=None,
    levels=(0.68, 0.95),
    ylim=(0, 30),
    method: Optional[str] = None,
) -> None:
    """
    Combined 1D figure: the marginal profile-likelihood curves (left) and the
    corresponding confidence intervals per model (right), sharing a single
    title and model legend.
    """
    assert len(llr_list) == len(labels), f"Number of LLR arrays and labels don't match"

    # Set defaults
    if colors is None:
        colors = plt.cm.tab10.colors[: len(llr_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(llr_list) // 4) + 1)

    param_label = PARAM2LABEL.get(param_name, param_name)
    std_list = std_list if std_list else [None] * len(llr_list)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, height_ratios=[1, 5], hspace=0.05, wspace=0.18)

    # --- Shared header: title + model legend spanning both panels ---
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    if method:
        ax_head.set_title(method, fontsize=22, pad=8)

    handles = [
        mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lab)
        for c, ls, lab in zip(colors, linestyles, labels)
    ]
    ax_head.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=False,
        fontsize=18,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.5,
    )

    # --- Left panel: marginal profile-likelihood curves ---
    ax_curve = fig.add_subplot(gs[1, 0])
    for llr, std, color, ls in zip(llr_list, std_list, colors, linestyles):
        x = grid[:, 0]
        y = -2 * llr
        ax_curve.plot(x, y, color=color, linestyle=ls, linewidth=1.5)
        if std is not None:
            ax_curve.fill_between(x, y + 2 * std, y - 2 * std, color=color, alpha=0.3)

    for level in levels:
        value = stats.chi2.ppf(level, 1)
        ax_curve.hlines(value, *x_range, colors="grey", linestyles="--", alpha=0.5)
        ax_curve.text(
            x=x_range[-1] - 0.33,
            y=value + 0.3,
            s=f"{int(level * 100)}\\% CI",
            color="grey",
        )

    ax_curve.set_ylim(*ylim)
    ax_curve.set_xlim(x_range)
    ax_curve.set_xlabel(param_label)
    ax_curve.set_ylabel(r"$-2\log\Lambda$")
    _lumi_label = r"$\sqrt{s}=13.6$ TeV" + "\n" + r"$L=300~\mathrm{fb}^{-1}$"
    ax_curve.text(
        0.9 * ax_curve.get_xlim()[0],
        0.8 * ax_curve.get_ylim()[-1],
        _lumi_label,
        fontsize=18,
    )

    # --- Right panel: confidence intervals per model ---
    ax_int = fig.add_subplot(gs[1, 1])
    n_models = len(llr_list)
    row_height = 0.1
    for m, (llr, color) in enumerate(zip(llr_list, colors)):
        profile = -2 * llr
        profile = profile - profile.min()
        theta_mle, intervals = _find_ci_intervals(grid[:, 0], profile, levels=levels)
        y = (n_models - 1 - m) * row_height

        ax_int.scatter(theta_mle, y, color=color)
        for interval, style in zip(intervals.values(), ["solid", "dashed"]):
            ax_int.hlines(y, interval[0], interval[1], linestyles=style, color=color)

    ax_int.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax_int.set_yticks([(n_models - 1) * row_height / 2])
    ax_int.set_yticklabels([param_label])
    ax_int.tick_params(axis="y", length=0)
    ax_int.set_xlim(-0.2, 0.2)
    ax_int.set_ylim(-0.1, (n_models - 1) * row_height + 0.1)
    for spine in ("right", "top", "left"):
        ax_int.spines[spine].set_visible(False)

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_{param_name}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    return fig, (ax_curve, ax_int)


def plot_ratio_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 50,
    to=None,
) -> plt.Figure:
    """True vs estimated log ratio with frequency colorbar and pull scatter.

    Top panel: 2D histogram of (true, estimated) log r with a colorbar showing
    event density, plus a diagonal y=x reference line (equal aspect).

    Bottom panel: scatter of the standardized residual
    (log r - log r_hat) / sigma_{log r_hat}, where sigma is the global std of
    (log r_hat - log r) over all events.
    """
    xlabel = r"True $\log r(x)$"
    ylabel = r"Estimated $\log \hat{r}(x)$"

    fig = plt.figure(figsize=(7, 9))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4, figure=fig)
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1])

    # --- Top: scatter with diagonal reference. Use percentile-based limits so
    # a handful of outliers don't compress the bulk into a tiny corner and
    # make a moderate-correlation cloud look "way off" the diagonal.
    lo = float(np.percentile(np.concatenate([y_true, y_pred]), 0.5))
    hi = float(np.percentile(np.concatenate([y_true, y_pred]), 99.5))
    ax_main.scatter(y_true, y_pred, s=2, color="steelblue", alpha=0.3, edgecolor="none")
    ax_main.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.8, label="$y = x$")
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    rho = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    ax_main.set_title(rf"Pearson $\rho = {rho:.3f}$", fontsize=10)
    ax_main.legend(fontsize=10, loc="upper left")

    # --- Bottom: standardized residual scatter ---
    residual = y_true - y_pred
    sigma = float(np.std(y_pred - y_true))
    pull = residual / sigma if sigma > 0 else residual
    ax_resid.scatter(y_true, pull, s=2, color="steelblue", alpha=0.3, edgecolor="none")
    ax_resid.axhline(0.0, color="r", linestyle="--", linewidth=1.0)
    ax_resid.set_xlim(lo, hi)
    ax_resid.set_xlabel(xlabel)
    ax_resid.set_ylabel(r"$(\log r - \log \hat{r}) / \sigma_{\log \hat{r}}$")

    if to is not None:
        fig.savefig(to, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig


def plot_score_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    param_names=None,
    n_bins: int = 50,
    to=None,
) -> plt.Figure:
    """True vs estimated score per component with frequency colorbar and pull scatter.

    For multi-dimensional scores, each component gets its own column. The layout
    mirrors plot_ratio_calibration: a 2D density histogram on top with a diagonal
    y=x reference (equal aspect), and a standardized-residual scatter below
    using the per-component global std of (t_hat - t) as sigma.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    n_dims = y_true.shape[1]
    if param_names is None:
        param_names = [f"dim {d}" for d in range(n_dims)]

    fig = plt.figure(figsize=(5 * n_dims, 9))
    gs = GridSpec(2, n_dims, height_ratios=[3, 1], hspace=0.4, figure=fig)

    for d in range(n_dims):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        pname = PARAM2LABEL.get(param_names[d], param_names[d])
        xlabel = rf"True score $t$ ({pname})"
        ylabel = rf"Estimated score $\hat{{t}}$ ({pname})"

        ax_main = fig.add_subplot(gs[0, d])
        ax_resid = fig.add_subplot(gs[1, d])

        # Top: scatter with diagonal reference. Use percentile-based limits so
        # a handful of outliers don't compress the bulk into a tiny corner and
        # make a moderate-correlation cloud look "way off" the diagonal.
        lo = float(np.percentile(np.concatenate([yt, yp]), 0.5))
        hi = float(np.percentile(np.concatenate([yt, yp]), 99.5))
        ax_main.scatter(yt, yp, s=2, color="steelblue", alpha=0.3, edgecolor="none")
        ax_main.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.8, label="$y = x$")
        ax_main.set_xlim(lo, hi)
        ax_main.set_ylim(lo, hi)
        ax_main.set_aspect("equal", adjustable="box")
        ax_main.set_xlabel(xlabel)
        ax_main.set_ylabel(ylabel)
        rho = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 else float("nan")
        ax_main.set_title(rf"Pearson $\rho = {rho:.3f}$", fontsize=10)
        ax_main.legend(fontsize=9, loc="upper left")

        # Bottom: standardized residual scatter
        residual = yt - yp
        sigma = float(np.std(yp - yt))
        pull = residual / sigma if sigma > 0 else residual
        ax_resid.scatter(yt, pull, s=2, color="steelblue", alpha=0.3, edgecolor="none")
        ax_resid.axhline(0.0, color="r", linestyle="--", linewidth=1.0)
        ax_resid.set_xlim(lo, hi)
        ax_resid.set_xlabel(xlabel)
        ax_resid.set_ylabel(r"$(t - \hat{t}) / \sigma_{\hat{t}}$" if d == 0 else "")

    if to is not None:
        fig.savefig(to, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig


def plot_learning_curves(losses, to=None):
    epochs = np.arange(len(losses.train))

    fig, ax = plt.subplots()

    ax.plot(epochs, losses.train, label="train")
    ax.plot(epochs, losses.val, label="val")

    ax.legend()

    fig.tight_layout()

    if to is not None:
        fig.savefig(to)

    return fig, ax


def plot_intervals(
    llr_list,
    grid,
    labels,
    to,
    colors,
    resolutions=None,
    param_names=None,
    mode: Literal["average", "slice", "mle"] = "average",
    method=None,
):
    # For an N-dimensional scan the LLR is reduced over the other coefficients
    # to obtain a 1D profile per coefficient, and each coefficient gets its own
    # block of model rows. `mode` must match what `plot_llr`/`_plot_marginals`
    # use so the intervals line up with the marginal curves. The 1D case falls
    # out as N == 1.
    N = grid.shape[1]
    if resolutions is None:
        resolutions = [grid.shape[0]]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(N)]

    grid_axes = [np.unique(grid[:, d]) for d in range(N)]

    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)

    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")

    if method:
        ax_top.set_title(method, fontsize=22, pad=8)

    handles = [Line2D([0], [0], color=c, lw=3) for c in colors]

    # Lay the model labels out in a horizontal grid (multiple columns) so a
    # large ensemble doesn't stack into one tall column that overflows the
    # short header panel and gets clipped. Anchored to the bottom of the header
    # strip so it sits just above the plot; the lumi / CoM-energy annotation
    # lives inside the main axes.
    legend_ncol = min(len(labels), 3)
    ax_top.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=18,
        columnspacing=1.5,
        handlelength=1.6,
        handletextpad=0.5,
    )

    ax = fig.add_subplot(gs[1])

    n_models = len(llr_list)
    row_height = 0.1
    block_height = n_models * row_height
    block_gap = 0.12
    block_pitch = block_height + block_gap

    yticks = []
    yticklabels = []

    for p in range(N):
        others = tuple(k for k in range(N) if k != p)
        xi = grid_axes[p]

        # Coefficients run top-to-bottom; within a block, models run top-to-bottom
        # in the same order as the legend.
        block_base = (N - 1 - p) * block_pitch

        for m, (llr, color) in enumerate(zip(llr_list, colors)):
            values_nd = (-2 * llr).reshape(resolutions)

            # Reduce over the other coefficients exactly as the marginals plot
            # does, so the intervals match the marginal curves.
            if mode == "mle":
                profile = values_nd.min(axis=others)
            elif mode == "average":
                profile = values_nd.mean(axis=others)
            elif mode == "slice":
                slicer = [
                    resolutions[k] // 2 if k in others else slice(None)
                    for k in range(N)
                ]
                profile = values_nd[tuple(slicer)]
            else:
                raise ValueError("mode must be 'average', 'slice', or 'mle'")

            profile = profile - profile.min()

            theta_mle, intervals = _find_ci_intervals(xi, profile)
            y = block_base + (n_models - 1 - m) * row_height

            ax.scatter(theta_mle, y, color=color)
            for interval, style in zip(intervals.values(), ["solid", "dashed"]):
                ax.hlines(y, interval[0], interval[1], linestyles=style, color=color)

        yticks.append(block_base + (n_models - 1) * row_height / 2)
        yticklabels.append(PARAM2LABEL.get(param_names[p], param_names[p]))

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="y", length=0)

    xmin, xmax = -0.4, 0.4
    ax.set_xlim(xmin, xmax)
    # Reserve headroom above the topmost block for the lumi annotation so it
    # doesn't overlap the interval lines.
    top_y = (N - 1) * block_pitch + (n_models - 1) * row_height
    ax.set_ylim(-0.1, top_y + 0.45)

    ax.text(
        xmin + 0.02 * (xmax - xmin),
        top_y + 0.40,
        r"$\sqrt{s} = 13.6$ TeV" + "\n" + r"$L=300~\mathrm{fb}^{-1}$",
        fontsize=18,
        va="top",
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.savefig(to, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _find_ci_intervals(x, y, levels=(0.68, 0.95)):
    """
    Given arrays x and y = -2 log LLR(x), return:
    - x_mle : x-value at the minimum LLR
    - intervals : dict of {level: (low, high)}
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_mle = x[np.argmin(y)]

    intervals = {}
    for level in levels:
        threshold = stats.chi2.ppf(level, 1)

        # Identify where y crosses the threshold
        mask = y >= threshold
        idx = np.where(mask[:-1] != mask[1:])[0]

        if len(idx) == 0:
            intervals[level] = (np.nan, np.nan)
            continue

        # Compute all crossing points
        crossings = []
        for i in idx:
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            slope = (y2 - y1) / ((x2 - x1) + 1e-8)
            crossings.append(x1 + (threshold - y1) / slope)

        crossings = np.sort(crossings)

        # If only one crossing, form a one-sided interval
        if len(crossings) == 1:
            xc = crossings[0]
            if xc < x_mle:
                intervals[level] = (xc, np.max(x))
            else:
                intervals[level] = (np.min(x), xc)
        else:
            # Take the outermost crossings
            intervals[level] = (crossings[0], crossings[-1])

    return x_mle, intervals


