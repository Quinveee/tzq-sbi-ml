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

LOGGER = _LOGGER.getChild(__name__)


def plot_llr(
    *,
    llr_list: List[np.ndarray],
    std_list: Optional[List[np.ndarray]] = None,
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
            param_name=param_names.pop(),
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            to=to,
        )
    D = resolutions

    pairs = list(combinations(range(N), 2))
    ncols = min(2, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
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
        ax.set_xlabel(PARAM2LABEL[param_names[i]])
        ax.set_ylabel(PARAM2LABEL[param_names[j]])
        ax.set_title(
            f"({PARAM2LABEL[param_names[i]]}, {PARAM2LABEL[param_names[j]]}) projection",
            fontsize=20,
        )

    # Shared legend on top
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=20,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.89])

    ax.text(
        2.30,
        1.05,  # x, y position
        r"$\sqrt{s}=13.6$ Tev~~$L=300\mathrm{fb}^{-1}$",  # text
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
    )

    axes[-1].set_visible(False)
    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_" + "_".join(param_names)), dpi=300
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
        ncol=len(labels),
        frameon=False,
        fontsize=16,
    )
    fig.suptitle("Marginal profile likelihood", fontsize=18, y=1.01)
    fig.tight_layout()

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_marginals"), dpi=300
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
) -> None:
    """
    Plot 1 dimensional contours of the LLR

    :param llr_list: Description
    :type llr_list: List[np.ndarray]
    :param std_list: Description
    :type std_list: Optional[List[np.ndarray]]
    :param grid: Description
    :type grid: np.ndarray
    :param param_name: Description
    :type param_name: str
    :param labels: Description
    :type labels: List[str]
    :param x_range: Description
    :type x_range: Tuple[float, float]
    :param colors: Description
    :param linestyles: Description
    :param to: Description
    :param levels: Description
    :param ylim: Description
    """
    assert len(llr_list) == len(labels), f"Number of LLR arrays and labels don't match"

    # Set defaults
    if colors is None:
        colors = plt.cm.tab10.colors[: len(llr_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(llr_list) // 4) + 1)

    param_label = PARAM2LABEL[param_name]

    fig, ax = plt.subplots()

    # Allow for not plotting std of the LLR over runs
    std_list = std_list if std_list else [None] * len(llr_list)

    # Plot the different LLR 1d contours
    for llr, std, label, color, ls in zip(
        llr_list, std_list, labels, colors, linestyles
    ):
        x = grid[:, 0]
        y = -2 * llr
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2, label=label)
        if std is not None:
            ax.fill_between(x, y + std, y - std, color=color, alpha=0.3)

    # Plot horizontal lines for the different confidence levels
    for level in levels:
        value = stats.chi2.ppf(level, 1)
        ax.hlines(
            value,
            *x_range,
            colors="grey",
            linestyles="--",
            alpha=0.5,
        )
        ax.text(
            x=x_range[-1] - 0.33,
            y=value + 0.3,
            s=f"{int(level * 100)}\\% CI",
            color="grey",
        )

    # Plot limits
    ax.set_ylim(*ylim)
    ax.set_xlim(x_range)

    # Labels
    ax.set_xlabel(param_label)
    ax.set_ylabel(r"$-2\log\Lambda$")
    ax.text(
        0.9 * ax.get_xlim()[0],
        0.8 * ax.get_ylim()[-1],
        f'{r"$\sqrt{s}=13.6$ TeV"}\n{r"$L=300~\mathrm{fb}^{-1}$"}',
        fontsize=18,
    )

    ax.legend(frameon=False, fontsize=20)

    fig.tight_layout()

    if to is not None:
        fig.savefig(Path(to).with_stem(f"{Path(to).stem}_{param_name}"), dpi=300)
    plt.close(fig)

    return fig, ax


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


def plot_intervals(llr_list, grid, labels, to, colors):
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.3)

    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")

    ax_top.text(
        0.02,
        0.40,
        r"$\sqrt{s} = 13.6$ TeV",
        fontsize=25,
        transform=ax_top.transAxes,
    )
    ax_top.text(
        0.02,
        0.08,
        r"$L=300~\mathrm{fb}^{-1}$",
        fontsize=25,
        transform=ax_top.transAxes,
    )

    handles = [Line2D([0], [0], color=c, lw=3) for c in colors]

    ax_top.legend(handles, labels, loc="upper right", frameon=False, fontsize=25)

    ax = fig.add_subplot(gs[1])

    for i, (llr, color) in enumerate(zip(llr_list[::-1], colors[::-1])):
        theta_mle, intervals = _find_ci_intervals(grid[:, 0], -2 * llr)
        y = 0.1 * i

        ax.scatter(theta_mle, y, color=color)
        for interval, style in zip(intervals.values(), ["solid", "dashed"]):
            ax.hlines(y, interval[0], interval[1], linestyles=style, color=color)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_yticks([0.2])
    ax.set_yticklabels([r"$c_{H t}$"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, 0.5)

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
            slope = (y2 - y1) / (x2 - x1)
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
