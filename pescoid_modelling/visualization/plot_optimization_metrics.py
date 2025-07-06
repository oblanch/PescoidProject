"""Code to plot trajectories of a single simulation."""

from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters

PARAM_NAMES_FORMAL = [
    "δ",
    "F",
    "Tau_m",
    "Γ",
    "A",
    "β",
    "r",
    "m_sensitivity",
    "morphogen_feedback",
]


def plot_optimization_metrics(
    optim_csv: str = "cma_restart.logxrecentbest.dat",
    axlen_csv: str = "cma_restart.logaxlen.dat",
    stddev_csv: str = "cma_restart.logstddev.dat",
    fit_csv: str = "cma_restart.logfit.dat",
) -> None:
    """Entry point to plot optimization results."""
    _set_matplotlib_publication_parameters()

    it_fit, fitness, sigma, axis_r = _load_fit(fit_csv)
    it_std, stddev = _load_stddev(stddev_csv)
    it_ax, axlen = _load_axis_lengths(axlen_csv)
    it_opt, _, _, params = _load_cma_data(optim_csv)

    plot_fitness_sigma_axis(it_fit, fitness, sigma, axis_r)
    plot_axislen_stddev(it_ax, axlen, it_std, stddev)
    plot_parameter_evolution(it_opt, params, param_names=PARAM_NAMES_FORMAL)


def _load_cma_data(dat_file: str) -> Tuple[list, list, list, np.ndarray]:
    """Load CMA-ES optimization data."""
    col_names = [
        "iteration",
        "evals",
        "sigma",
        "zero_col",
        "fitness",
        "diffusivity",
        "flow",
        "tau_m",
        "gamma",
        "activity",
        "beta",
        "r",
        "m_sensitivity",
        "morphogen_feedback",
    ]

    df = pd.read_csv(
        dat_file,
        header=None,
        skiprows=1,
        sep=r"\s+",
        names=col_names,
    )

    iterations = df["iteration"].tolist()
    fitness = df["fitness"].tolist()
    sigma = df["sigma"].tolist()
    params = df[_ORDER].to_numpy()

    return iterations, fitness, sigma, params


def _load_stddev(dat_file) -> Tuple[list, np.ndarray]:
    """Load coordinate-wise sampling standard deviations normalized by
    sigma."""
    raw = np.loadtxt(dat_file, comments="%")
    iterations = raw[:, 0].astype(int).tolist()
    sigma = raw[:, 2]
    stddev_raw = raw[:, 5:]
    stddev = stddev_raw / sigma[:, None]
    return iterations, stddev


def _load_axis_lengths(dat_file) -> Tuple[list, np.ndarray]:
    """Return iterations and the eigenvalues multiplied by sigma."""
    with open(dat_file) as fh:
        next(fh)
    raw = np.loadtxt(dat_file, comments="%")
    iterations = raw[:, 0].astype(int).tolist()
    sigma = raw[:, 2]
    axlen = raw[:, 5:]
    axlen = sigma[:, None] * axlen
    return iterations, axlen


def _load_fit(dat_file: str) -> Tuple[list, list, list, list]:
    """Return iteration, best fitness, sigma and axis-ratio."""
    raw = np.loadtxt(dat_file, comments="%")
    iters = raw[:, 0].astype(int).tolist()
    sigma = raw[:, 2].tolist()
    axis_ratio = raw[:, 3].tolist()
    fitness_best = raw[:, 5].tolist()
    return iters, fitness_best, sigma, axis_ratio


def plot_fitness_sigma_axis(
    iters: List[int],
    fitness: List[float],
    sigma: List[float],
    axis_ratio: List[float],
    out_path: str = "fit_sigma_axis.svg",
) -> None:
    """Facet the three key scalars into one 1x3 figure."""
    fig, axes = plt.subplots(1, 3, figsize=(4, 1.15), sharex=True)
    for ax in axes:
        ax.margins(x=0.00001, y=0.05)

    axes[0].plot(
        iters,
        fitness,
        marker="o",
        markersize=1.2,
        linewidth=0.6,
        markerfacecolor="white",
        markeredgewidth=0.4,
    )
    axes[0].set_ylabel("Best fitness")

    axes[1].plot(
        iters,
        sigma,
        color="tab:orange",
        marker="o",
        markersize=1.2,
        linewidth=0.6,
        markerfacecolor="white",
        markeredgewidth=0.4,
    )
    axes[1].set_ylabel("σ (step-size)")
    axes[1].set_xlabel("Iterations")

    axes[2].plot(
        iters,
        axis_ratio,
        color="tab:green",
        marker="o",
        markersize=1.2,
        linewidth=0.6,
        markerfacecolor="white",
        markeredgewidth=0.4,
    )
    axes[2].set_ylabel("Axis ratio")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_axislen_stddev(
    it_ax: List[int],
    axlen: np.ndarray,
    it_std: List[int],
    stddev: np.ndarray,
    out_path: str = "axis_stddev_facet.svg",
) -> None:
    """Facet axis lengths and stddev side-by-side with shared legend."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 1.75), sharex=True)

    n = axlen.shape[1]
    cmap = plt.get_cmap("viridis")

    handles, labels = [], []
    for i in range(n):
        col = cmap(i / max(n - 1, 1))
        (h,) = ax1.plot(
            it_ax,
            axlen[:, i],
            linewidth=0.5,
            marker="o",
            markersize=0.35,
            color=col,
        )

        ax2.plot(
            it_std,
            stddev[:, i],
            linewidth=0.5,
            marker="o",
            markersize=0.35,
            color=col,
        )
        handles.append(h)
        labels.append(PARAM_NAMES_FORMAL[i])

    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Principal-axis length")
    ax1.margins(x=1e-5, y=0.01)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"Coordinate std dev × $\sigma^{-1}$")

    ax2.margins(x=1e-5, y=0.01)
    fig.tight_layout(rect=[0, 0, 1, 0.8])  # type: ignore
    fig.subplots_adjust(top=0.75)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=fig.transFigure,
    )

    fig.subplots_adjust(wspace=0.4, top=0.8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_parameter_evolution(
    iterations: List[int],
    params: np.ndarray,
    param_names: List[str] | None = None,
) -> None:
    """Plot evolution of best parameter values over iterations."""
    n_params = params.shape[1]
    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(n_params)]

    n_rows = n_cols = 3

    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4, 1 * n_rows))
    axes = axes.flatten()
    cmap = plt.get_cmap("viridis")
    x_min, x_max = min(iterations), max(iterations)

    for i in range(n_params):
        axes[i].plot(
            iterations,
            params[:, i],
            marker="o",
            linestyle="-",
            linewidth=0.55,
            markersize=1.2,
            markerfacecolor="white",
            markeredgewidth=0.4,
            color=cmap(i / (n_params - 1)),
        )
        axes[i].set_title(param_names[i], pad=2)
        axes[i].margins(x=0.01, y=0.01)
        axes[i].set_xlim(x_min, x_max)

        # Hide x-ticks for non-bottom rows
        row = i // n_cols
        if row < n_rows - 1:
            axes[i].set_xticks([])

    for ax in axes[n_params:]:
        ax.set_visible(False)

    # Set xlabel only for bottom row
    bottom_row_start = (n_rows - 1) * n_cols
    for i in range(max(0, bottom_row_start), min(n_params, bottom_row_start + n_cols)):
        axes[i].set_xlabel("Iteration")

    plt.tight_layout(h_pad=1.5)
    plt.savefig("parameter_evolution.svg")
    plt.close()
