"""Code to plot trajectories of a single simulation."""

from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters

PARAM_NAMES_FORMAL = [
    "δ\n(diffusivity)",
    "F\n(advection)",
    "Tau_m\n(mesoderm growth timescale)",
    "Γ\n(non friction coefficient)",
    "A\n(activity)",
    "β\n(contribution of mesoderm fate to cell's contractility)",
    "r\n(sensitivity of cells to mechanical feedback)",
    "m_sensitivity\n(sensitivity of the increase in contractility\nwhen cells become mesoderm)",
]


def plot_optimization_metrics(optim_csv: str = "cma_restart_4xrecentbest.dat") -> None:
    """Entry point to plot optimization results."""
    _set_matplotlib_publication_parameters()
    iterations, fitness, sigma, params = load_cma_data(optim_csv)

    plot_fitness(iterations, fitness)
    plot_sigma(iterations, sigma)
    plot_parameter_evolution(iterations, params, param_names=PARAM_NAMES_FORMAL)


def load_cma_data(csv_file: str) -> Tuple[list, list, list, np.ndarray]:
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
    ]

    df = pd.read_csv(
        csv_file,
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


def plot_fitness(iterations: List[int], fitness: List[float]) -> None:
    """Plot best fitness over iterations."""
    plt.figure(figsize=(1.7, 1.20))

    ax = plt.gca()
    ax.margins(x=0.01, y=0.01)

    plt.plot(
        iterations, fitness, marker="o", linestyle="-", markersize=1, linewidth=0.5
    )
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.tight_layout()
    plt.savefig("fitness_over_iterations.svg")
    plt.close()


def plot_sigma(iterations: List[int], sigma: List[float]) -> None:
    """Plot sigma (step-size) over iterations."""
    plt.figure(figsize=(1.7, 1.20))

    ax = plt.gca()
    ax.margins(x=0.01, y=0.01)

    plt.plot(
        iterations,
        sigma,
        marker="o",
        linestyle="-",
        markersize=1,
        linewidth=0.5,
        color="orange",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Sigma (step-size)")
    plt.tight_layout()
    plt.savefig("sigma_over_iterations.svg")
    plt.close()


def plot_parameter_evolution(
    iterations: List[int],
    params: np.ndarray,
    param_names: List[str] | None = None,
) -> None:
    """Plot evolution of each parameter over iterations.

    Args:
      params: ndarray of shape (n_iterations, n_params)
      param_names: list of strings for labeling each subplot. If None, defaults
      to Param 1, Param 2, etc.
    """
    n_params = params.shape[1]
    if param_names is None:
        param_names = [f"Param {i+1}" for i in range(n_params)]

    n_rows = int(np.ceil(n_params / 2))
    n_cols = 2

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4, 1.3 * n_rows), sharex=True
    )
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")

    for i in range(n_params):
        axes[i].plot(
            iterations,
            params[:, i],
            marker="o",
            linestyle="-",
            markersize=1.5,
            linewidth=0.75,
            color=cmap(i / (n_params - 1)),
        )
        axes[i].set_title(param_names[i])
        axes[i].margins(x=0.01, y=0.01)

    for ax in axes[n_params:]:
        ax.set_visible(False)

    for ax in axes[-2:]:
        ax.set_xlabel("Iteration")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore
    plt.savefig("parameter_evolution.svg")
    plt.close()
