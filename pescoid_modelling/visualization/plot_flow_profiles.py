#!/usr/bin/env python3

"""Code to plot L2 euclidean update norm of the solution vector for each
coupled equation.
"""

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from pescoid_modelling.visualization import _load_simulation_data
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


def plot_flow_profile(
    sim_data: dict,
    time_points: list[float] | None = None,
    max_curves: int | None = None,
    colormap: str = "viridis",
    minutes_per_generation: float = 30.0,
) -> Figure:
    """Plot the velocity field (flow profile) at specified times,
    or across all snapshots if time_points is None.

    Args:
      sim_data: output of load_simulation_data()
      time_points: times to plot
        If None, plots at every snapshot.
      max_curves: maximum number of curves to plot
    """
    x = sim_data["x_coords"]
    times_generation = sim_data["time"]
    times = times_generation * minutes_per_generation
    velocity = sim_data["velocity"]

    if time_points is None:
        all_idxs = np.arange(len(times))
        if max_curves is not None and len(all_idxs) > max_curves:

            # uniform subsample
            step = len(all_idxs) / max_curves
            idxs = [int(i * step) for i in range(max_curves)]
        else:
            idxs = list(all_idxs)
    else:
        idxs = [int(np.argmin(np.abs(times - tp))) for tp in time_points]

    cmap = plt.get_cmap(colormap)
    t_min, t_max = times[idxs].min(), times[idxs].max()
    norm = Normalize(vmin=t_min, vmax=t_max)
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    # plot curves
    fig, ax = plt.subplots(figsize=(3, 1.5))
    for idx in idxs:
        color = cmap(norm(times[idx]))
        ax.plot(
            x,
            velocity[idx],
            color=color,
            linewidth=0.25,
            alpha=0.7,
        )

    ax.margins(x=0, y=0.025)
    ax.set_xlabel("x")
    ax.set_ylabel("Velocity")

    cbar_opts = dict(orientation="vertical", shrink=0.3, aspect=6.5)

    cbar = fig.colorbar(mappable, ax=ax, **cbar_opts)  # type: ignore
    cbar.set_label(r"$t$(minutes)")
    return fig


if __name__ == "__main__":
    sim = _load_simulation_data("simulation_results.npz")
    n_snap = len(sim["time"])
    _set_matplotlib_publication_parameters()
    fig = plot_flow_profile(sim)

    fig.savefig("flow_profiles.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
