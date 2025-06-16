#!/usr/bin/env python3


"""Comprehensive pescoid simulation output visualization."""

from typing import Any, Dict, List, Optional

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from pescoid_modelling.visualization import _load_simulation_data
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


def plot_tissue_metrics(
    sim_data: Dict[str, Any], minutes_per_generation: float = 30.0
) -> Figure:
    """Plot tissue metrics: tissue_size, boundary_positions,
    boundary_velocity."""
    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.25))

    # Convert time to minutes
    time_minutes = sim_data["time"] * minutes_per_generation

    # Tissue size
    axes[0].plot(
        time_minutes,
        sim_data["tissue_size"],
        color="tab:blue",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Tissue size (R/R₀)")
    axes[0].set_title("Tissue size")
    axes[0].margins(x=0, y=0.025)

    # Boundary positions
    boundary_time_minutes = sim_data["boundary_times"] * minutes_per_generation
    axes[1].plot(
        boundary_time_minutes,
        sim_data["boundary_positions"],
        color="tab:orange",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Boundary position")
    axes[1].set_title("Boundary position")
    axes[1].margins(x=0, y=0.025)

    # Boundary velocity
    axes[2].plot(
        boundary_time_minutes,
        sim_data["boundary_velocity"],
        color="tab:green",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Boundary velocity")
    axes[2].set_title("Boundary velocity")
    axes[2].margins(x=0, y=0.025)

    plt.tight_layout()
    return fig


def plot_mesoderm_metrics(
    sim_data: Dict[str, Any], minutes_per_generation: float = 30.0
) -> Figure:
    """Plot mesoderm metrics: mean, center, average, fraction, max."""
    fig, axes = plt.subplots(2, 3, figsize=(5, 2.5))

    # Convert time to minutes
    time_minutes = sim_data["time"] * minutes_per_generation

    # Mesoderm mean
    axes[0, 0].plot(
        time_minutes,
        sim_data["mesoderm_mean"],
        color="tab:red",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 0].set_xlabel("Time (minutes)")
    axes[0, 0].set_ylabel("Mesoderm mean")
    axes[0, 0].set_title("Mesoderm mean")
    axes[0, 0].margins(x=0, y=0.025)

    # Mesoderm center
    axes[0, 1].plot(
        time_minutes,
        sim_data["mesoderm_center"],
        color="tab:purple",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("Mesoderm center")
    axes[0, 1].set_title("Mesoderm center")
    axes[0, 1].margins(x=0, y=0.025)

    # Mesoderm average
    axes[0, 2].plot(
        time_minutes,
        sim_data["mesoderm_average"],
        color="tab:brown",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 2].set_xlabel("Time (minutes)")
    axes[0, 2].set_ylabel("Mesoderm average")
    axes[0, 2].set_title("Mesoderm average")
    axes[0, 2].margins(x=0, y=0.025)

    # Mesoderm fraction
    axes[1, 0].plot(
        time_minutes,
        sim_data["mesoderm_fraction"],
        color="tab:pink",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 0].set_ylabel("Mesoderm fraction")
    axes[1, 0].set_title("Mesoderm fraction")
    axes[1, 0].margins(x=0, y=0.025)

    # Max mesoderm
    axes[1, 1].plot(
        time_minutes,
        sim_data["max_mesoderm"],
        color="tab:olive",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[1, 1].set_xlabel("Time (minutes)")
    axes[1, 1].set_ylabel("Max mesoderm")
    axes[1, 1].set_title("Max mesoderm")
    axes[1, 1].margins(x=0, y=0.025)

    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def plot_morphogen_metrics(
    sim_data: Dict[str, Any], minutes_per_generation: float = 30.0
) -> Figure:
    """Plot morphogen metrics: mean, center, max, edge, gradient_max,
    gradient_center.
    """
    fig, axes = plt.subplots(2, 3, figsize=(5, 2.5))
    time_minutes = sim_data["time"] * minutes_per_generation

    # Morphogen mean
    axes[0, 0].plot(
        time_minutes,
        sim_data["morphogen_mean"],
        color="tab:cyan",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 0].set_xlabel("Time (minutes)")
    axes[0, 0].set_ylabel("Morphogen mean")
    axes[0, 0].set_title("Morphogen mean")
    axes[0, 0].margins(x=0, y=0.025)

    # Morphogen center
    axes[0, 1].plot(
        time_minutes,
        sim_data["morphogen_center"],
        color="tab:gray",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("Morphogen center")
    axes[0, 1].set_title("Morphogen center")
    axes[0, 1].margins(x=0, y=0.025)

    # Max morphogen
    axes[0, 2].plot(
        time_minutes,
        sim_data["max_morphogen"],
        color="teal",
        linewidth=0.5,
        marker="o",
        markersize=0.5,
        markevery=5,
    )
    axes[0, 2].set_xlabel("Time (minutes)")
    axes[0, 2].set_ylabel("Max morphogen")
    axes[0, 2].set_title("Max morphogen")
    axes[0, 2].margins(x=0, y=0.025)

    # Morphogen edge
    if "morphogen_edge" in sim_data:
        axes[1, 0].plot(
            time_minutes,
            sim_data["morphogen_edge"],
            color="navy",
            linewidth=0.5,
            marker="o",
            markersize=0.5,
            markevery=5,
        )
        axes[1, 0].set_xlabel("Time (minutes)")
        axes[1, 0].set_ylabel("Morphogen edge")
        axes[1, 0].set_title("Morphogen edge")
        axes[1, 0].margins(x=0, y=0.025)
    else:
        axes[1, 0].axis("off")

    # Morphogen gradient max
    if "morphogen_gradient_max" in sim_data:
        axes[1, 1].plot(
            time_minutes,
            sim_data["morphogen_gradient_max"],
            color="darkred",
            linewidth=0.5,
        )
        axes[1, 1].set_xlabel("Time (minutes)")
        axes[1, 1].set_ylabel("Morphogen gradient max")
        axes[1, 1].set_title("Morphogen gradient max")
        axes[1, 1].margins(x=0, y=0.025)
    else:
        axes[1, 1].axis("off")

    # Morphogen gradient center
    if "morphogen_gradient_center" in sim_data:
        axes[1, 2].plot(
            time_minutes,
            sim_data["morphogen_gradient_center"],
            color="darkgreen",
            linewidth=0.5,
        )
        axes[1, 2].set_xlabel("Time (minutes)")
        axes[1, 2].set_ylabel("Morphogen gradient center")
        axes[1, 2].set_title("Morphogen gradient center")
        axes[1, 2].margins(x=0, y=0.025)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def plot_spatial_fields(
    sim_data: Dict[str, Any],
    time_points: Optional[List[float]] = None,
    minutes_per_generation: float = 30.0,
) -> Figure:
    """Plot spatial fields: density, mesoderm, velocity, stress, morphogen."""
    if time_points is None:
        if len(sim_data["time"]) > 20:
            indices = np.linspace(0, len(sim_data["time"]) - 1, 250, dtype=int)
        else:
            indices = np.arange(len(sim_data["time"]))
    else:
        times_minutes = sim_data["time"] * minutes_per_generation
        indices = [np.argmin(np.abs(times_minutes - tp)) for tp in time_points]  # type: ignore

    fig, axes = plt.subplots(2, 3, figsize=(11, 4.5))
    axes = axes.flatten()

    x_coords = sim_data["x_coords"]
    fields = ["density", "mesoderm", "velocity", "stress", "morphogen"]
    field_labels = ["Density ρ", "Mesoderm m", "Velocity u", "Stress σ", "Morphogen c"]

    colormaps = ["Blues", "Reds", "Greens", "Oranges", "Purples"]
    time_values = sim_data["time"][indices] * minutes_per_generation
    norm = Normalize(vmin=time_values.min(), vmax=time_values.max())

    for i, (field, label, cmap_name) in enumerate(zip(fields, field_labels, colormaps)):
        ax = axes[i]
        cmap = plt.get_cmap(cmap_name)

        for j, idx in enumerate(indices):
            time_min = sim_data["time"][idx] * minutes_per_generation
            color = cmap(norm(time_min))

            ax.plot(
                x_coords, sim_data[field][idx], color=color, linewidth=0.25, alpha=0.8
            )

        ax.set_ylabel(label)
        ax.set_xlabel("Position x")
        ax.set_title(f"{label} evolution")
        ax.margins(x=0, y=0.01)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.2, aspect=5)
        cbar.set_label("Time (min)", rotation=270, labelpad=8)

    axes[5].axis("off")

    plt.tight_layout()
    return fig


def plot_update_norms(
    sim_data: Dict[str, Any], minutes_per_generation: float = 30.0
) -> Figure:
    """Plot update norms: rho_norm, m_norm, u_norm, c_norm."""
    fig, ax = plt.subplots(1, 1, figsize=(3.85, 1.75))
    dt = sim_data.get("dt", [0.01])[0] if "dt" in sim_data else 0.01
    t_norm = np.arange(len(sim_data["rho_norm"])) * dt * minutes_per_generation

    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    labels = ["Density ‖Δρ‖₂", "Mesoderm ‖Δm‖₂", "Velocity ‖Δu‖₂", "Morphogen ‖Δc‖₂"]
    fields = ["rho_norm", "m_norm", "u_norm", "c_norm"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "d"]

    for field, label, color, ls, marker in zip(
        fields, labels, colors, linestyles, markers
    ):
        if len(sim_data[field]) > 0:
            mk = max(1, len(t_norm) // 20)
            ax.semilogy(
                t_norm,
                sim_data[field],
                label=label,
                color=color,
                linestyle=ls,
                linewidth=0.5,
                marker=marker,
                markevery=mk,
                markersize=1,
            )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Update L₂ Norm")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.5, 1))
    ax.margins(x=0, y=0)

    plt.tight_layout()
    return fig


def plot_all_diagnostics(
    sim_data: Dict[str, Any],
    save_prefix: str = "simulation_diagnostics",
    minutes_per_generation: float = 30.0,
) -> None:
    """Generate and save all diagnostic plots."""
    _set_matplotlib_publication_parameters()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Nimbus Sans"]

    fig1 = plot_tissue_metrics(sim_data, minutes_per_generation)
    fig1.savefig(f"{save_prefix}_tissue_metrics.svg", bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_mesoderm_metrics(sim_data, minutes_per_generation)
    fig2.savefig(f"{save_prefix}_mesoderm_metrics.svg", bbox_inches="tight")
    plt.close(fig2)

    fig3 = plot_morphogen_metrics(sim_data, minutes_per_generation)
    fig3.savefig(f"{save_prefix}_morphogen_metrics.svg", bbox_inches="tight")
    plt.close(fig3)

    fig4 = plot_spatial_fields(sim_data, minutes_per_generation=minutes_per_generation)
    fig4.savefig(f"{save_prefix}_spatial_fields.svg", bbox_inches="tight")
    plt.close(fig4)

    fig5 = plot_update_norms(sim_data, minutes_per_generation)
    fig5.savefig(f"{save_prefix}_update_norms.svg", bbox_inches="tight")
    plt.close(fig5)


def main():
    """Main function to run the diagnostics."""
    sim_data = _load_simulation_data("simulation_results.npz")
    plot_all_diagnostics(sim_data, "simulation")


if __name__ == "__main__":
    main()
