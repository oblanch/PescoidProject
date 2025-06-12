"""Code to plot trajectories of a single simulation."""

import os
from typing import Dict, Optional, Tuple

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from pescoid_modelling.objective import _calculate_trajectory_mismatch
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


def visualize_simulation_results(
    data_path: str,
    experimental_npz: Optional[str] = None,
    output_dir: str = ".",
    save_prefix: str = "simulation",
) -> None:
    """Main function to visualize simulation results.

    Args:
      data_path: Path to the simulation NPZ file.
      experimental_npz: Optional path to the experimental NPZ file.
      output_dir: Directory to save the output plots.
      save_prefix: Prefix for the saved plot files.

    Examples::
    >>> visualize_simulation_results(
            data_path="simulation_results.npz",
            experimental_npz="reference_timeseries.npz",
        )
    """
    _set_matplotlib_publication_parameters()
    os.makedirs(output_dir, exist_ok=True)

    sim_data = load_trajectory_data(data_path, simulation=True)
    experimental_data = None
    if experimental_npz is not None:
        experimental_data = load_trajectory_data(experimental_npz, simulation=False)
        sim_data, experimental_data = truncate_to_common_timespan(
            sim_data, experimental_data
        )

    plot_simulation_timeseries(sim_data, experimental_data, output_dir, save_prefix)


def load_trajectory_data(
    file_path: str, simulation: bool = False
) -> Dict[str, np.ndarray]:
    """Load simulation NPZ."""
    data = np.load(file_path)
    trajectory_data = {}
    trajectory_data["mesoderm_fraction"] = data["mesoderm_fraction"]
    trajectory_data["tissue_size"] = data["tissue_size"]
    trajectory_data["time"] = data["time"]

    if simulation:
        trajectory_data["time"] = (
            trajectory_data["time"] * 30.0
        )  # Convert time to minutes
        trajectory_data["mesoderm_fraction"] = data["mesoderm_fraction"]

    return trajectory_data


def calculate_normalization_scales(
    experimental_data: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """Calculate scales for standard deviation-based normalization.

    Returns:
      Tuple of (tissue_std, mesoderm_std)
    """
    tissue_std = float(np.std(experimental_data["tissue_size"]))
    mesoderm_std = float(np.std(experimental_data["mesoderm_fraction"]))

    # Protect against zero std (constant signals)
    tissue_std = tissue_std if tissue_std > 0 else 1.0
    mesoderm_std = mesoderm_std if mesoderm_std > 0 else 1.0

    return tissue_std, mesoderm_std


def interpolate_simulation_to_experimental_timepoints(
    sim_time_minutes: np.ndarray,
    sim_values: np.ndarray,
    exp_time_minutes: np.ndarray,
) -> np.ndarray:
    """Interpolate simulation data onto experimental time grid."""
    valid_exp_mask = exp_time_minutes <= sim_time_minutes[-1]

    if not np.any(valid_exp_mask):
        raise ValueError("No experimental time points within simulation range")

    exp_time_valid = exp_time_minutes[valid_exp_mask]
    return np.interp(exp_time_valid, sim_time_minutes, sim_values)


def calculate_l2_errors(
    sim_data: Dict[str, np.ndarray], exp_data: Dict[str, np.ndarray]
) -> Tuple[float, float]:
    """Calculate normalized L2² errors for tissue size and Mesoderm fraction."""
    tissue_std, mesoderm_std = calculate_normalization_scales(exp_data)

    # Interpolate simulation data onto experimental time grid
    tissue_sim_interp = interpolate_simulation_to_experimental_timepoints(
        sim_data["time"], sim_data["tissue_size"], exp_data["time"]
    )
    meso_sim_interp = interpolate_simulation_to_experimental_timepoints(
        sim_data["time"], sim_data["mesoderm_fraction"], exp_data["time"]
    )

    valid_exp_mask = exp_data["time"] <= sim_data["time"][-1]
    exp_tissue_valid = exp_data["tissue_size"][valid_exp_mask]
    exp_meso_valid = exp_data["mesoderm_fraction"][valid_exp_mask]

    # meso_sim_scaled, scale_factor = _least_squares_rescale(
    #     meso_sim_interp, exp_meso_valid
    # )
    tissue_l2_sq = _calculate_trajectory_mismatch(
        tissue_sim_interp, exp_tissue_valid, tissue_std
    )
    meso_l2_sq = _calculate_trajectory_mismatch(
        meso_sim_interp, exp_meso_valid, mesoderm_std
    )
    # meso_l2_sq = _calculate_trajectory_mismatch(
    #     meso_sim_scaled, exp_meso_valid, mesoderm_std
    # )

    return tissue_l2_sq, meso_l2_sq


def truncate_to_common_timespan(
    sim_data: Dict[str, np.ndarray], exp_data: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Truncate dataset to share common time span."""
    sim_final_time = sim_data["time"][-1]
    exp_final_time = exp_data["time"][-1]

    min_final_time = min(sim_final_time, exp_final_time)

    sim_mask = sim_data["time"] <= min_final_time
    sim_truncated = {key: values[sim_mask] for key, values in sim_data.items()}

    exp_mask = exp_data["time"] <= min_final_time
    exp_truncated = {key: values[exp_mask] for key, values in exp_data.items()}

    return sim_truncated, exp_truncated


def plot_simulation_timeseries(
    data: Dict[str, np.ndarray],
    experimental_data: Optional[Dict[str, np.ndarray]] = None,
    output_dir: str = ".",
    save_prefix: str = "simulation",
) -> None:
    """Plot tissue size and Mesoderm fraction over time."""
    base_width = 2.25
    base_height = 1.75
    ratio = 0.625

    fig_width = base_width * ratio
    fig_height = base_height * ratio
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Axes positions
    left = 0.15
    bottom = 0.15
    width = 0.65
    height = 0.75
    ax1 = fig.add_axes([left, bottom, width, height])  # type: ignore

    # Plot simulation tissue size on primary y-axis
    line1 = ax1.plot(
        data["time"],
        data["tissue_size"],
        color="tab:blue",
        linewidth=0.75,
        linestyle="-",
        label="Tissue size (sim)",
    )

    # Plot experimental tissue size
    if experimental_data is not None:
        exp_line1 = ax1.plot(
            experimental_data["time"],
            experimental_data["tissue_size"],
            color="tab:blue",
            linewidth=0.75,
            linestyle="--",
            label="Tissue size (exp)",
        )

    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Tissue size", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.spines["left"].set_color("tab:blue")

    # Plot simulation Mesoderm fraction on secondary y-axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        data["time"],
        data["mesoderm_fraction"],
        color="tab:red",
        linewidth=0.75,
        linestyle="-",
        label="Mesoderm fraction (sim)",
    )

    # Plot experimental Mesoderm fraction
    if experimental_data is not None:
        exp_line2 = ax2.plot(
            experimental_data["time"],
            experimental_data["mesoderm_fraction"],
            color="tab:red",
            linewidth=0.75,
            linestyle="--",
            label="Mesoderm fraction (exp)",
        )

    ax2.set_ylabel("Mesoderm fraction", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.spines["right"].set_color("tab:red")

    # Add legend
    custom_lines = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=0.75, label="Simulation"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=0.75,
            label="Experimental",
        ),
    ]
    legend = ax1.legend(
        handles=custom_lines,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(-0.275, 1.3),
    )

    # Add normalized L2² error text if experimental data is available
    if experimental_data is not None:
        tissue_l2_sq, meso_l2_sq = calculate_l2_errors(data, experimental_data)
        error_text = (
            f"L²(tissue) = {tissue_l2_sq:.3f}\n" f"L²(mesoderm) = {meso_l2_sq:.3f}\n"
        )
        ax1.text(
            0.5,
            1.215,
            error_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            linespacing=1.5,
        )

    ax1.set_xmargin(0)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # plt.tight_layout()
    suffix = "_with_exp" if experimental_data is not None else ""
    output_path = os.path.join(output_dir, f"{save_prefix}_timeseries{suffix}.svg")
    plt.savefig(output_path, bbox_inches="tight")
