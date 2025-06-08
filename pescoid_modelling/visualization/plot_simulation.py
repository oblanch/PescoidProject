"""Code to plot results from a single simulation."""

import os
from typing import Any, Dict, List, Tuple

import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


def load_simulation_data(file_path: str) -> Dict[str, np.ndarray]:
    """Load simulation NPZ."""
    data = np.load(file_path)
    sim_data = {}
    sim_data["density"] = np.flip(data["density"], axis=1)
    sim_data["mesoderm"] = np.flip(data["mesoderm"], axis=1)
    sim_data["max_mesoderm"] = data["max_mesoderm"]
    sim_data["mesoderm_fraction"] = data["mesoderm_fraction"]
    sim_data["mesoderm_average"] = data["mesoderm_average"]
    sim_data["mesoderm_center"] = data["mesoderm_center"]
    sim_data["velocity"] = np.flip(data["velocity"], axis=1)
    sim_data["stress"] = np.flip(data["stress"], axis=1)
    sim_data["time"] = data["time"]
    sim_data["boundary_positions"] = data["boundary_positions"]
    sim_data["boundary_times"] = data["boundary_times"]
    sim_data["boundary_velocity"] = data["boundary_velocity"]
    sim_data["x_coords"] = data["x_coords"]

    validate_data_consistency(sim_data)
    return sim_data


def validate_data_consistency(data: Dict[str, np.ndarray]) -> None:
    """Validate the consistency of simulation data ensuring array lengths
    match.
    """
    if not (len(data["density"]) == len(data["velocity"]) == len(data["time"])):
        raise ValueError("Mismatched lengths of density, velocity, or time data arrays")

    if not (
        len(data["boundary_positions"])
        == len(data["boundary_times"])
        == len(data["boundary_velocity"])
    ):
        raise ValueError(
            f"Mismatched lengths for boundary data: "
            f"positions ({len(data['boundary_positions'])}), "
            f"times ({len(data['boundary_times'])}), "
            f"velocity ({len(data['boundary_velocity'])})"
        )


def setup_evolution_plots(
    data: Dict[str, np.ndarray],
) -> Tuple[Tuple[Figure, Axes, List[Line2D], Text], Tuple[Figure, Axes, Line2D, Text]]:
    """Set up figures and axes for both density and velocity plots.

    Returns:
      (Density plot elements, Velocity plot elements)
    """
    fig_density, ax_density = plt.subplots(figsize=(2.65, 2.5))

    (density_line,) = ax_density.plot(
        data["x_coords"], data["density"][0], color="blue", label="Density"
    )

    (mesoderm_line,) = ax_density.plot(
        data["x_coords"], data["mesoderm"][0], color="purple", label="Mesoderm"
    )

    (stress_line,) = ax_density.plot(
        data["x_coords"], data["stress"][0], color="green", label="Stress"
    )

    # Configure density plot
    ax_density.set_xlabel("Position (x/100 μm)")
    ax_density.set_ylabel("Density and mesoderm (nondimensionalized)")
    ax_density.legend(frameon=False)
    ax_density.set_ylim(-3.0, 10.0)
    ax_density.set_xlim(-3, 3)

    # Add time annotation for density plot
    density_time_text = ax_density.text(0.02, 0.95, "", transform=ax_density.transAxes)

    # Setup velocity plot
    fig_velocity, ax_velocity = plt.subplots(figsize=(2.65, 2.5))

    (velocity_line,) = ax_velocity.plot(
        data["x_coords"], data["velocity"][0], color="red", label="Velocity"
    )

    # Configure velocity plot
    ax_velocity.set_xlabel("Position (x/100 μm)")
    ax_velocity.set_ylabel("Velocity (Nondimensionalized)")
    ax_velocity.legend(frameon=False)
    ax_velocity.set_ylim(-5, 5)
    ax_velocity.set_xlim(-3, 3)

    # Add time annotation for velocity plot
    velocity_time_text = ax_velocity.text(
        0.02, 0.95, "", transform=ax_velocity.transAxes
    )

    density_elements = (
        fig_density,
        ax_density,
        [density_line, mesoderm_line, stress_line],
        density_time_text,
    )
    velocity_elements = (fig_velocity, ax_velocity, velocity_line, velocity_time_text)

    return density_elements, velocity_elements


def update_density_plot(
    frame: int, data: Dict[str, np.ndarray], lines: List[Line2D], time_text: Text
) -> List[Any]:
    """Update the density plot for animation.

    Args:
      frame: Current animation frame
      data: Dictionary containing simulation data
      lines: List of line objects [density_line, mesoderm_line, stress_line]
      time_text: Text object for displaying time

    Returns:
      List of updated artists
    """
    density_line, mesoderm_line, stress_line = lines

    # Update the data for each line
    density_line.set_ydata(data["density"][frame])
    mesoderm_line.set_ydata(data["mesoderm"][frame])
    stress_line.set_ydata(data["stress"][frame])

    # Update time text
    time_text.set_text(f"Time: {data['time'][frame]:.2f}s")

    return [density_line, mesoderm_line, stress_line, time_text]


def update_velocity_plot(
    frame: int, data: Dict[str, np.ndarray], velocity_line: Line2D, time_text: Text
) -> List[Any]:
    """Update the velocity plot for animation.

    Args:
      frame: Current animation frame
      data: Dictionary containing simulation data
      velocity_line: Line object for velocity data
      time_text: Text object for displaying time

    Returns:
      List of updated artists
    """
    # Update the velocity line
    velocity_line.set_ydata(data["velocity"][frame])

    # Update time text (converting to minutes)
    time_text.set_text(f"Time: {data['time'][frame]*30:.2f} min")

    return [velocity_line, time_text]


def create_evolution_animations(
    data: Dict[str, np.ndarray], output_dir: str, interval: int = 100
) -> None:
    """Create and save animations for both density and velocity plots.

    Args:
      data: Dictionary containing simulation data
      output_dir: Directory to save output files
      interval: Time interval between frames in milliseconds
    """
    density_elements, velocity_elements = setup_evolution_plots(data)
    fig_density, ax_density, density_lines, density_time_text = density_elements
    fig_velocity, ax_velocity, velocity_line, velocity_time_text = velocity_elements

    # Create density animation
    ani_density = animation.FuncAnimation(
        fig_density,
        update_density_plot,
        frames=len(data["density"]),
        fargs=(data, density_lines, density_time_text),
        interval=interval,
        blit=True,
    )

    # Create velocity animation
    ani_velocity = animation.FuncAnimation(
        fig_velocity,
        update_velocity_plot,
        frames=len(data["velocity"]),
        fargs=(data, velocity_line, velocity_time_text),
        interval=interval,
        blit=True,
    )

    # Save animations
    density_path = os.path.join(output_dir, "density_evolution.mp4")
    velocity_path = os.path.join(output_dir, "velocity_evolution.mp4")

    ani_density.save(density_path, writer="ffmpeg")
    ani_velocity.save(velocity_path, writer="ffmpeg")

    plt.close(fig_density)
    plt.close(fig_velocity)

    print(f"Saved density animation to {density_path}")
    print(f"Saved velocity animation to {velocity_path}")


def create_boundary_plots(data: Dict[str, np.ndarray], output_dir: str) -> None:
    """Create and save static boundary plots."""
    # Plot boundary positions and mesoderm fraction
    fig_boundary, ax_boundary = plt.subplots(figsize=(1.75, 1.25))

    ax_boundary.plot(
        data["boundary_times"], data["boundary_positions"], label="Boundary Position"
    )

    ax_boundary.plot(
        data["boundary_times"], data["mesoderm_fraction"], label="Mesoderm Fraction"
    )

    # Configure plot
    ax_boundary.set_xlabel("Time")
    ax_boundary.set_ylabel("Radius (nondimensionalized)")
    ax_boundary.set_ylim(0, 2.5)
    ax_boundary.set_title("Boundary tracking")
    ax_boundary.legend(frameon=False)
    ax_boundary.margins(x=0, y=0)

    # Save
    boundary_path = os.path.join(output_dir, "leading_edge_radius.png")
    plt.savefig(boundary_path, dpi=450, bbox_inches="tight")
    plt.close(fig_boundary)

    # Plot boundary velocity
    fig_velocity, ax_velocity = plt.subplots(figsize=(1.75, 1.25))
    ax_velocity.plot(data["boundary_times"], data["boundary_velocity"])

    # Configure plot
    ax_velocity.set_xlabel("Time")
    ax_velocity.set_ylabel(r"Velocity at $x_0$ for which $\rho=c$")
    ax_velocity.set_title("Boundary velocity over time")
    ax_velocity.margins(x=0, y=0)

    # Sav
    velocity_path = os.path.join(output_dir, "boundary_velocity.png")
    plt.savefig(velocity_path, dpi=450, bbox_inches="tight")
    plt.close(fig_velocity)


def visualize_simulation_results(
    data_path: str,
    output_dir: str = ".",
) -> None:
    """Main function to visualize simulation results."""
    _set_matplotlib_publication_parameters()
    os.makedirs(output_dir, exist_ok=True)
    data = load_simulation_data(data_path)
    create_evolution_animations(data, output_dir)
    create_boundary_plots(data, output_dir)


def main() -> None:
    """Main function to run the visualization."""
    data_path = (
        ".../.../PescoidProject/opt_out/optimization_config/simulation_results.npz"
    )
    visualize_simulation_results(data_path)


if __name__ == "__main__":
    main()
