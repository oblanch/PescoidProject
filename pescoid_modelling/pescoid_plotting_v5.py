import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def pescoid_plot():
    # Directory where the simulation data is stored
    input_dir = "simulation_results"

    # Load data
    density_data = np.load(os.path.join(input_dir, "density_data.npy"))
    mesoderm_data = np.load(os.path.join(input_dir, "mesoderm_data.npy"))
    max_mesoderm_data = np.load(os.path.join(input_dir, "max_mesoderm_data.npy"))
    smax_mesoderm_data = [x / np.max(max_mesoderm_data) for x in max_mesoderm_data]
    meso_frac_data = np.load(os.path.join(input_dir, "meso_frac_data.npy"))
    velocity_data = np.load(os.path.join(input_dir, "velocity_data.npy"))
    stress_data = np.load(os.path.join(input_dir, "stress_data.npy"))
    time_data = np.load(os.path.join(input_dir, "time_data.npy"))
    boundary_positions = np.load(os.path.join(input_dir, "boundary_positions.npy"))
    boundary_time_data = np.load(os.path.join(input_dir, "boundary_time_data.npy"))
    boundary_velocity_data = np.load(
        os.path.join(input_dir, "boundary_velocity_data.npy")
    )  # Load boundary velocity data
    mesh_params = np.load(os.path.join(input_dir, "mesh_params.npy"))
    x_coords = np.load(os.path.join(input_dir, "x_coords.npy"))
    param_list = np.load(os.path.join(input_dir, "params.npy"))

    # Reverse the density and mesoderm data if it's in reverse order
    density_data = np.flip(density_data, axis=1)
    mesoderm_data = np.flip(mesoderm_data, axis=1)
    stress_data = np.flip(stress_data, axis=1)
    velocity_data = np.flip(velocity_data, axis=1)
    # Sanity check for consistent shapes and lengths
    assert (
        len(density_data) == len(velocity_data) == len(time_data)
    ), "Mismatched lengths of density, velocity, or time data arrays"

    # Ensure length consistency
    assert (
        len(boundary_positions)
        == len(boundary_time_data)
        == len(boundary_velocity_data)
    ), f"Mismatched lengths for boundary_positions ({len(boundary_positions)}), boundary_time_data ({len(boundary_time_data)}), or boundary_velocity_data ({len(boundary_velocity_data)})"

    # Compute min and max values for velocity data
    velocity_min = np.min(velocity_data)
    velocity_max = np.max(velocity_data)

    # Create figures and axes
    fig_density, ax_density = plt.subplots()
    fig_velocity, ax_velocity = plt.subplots()

    # Initialize plot lines
    (density_line,) = ax_density.plot(
        x_coords, density_data[0], color="blue", label="Density"
    )
    (mesoderm_line,) = ax_density.plot(
        x_coords, mesoderm_data[0], color="purple", label="Mesoderm"
    )
    (velocity_line,) = ax_velocity.plot(
        x_coords, velocity_data[0], color="red", label="Velocity"
    )
    (stress_line,) = ax_density.plot(
        x_coords, stress_data[0], color="green", label="stress"
    )

    # Set x and y labels for the density and mesoderm plot
    ax_density.set_xlabel("Position (x/100 um)")
    ax_density.set_ylabel("Density and Mesoderm (Nondimensionalized)")

    # Add legend to the density plot
    ax_density.legend()

    # Set y-axis limits for density plot
    ax_density.set_ylim(-3.0, 10.0)
    ax_density.set_xlim(-3, 3)

    # Set y-axis limits for velocity plot based on computed min and max values
    ax_velocity.set_ylim(-5, 5)
    ax_velocity.set_xlim(-3, 3)

    # Add text annotation for time
    density_time_text = ax_density.text(0.02, 0.95, "", transform=ax_density.transAxes)
    velocity_time_text = ax_velocity.text(
        0.02, 0.95, "", transform=ax_velocity.transAxes
    )

    def update_density_plot(frame):
        # Update the density and mesoderm lines for each frame
        density_line.set_ydata(density_data[frame])
        mesoderm_line.set_ydata(mesoderm_data[frame])
        stress_line.set_ydata(stress_data[frame])
        density_time_text.set_text(f"Time: {time_data[frame]:.2f}s")
        return density_line, density_time_text, mesoderm_line, stress_line

    def update_velocity_plot(frame):
        velocity_line.set_ydata(velocity_data[frame])
        # mesoderm_line.set_ydata(mesoderm_data[frame])
        # stress_line.set_ydata(stress_data[frame])
        # mesoderm_line.set_ydata(mesoderm_data[frame])
        velocity_time_text.set_text(f"Time: {time_data[frame]*30:.2f}min")
        return velocity_time_text, velocity_line

    # Create animations
    ani_density = animation.FuncAnimation(
        fig_density,
        update_density_plot,
        frames=len(density_data),
        interval=100,
        blit=True,
    )

    ani_velocity = animation.FuncAnimation(
        fig_velocity,
        update_velocity_plot,
        frames=len(velocity_data),
        interval=100,
        blit=True,
    )

    # Save animations
    ani_density.save(
        "/mnt/c/Users/oblanch/FEniCS_tutorial/Dimensional Plots/density_evolution.mp4",
        writer="ffmpeg",
    )
    ani_velocity.save(
        "/mnt/c/Users/oblanch/FEniCS_tutorial/Dimensional Plots/velocity_evolution.mp4",
        writer="ffmpeg",
    )

    # Plot boundary positions over time
    plt.figure()
    plt.plot(boundary_time_data, boundary_positions)
    plt.plot(boundary_time_data, meso_frac_data)
    plt.xlabel("Time")
    plt.ylabel(r"Radius (Nondimensionalized)")
    plt.ylim(0, 2.5)
    plt.title("Boundary Tracking")
    plt.grid()
    plt.savefig(
        "/mnt/c/Users/oblanch/FEniCS_tutorial/Dimensional Plots/leading_edge_radius.png"
    )
    plt.show()
    plt.close()
    # length_scale, Delta, Da, tau_m, Gamma, Activity
    # Plot boundary velocity over time
    plt.figure()
    plt.plot(boundary_time_data, boundary_velocity_data)
    plt.xlabel("Time")
    plt.ylabel(r"Velocity at $x_0$ for which $\rho=c$")
    plt.title(
        f"Boundary Velocity over Time (L_0 = {param_list[0]}, delta = {param_list[1]}, F = {param_list[2]}, T_m = {param_list[3]}, Gamma = {param_list[4]}, A = {param_list[5]})"
    )
    plt.grid()
    plt.savefig(
        "/mnt/c/Users/oblanch/FEniCS_tutorial/Dimensional Plots/boundary_velocity.png"
    )
    # plt.show()
    plt.close()
