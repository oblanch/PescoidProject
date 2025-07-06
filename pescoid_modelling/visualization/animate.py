"""Animate pescoid wetting and dewetting from simulation data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
import numpy as np

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


@dataclass(frozen=True)
class AnimationConfig:
    """Parameter bundle for the animation."""

    axis_limit: float = 1.85
    n_rings: int = 150
    fps: int = 60
    dpi: int = 450
    cmap_bins: int = 256
    phase_box_lw: float = 0.25
    field_min: float = 0.0
    field_max: float = 1.0
    white_threshold: float = 1e-4


def build_white_reds(
    bins: int, vmin: float, vmax: float
) -> Tuple[ListedColormap, Normalize]:
    """Return a "Reds"‐based colourmap whose lowest entry is pure white."""
    base = plt.get_cmap("Reds", bins)
    colors = base(np.linspace(0, 1, bins))
    colors[0] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors), Normalize(vmin=vmin, vmax=vmax, clip=True)


def load_simulation_data(npz_path: Path | str) -> Dict[str, np.ndarray]:
    """Load simulation data."""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def animate_pescoid_radial_symmetry(
    data: Dict[str, np.ndarray],
    output_path: Path | str,
    cfg: AnimationConfig = AnimationConfig(),
) -> None:
    """Render the pescoid animation."""
    times: np.ndarray = data["time"] * 30.0
    tissue_sizes: np.ndarray = data["tissue_size"]
    x_coords: np.ndarray = data["x_coords"]
    mesoderm: np.ndarray = data["mesoderm"]

    fig, ax = plt.subplots(figsize=(2.9, 2.25))
    ax.set(
        xlim=(-cfg.axis_limit, cfg.axis_limit),
        ylim=(-cfg.axis_limit, cfg.axis_limit),
        aspect="equal",
        title="1D pescoid model visualized with radial symmetry",
    )

    cmap, norm = build_white_reds(cfg.cmap_bins, cfg.field_min, cfg.field_max)

    theta = np.linspace(0, 2 * np.pi, 50)
    radii = np.linspace(0, cfg.axis_limit, cfg.n_rings)
    wedges: List[Wedge] = []
    for i in range(cfg.n_rings - 1):
        for j in range(len(theta) - 1):
            wedge = Wedge(
                (0, 0),
                radii[i + 1],
                np.degrees(theta[j]),
                np.degrees(theta[j + 1]),
                width=radii[i + 1] - radii[i],
                facecolor="white",
                edgecolor="none",
            )
            ax.add_patch(wedge)
            wedges.append(wedge)

    circle = Circle((0, 0), 1.0, fill=False, edgecolor="lightgray", linewidth=0.6)
    ax.add_patch(circle)

    phase_text = ax.text(
        0.95,
        0.95,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round", facecolor="wheat", alpha=0.8, linewidth=cfg.phase_box_lw
        ),
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        fraction=0.05,
        pad=0.04,
        shrink=0.25,
        aspect=5,
    )
    cbar.set_label("Mesoderm (m)")

    center_idx = int(np.argmin(np.abs(x_coords)))
    max_radius_idx = int(np.argmax(tissue_sizes))

    def init() -> List[Wedge]:
        """Initialize the animation."""
        circle.set_radius(1.0)
        phase_text.set_text("")
        return [circle, phase_text, *wedges]  # type: ignore

    def animate(frame: int) -> List[Wedge]:
        """Update the animation for each frame."""
        radius = float(tissue_sizes[frame])
        circle.set_radius(radius)

        is_wetting = frame < max_radius_idx
        phase_text.set_text("Wetting" if is_wetting else "Dewetting")
        phase_text.set_bbox(
            dict(
                facecolor="green" if is_wetting else "blue",
                alpha=0.075,
                linewidth=cfg.phase_box_lw,
            )
        )

        field = mesoderm[frame]
        edge_idx = center_idx + int(
            radius * len(x_coords) / (2 * np.max(np.abs(x_coords)))
        )
        edge_idx = min(edge_idx, len(x_coords) - 1)

        x_tissue = x_coords[center_idx : edge_idx + 1]
        field_tissue = field[center_idx : edge_idx + 1]

        wedge_idx = 0
        for r in radii[:-1]:
            if r < radius:
                value = (
                    np.interp(r, x_tissue / x_tissue[-1] * radius, field_tissue)
                    if len(x_tissue) > 1
                    else float(field_tissue[0])
                )
                if abs(value) <= cfg.white_threshold:
                    colour = (1.0, 1.0, 1.0, 1.0)
                else:
                    colour = cmap(norm(value))
            else:
                colour = (1.0, 1.0, 1.0, 1.0)

            for _ in range(len(theta) - 1):
                wedges[wedge_idx].set_facecolor(colour)
                wedge_idx += 1

        return [circle, phase_text, *wedges]  # type: ignore

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(times),
        interval=1000 / cfg.fps,
        blit=True,
        repeat=True,
    )
    fig.tight_layout()
    anim.save(
        str(output_path) + ".mp4",
        fps=cfg.fps,
        dpi=cfg.dpi,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    gif_writer = PillowWriter(fps=cfg.fps // 2)
    anim.save(
        str(output_path) + ".gif",
        dpi=cfg.dpi,
        writer=gif_writer,
    )
    plt.close(fig)


def animate_pescoid_1d_profiles(
    data: Dict[str, np.ndarray],
    output_path: Path | str,
    cfg: AnimationConfig = AnimationConfig(),
) -> None:
    """Create animated 1D profile plots."""
    times: np.ndarray = data["time"] * 30.0
    tissue_sizes: np.ndarray = data["tissue_size"]
    x_coords: np.ndarray = data["x_coords"]
    mesoderm: np.ndarray = data["mesoderm"]
    density: np.ndarray = data["density"]
    velocity: np.ndarray = data["velocity"]
    stress: np.ndarray = data["stress"]

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.55), sharex=True, sharey=False)
    ax_density, ax_velocity, ax_mesoderm, ax_stress = axes.flatten()

    # Density
    ax_density.set(
        xlim=(-2, 2),
        ylim=(0, 1.25),
        ylabel="Density (ρ)",
    )
    (line_density,) = ax_density.plot([], [], linewidth=0.55, color="tab:blue")
    fill_density = None

    # Velocity
    ax_velocity.set(
        xlim=(-2, 2),
        ylim=(-0.7, 0.7),
        ylabel="Velocity (u)",
    )
    (line_velocity,) = ax_velocity.plot([], [], linewidth=0.55, color="tab:green")

    # Mesoderm
    ax_mesoderm.set(
        xlim=(-2, 2),
        ylim=(-1, 1.75),
        ylabel="Mesoderm m",
        xlabel="X (coordinate position)",
    )
    (line_mesoderm,) = ax_mesoderm.plot([], [], linewidth=0.6, color="tab:red")
    fill_mesoderm = None

    # Stress
    ax_stress.set(
        xlim=(-2, 2),
        ylim=(0, 2.55),
        ylabel="Stress (σ)",
        xlabel="X (coordinate position)",
    )
    (line_stress,) = ax_stress.plot([], [], linewidth=0.55, color="tab:purple")
    fill_stress = None

    center_idx = int(np.argmin(np.abs(x_coords)))

    def init() -> List:
        """Initialize the animation."""
        line_density.set_data([], [])
        line_velocity.set_data([], [])
        line_mesoderm.set_data([], [])
        line_stress.set_data([], [])
        return [line_density, line_velocity, line_mesoderm, line_stress]

    def animate(frame: int) -> List:
        """Update the animation for each frame."""
        radius = float(tissue_sizes[frame])

        density_field = density[frame]
        velocity_field = velocity[frame]
        mesoderm_field = mesoderm[frame]
        stress_field = stress[frame]

        edge_idx = center_idx + int(
            radius * len(x_coords) / (2 * np.max(np.abs(x_coords)))
        )
        edge_idx = min(edge_idx, len(x_coords) - 1)

        left_idx = center_idx - (edge_idx - center_idx)
        left_idx = max(0, left_idx)

        x_tissue = x_coords[left_idx : edge_idx + 1]
        density_tissue = density_field[left_idx : edge_idx + 1]
        velocity_tissue = velocity_field[left_idx : edge_idx + 1]
        mesoderm_tissue = mesoderm_field[left_idx : edge_idx + 1]
        stress_tissue = stress_field[left_idx : edge_idx + 1]

        line_density.set_data(x_tissue, density_tissue)
        line_velocity.set_data(x_tissue, velocity_tissue)
        line_mesoderm.set_data(x_tissue, mesoderm_tissue)
        line_stress.set_data(x_tissue, stress_tissue)

        nonlocal fill_density, fill_mesoderm, fill_stress

        if fill_density is not None:
            fill_density.remove()
        if fill_mesoderm is not None:
            fill_mesoderm.remove()
        if fill_stress is not None:
            fill_stress.remove()

        fill_density = ax_density.fill_between(
            x_tissue, 0, density_tissue, alpha=0.3, color="blue"
        )
        fill_mesoderm = ax_mesoderm.fill_between(
            x_tissue, 0, mesoderm_tissue, alpha=0.3, color="red"
        )
        fill_stress = ax_stress.fill_between(
            x_tissue, 0, stress_tissue, alpha=0.3, color="purple"
        )

        return [
            line_density,
            line_velocity,
            line_mesoderm,
            line_stress,
            fill_density,
            fill_mesoderm,
        ]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(times),
        interval=1000 / cfg.fps,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()
    anim.save(
        str(output_path) + ".mp4",
        dpi=cfg.dpi,
        writer=animation.FFMpegWriter(
            fps=cfg.fps,
            codec="libx264",
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
            ],
        ),
    )

    gif_writer = PillowWriter(fps=cfg.fps // 2)
    anim.save(
        str(output_path) + ".gif",
        dpi=cfg.dpi,
        writer=gif_writer,
    )

    plt.close(fig)


def animate_pescoid_1d_overlay(
    data: Dict[str, np.ndarray],
    output_path: Path | str,
    cfg: AnimationConfig = AnimationConfig(),
) -> None:
    """Animate density, mesoderm and stress overlaid on one plot."""
    times = data["time"] * 30.0
    tissue_sizes = data["tissue_size"]
    x_coords = data["x_coords"]
    density = data["density"]
    mesoderm = data["mesoderm"]
    stress = data["stress"]

    fig, ax = plt.subplots(figsize=(2.25, 2), sharex=False)

    ax.set(
        xlim=(-2, 2),
        ylim=(-1.00, 2.55),
        xlabel="X (coordinate position)",
        ylabel="Nondimensionalised units",
    )

    (line_density,) = ax.plot([], [], color="tab:blue", lw=0.7, label="Density (ρ)")
    (line_mesoderm,) = ax.plot([], [], color="tab:red", lw=0.7, label="Mesoderm (m)")
    (line_stress,) = ax.plot([], [], color="tab:purple", lw=0.7, label="Stress (σ)")

    ax.legend(frameon=False, bbox_to_anchor=(0.725, 1.3))
    centre_idx = int(np.argmin(np.abs(x_coords)))

    def init() -> List:
        """Blank lines to start."""
        for ln in (line_density, line_mesoderm, line_stress):
            ln.set_data([], [])
        return [line_density, line_mesoderm, line_stress]

    def animate(frame: int) -> List:
        """Update traces for frame."""
        radius = float(tissue_sizes[frame])

        edge_idx = centre_idx + int(
            radius * len(x_coords) / (2 * np.abs(x_coords).max())
        )
        edge_idx = min(edge_idx, len(x_coords) - 1)
        left_idx = max(0, centre_idx - (edge_idx - centre_idx))

        x_t = x_coords[left_idx : edge_idx + 1]
        ρ_t = density[frame, left_idx : edge_idx + 1]
        m_t = mesoderm[frame, left_idx : edge_idx + 1]
        σ_t = stress[frame, left_idx : edge_idx + 1]

        line_density.set_data(x_t, ρ_t)
        line_mesoderm.set_data(x_t, m_t)
        line_stress.set_data(x_t, σ_t)

        return [line_density, line_mesoderm, line_stress]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(times),
        interval=1000 / cfg.fps,
        blit=False,
        repeat=True,
    )

    fig.tight_layout(pad=0.1)
    anim.save(
        str(output_path) + ".mp4",
        dpi=cfg.dpi,
        writer=animation.FFMpegWriter(
            fps=cfg.fps,
            codec="libx264",
            extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
        ),
    )

    gif_writer = PillowWriter(fps=cfg.fps // 2)
    anim.save(
        str(output_path) + ".gif",
        dpi=cfg.dpi,
        writer=gif_writer,
    )

    plt.close(fig)


def main() -> None:
    """CLI entry‑point that reproduces the legacy default behaviour."""
    _set_matplotlib_publication_parameters()
    data = load_simulation_data("simulation_results.npz")

    animate_pescoid_radial_symmetry(data, "pescoid_mesoderm_animation")
    animate_pescoid_1d_profiles(data, "pescoid_1d_profiles_animation")
    animate_pescoid_1d_overlay(data, "pescoid_1d_overlay_animation")


if __name__ == "__main__":
    main()
