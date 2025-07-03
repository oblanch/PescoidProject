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
    fps: int = 24
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
    gif_writer = PillowWriter(fps=cfg.fps)
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

    fig, (ax_density, ax_velocity, ax_mesoderm) = plt.subplots(
        3, 1, figsize=(2, 3.75), sharex=True
    )

    # Density
    ax_density.set(
        xlim=(x_coords.min(), x_coords.max()),
        ylim=(0, 3.5),
        ylabel="Density ρ",
        title="1D pescoid profiles",
    )
    ax_density.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.6)
    (line_density,) = ax_density.plot([], [], "b-", linewidth=0.6)
    fill_density = None

    # Velocity
    ax_velocity.set(
        xlim=(x_coords.min(), x_coords.max()),
        ylim=(-2, 2),
        ylabel="Velocity u",
    )
    ax_velocity.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.6)
    ax_velocity.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.6)
    (line_velocity,) = ax_velocity.plot([], [], "g-", linewidth=0.6)

    # Mesoderm
    ax_mesoderm.set(
        xlim=(x_coords.min(), x_coords.max()),
        ylim=(-1.2, 1.2),
        ylabel="Mesoderm m",
        xlabel="Position x",
    )
    ax_mesoderm.axvline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.6)
    ax_mesoderm.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.6)
    (line_mesoderm,) = ax_mesoderm.plot([], [], "r-", linewidth=0.6)
    fill_mesoderm = None

    boundary_lines = []
    for ax in [ax_density, ax_velocity, ax_mesoderm]:
        left_line = ax.axvline(
            -1.0, color="lightgray", linestyle=":", alpha=0.6, linewidth=0.6
        )
        right_line = ax.axvline(
            1.0, color="lightgray", linestyle=":", alpha=0.6, linewidth=0.6
        )
        boundary_lines.extend([left_line, right_line])

    phase_text = ax_density.text(
        0.95,
        0.95,
        "",
        transform=ax_density.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round", facecolor="wheat", alpha=0.8, linewidth=cfg.phase_box_lw
        ),
    )

    center_idx = int(np.argmin(np.abs(x_coords)))
    max_radius_idx = int(np.argmax(tissue_sizes))

    def init() -> List:
        """Initialize the animation."""
        line_density.set_data([], [])
        line_velocity.set_data([], [])
        line_mesoderm.set_data([], [])
        phase_text.set_text("")
        return [line_density, line_velocity, line_mesoderm, phase_text]

    def animate(frame: int) -> List:
        """Update the animation for each frame."""
        radius = float(tissue_sizes[frame])
        current_time = times[frame]

        for i in range(0, len(boundary_lines), 2):
            boundary_lines[i].set_xdata([-radius])
            boundary_lines[i + 1].set_xdata([radius])

        is_wetting = frame < max_radius_idx
        phase_text.set_text("Wetting" if is_wetting else "Dewetting")
        phase_text.set_bbox(
            dict(
                facecolor="green" if is_wetting else "blue",
                alpha=0.075,
                linewidth=cfg.phase_box_lw,
            )
        )

        density_field = density[frame]
        velocity_field = velocity[frame]
        mesoderm_field = mesoderm[frame]

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

        line_density.set_data(x_tissue, density_tissue)
        line_velocity.set_data(x_tissue, velocity_tissue)
        line_mesoderm.set_data(x_tissue, mesoderm_tissue)

        nonlocal fill_density, fill_mesoderm

        if fill_density is not None:
            fill_density.remove()
        if fill_mesoderm is not None:
            fill_mesoderm.remove()

        fill_density = ax_density.fill_between(
            x_tissue, 0, density_tissue, alpha=0.3, color="blue"
        )
        fill_mesoderm = ax_mesoderm.fill_between(
            x_tissue, 0, mesoderm_tissue, alpha=0.3, color="red"
        )

        return [
            line_density,
            line_velocity,
            line_mesoderm,
            phase_text,
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
        str(output_path),
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

    plt.close(fig)


def main() -> None:
    """CLI entry‑point that reproduces the legacy default behaviour."""
    _set_matplotlib_publication_parameters()

    data = load_simulation_data("simulation_results.npz")
    animate_pescoid_radial_symmetry(data, "pescoid_mesoderm_animation")

    profile_cfg = AnimationConfig(dpi=450)
    animate_pescoid_1d_profiles(data, "pescoid_1d_profiles_animation.mp4", profile_cfg)


if __name__ == "__main__":
    main()
