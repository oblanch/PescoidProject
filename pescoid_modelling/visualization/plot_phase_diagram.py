"""Visualize parameter NxN sweep as a phase diagram heat-map."""

import argparse
from pathlib import Path
from typing import Tuple

from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters

BASE_ONSET = 270.0


def _load_sweep(csv_path: Path, tag: str) -> pd.DataFrame:
    """Return a DF containing the specified sweep."""
    df = pd.read_csv(csv_path)
    if "pair" not in df.columns:
        raise ValueError("CSV must contain a 'pair' column")

    df = df[df["pair"] == tag].copy()
    if df.empty:
        raise ValueError(f"No rows found for sweep '{tag}' in {csv_path}")

    return df


def _pivot_grid(
    df: pd.DataFrame, x: str, y: str, z: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the DataFrame into a grid for plotting."""
    grid = df.pivot_table(index=y, columns=x, values=z, aggfunc="first")
    return (
        grid.columns.values.astype(float),
        grid.index.values.astype(float),
        grid.values.astype(float),
    )


def _choose_cmap_and_norm(
    var: str,
) -> Tuple[ListedColormap, BoundaryNorm | None]:
    """Get colormap, norm, label, and ticks for the variable."""
    if var == "state":
        cmap = plt.get_cmap("Blues_r", 4)
    else:
        cmap = plt.get_cmap("Greens_r", 4)
    boundaries = np.arange(-0.5, 4, 1)
    return cmap, BoundaryNorm(boundaries, cmap.N)  # type: ignore


def _classify_br_state(df: pd.DataFrame, baseline: float = BASE_ONSET) -> pd.DataFrame:
    """Return a DataFrame with a 'state' column for BR sweep."""
    diff = (df["onset_time"].astype(float) - baseline).abs()
    df = df.copy()
    df["state"] = np.where(
        diff <= 30,
        0,
        np.where(
            diff <= 60,
            1,
            np.where(
                diff <= 90,
                2,
                3,
            ),
        ),
    )
    # set onset time as state now
    df["onset_time"] = df["state"]
    return df


def plot_phase_diagram(
    csv_path: Path,
    tag: str,
    variable: str,
    save: Path,
) -> None:
    """Plot parameter sweep as a heat-map phase diagram."""
    df = _load_sweep(csv_path, tag)
    if tag == "BR":
        df = _classify_br_state(df)
        x_name, y_name = "beta", "r"
        x_label, y_label = r"$\beta$", r"$R$"

    elif tag == "RTm":
        df = _classify_br_state(df)
        x_name, y_name = "r", "tau_m"
        x_label, y_label = r"$R$", r"$\tau_m$"

    elif tag == "AF":
        x_name, y_name = "activity", "flow"
        x_label, y_label = r"$A$", r"$F$"

    else:
        raise ValueError(f"Unknown sweep tag: {tag}")

    x, y, z = _pivot_grid(df, x_name, y_name, variable)
    cmap, norm = _choose_cmap_and_norm(variable)

    fig, ax = plt.subplots(figsize=(1.52, 1.55), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="nearest")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if variable == "state":
        legend_vals = [0, 1, 2, 3]
        legend_txts = [
            "Wet and dewet",
            "Dewet only",
            "Wet only",
            "Lost behavior",
        ]
    else:
        legend_vals = [0, 1, 2, 3]
        legend_txts = [
            "Within 30 min",
            "30 – 60 min",
            "60 – 90 min",
            "> 90 min",
        ]

    handles = [
        Patch(
            facecolor=cmap(norm(v)),  # type: ignore
            label=lbl,
            edgecolor="black",
            linewidth=0.1,
        )
        for v, lbl in zip(legend_vals, legend_txts)
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=0.8,
    )

    # ---------------------------------------------------------------- save ----
    plt.savefig(save, dpi=450, bbox_inches="tight")
    print(f"Figure saved to {save.resolve()}")
    plt.close(fig)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a single Pescoid phase diagram."
    )
    parser.add_argument(
        "--csv", type=Path, help="Combined sweep CSV produced by the scan"
    )
    parser.add_argument(
        "--sweep",
        required=True,
        choices=("AF", "BR", "RTm"),
        help="Which sweep to plot (AF, BR, or RTm)",
    )
    parser.add_argument(
        "--variable",
        "-v",
        default="state",
        help="Column to visualize: 'state' for AF or 'onset_time' for BR / RTm (default: state)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point to plot phase diagram sweep."""
    args = _parse_arguments()
    _set_matplotlib_publication_parameters()
    plot_phase_diagram(
        csv_path=args.csv,
        tag=args.sweep,
        variable=args.variable,
        save=Path(f"phase_diagram_{args.sweep}.svg"),
    )


if __name__ == "__main__":
    main()
