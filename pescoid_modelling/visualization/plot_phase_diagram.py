"""Visualize parameter NxN sweep as a phase diagram heat-map."""

import argparse
from pathlib import Path
from typing import Tuple

from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore


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
        boundaries = np.arange(-0.5, 4, 1)
        norm = BoundaryNorm(boundaries, cmap.N)
    else:
        (
            cmap,
            norm,
        ) = (
            plt.cm.viridis,  # type: ignore
            None,
        )
    return cmap, norm  # type: ignore


def plot_phase_diagram(
    csv_path: Path,
    tag: str,
    variable: str,
    save: Path,
) -> None:
    """Plot parameter sweep as a heat-map phase diagram."""
    df = _load_sweep(csv_path, tag)
    if variable not in df.columns:
        raise ValueError(f"Column '{variable}' not found in '{tag}' sweep")

    if tag == "AF":
        x_name, y_name = "activity", "flow"
        x_label, y_label = r"$A$", r"$F$"
    else:
        x_name, y_name = "beta", "R"
        x_label, y_label = r"$\beta$", r"$R$"
    x, y, z = _pivot_grid(df, x_name, y_name, variable)
    cmap, norm = _choose_cmap_and_norm(variable)

    fig, ax = plt.subplots(figsize=(2.4, 1.65), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="nearest")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Adjust descriptions for states
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3], shrink=0.325, aspect=5)
    if variable == "state":
        cbar.set_ticklabels(
            [
                "Wet and dewet",
                "Dewet only",
                "Wet only",
                "Deviated",
            ]
        )
        cbar.ax.yaxis.set_ticks_position("none")
        cbar.ax.tick_params(which="both", length=0)
        cbar.set_label("")
        cbar.ax.invert_yaxis()

    plt.savefig(save, dpi=450, bbox_inches="tight")
    print(f"Figure saved to {save.resolve()}")
    plt.close(fig)


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualise a single Pescoid phase diagram."
    )
    parser.add_argument(
        "csv", type=Path, help="Combined sweep CSV produced by the scan"
    )
    parser.add_argument(
        "--sweep",
        required=True,
        choices=("AF", "BR"),
        help="Which sweep to plot (AF or BR)",
    )
    parser.add_argument(
        "--variable",
        "-v",
        default="state",
        help="Column to visualise (default: state)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point to plot phase diagram sweep."""
    args = _parse_arguments()
    plot_phase_diagram(
        csv_path=args.csv,
        tag=args.sweep,
        variable=args.variable,
        save=args.save or Path(f"phase_diagram_{args.sweep}_{args.variable}.png"),
    )


if __name__ == "__main__":
    main()
