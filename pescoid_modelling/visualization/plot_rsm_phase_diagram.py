"""Plot RSM fitted to a parameter sweep CSV."""

from pathlib import Path

from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore


def plot_rsm_surface(
    model: Pipeline,
    X: np.ndarray,
    label1: str,
    label2: str,
    fname: str,
) -> None:
    """Plot RSM surface with 270 min darkest green, lighter toward 0 and max."""
    N = 250
    x1_f = np.linspace(X[:, 0].min(), X[:, 0].max(), N)
    x2_f = np.linspace(X[:, 1].min(), X[:, 1].max(), N)
    X1g, X2g = np.meshgrid(x1_f, x2_f)
    Z = model.predict(np.c_[X1g.ravel(), X2g.ravel()]).reshape(X1g.shape)  # type: ignore

    # set up colorbar according to exp onset
    vmin = 0.0
    vcenter = 270.0
    vmax = float(np.nanmax(Z))
    n_half = 128
    greens = cm.get_cmap("Greens", n_half)
    colours_low = greens(np.linspace(0, 1, n_half))
    colours_hi = greens(np.linspace(1, 0, n_half))
    custom_cmap = colors.ListedColormap(
        np.vstack([colours_low, colours_hi]), name="GreenPeak"
    )

    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.775, 1.325), constrained_layout=True)
    cs = ax.contourf(
        X1g,
        X2g,
        Z,
        levels=20,
        cmap=custom_cmap,
        norm=norm,
        antialiased=True,
    )

    cbar = fig.colorbar(cs, ax=ax, shrink=0.5, aspect=5)
    cbar.set_ticks([vmin, vcenter, vmax])
    cbar.set_ticklabels([f"{int(vmin)}", f"{int(vcenter)}", f"{int(vmax)}"])
    cbar.set_label("Mesoderm onset")

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    plt.savefig(fname, dpi=450, bbox_inches="tight")
    print("figure saved to", Path(fname).resolve())
    plt.close(fig)
