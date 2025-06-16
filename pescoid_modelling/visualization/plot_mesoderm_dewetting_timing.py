#!/usr/bin/env python3

"""Plot mesoderm and dewetting timing across a parameter sweep."

NOTE: assumes that a param sweep has already been run and that each
simulation result .npz is in its own directory.
"""

from pathlib import Path
from typing import Dict, List

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters

SLOPE_THRESHOLD = -5e-4
FRAC_OF_PEAK = 0.10


def get_mesoderm_onset(
    times: np.ndarray,
    meso_frac: np.ndarray,
    frac_of_peak: float = FRAC_OF_PEAK,
) -> float:
    """Time at which mesoderm fraction reaches the given fraction of its
    peak value.
    """
    if meso_frac.size == 0 or np.all(np.isnan(meso_frac)):
        return np.nan

    peak = meso_frac.max()
    if peak == 0.0:
        return np.nan

    thresh = frac_of_peak * peak
    idx = np.argmax(meso_frac >= thresh)
    if meso_frac[idx] < thresh:
        return np.nan
    return float(times[idx])


def get_dewetting_onset(
    times: np.ndarray,
    boundary_positions: np.ndarray,
    slope_threshold: float = SLOPE_THRESHOLD,
) -> float:
    """Dewetting onset when the derivative of the boundary position (dR/dt) is
    less than the slope threshold.
    """
    if boundary_positions.size < 2:
        return np.nan

    dRdt = np.gradient(boundary_positions, times)
    mask = dRdt < slope_threshold
    if not np.any(mask):
        return np.nan
    return float(times[np.argmax(mask)])


def _load_simulation_data(path: Path, minutes_per_generation: float = 30.0) -> dict:
    """Load a single simulation_results.npz and return a dictionary."""
    data = np.load(path, allow_pickle=True)
    times = data["time"] * minutes_per_generation

    sim_data = {
        "run_name": path.parent.name,
        "npz_path": str(path),
        "times": times,
        "meso_frac": data["mesoderm_fraction"],
        "radius": data["tissue_size"],
        "t_mesoderm": get_mesoderm_onset(times, data["mesoderm_fraction"]),
        "t_dewetting": get_dewetting_onset(times, data["tissue_size"]),
    }
    sim_data["delay"] = (
        sim_data["t_dewetting"] - sim_data["t_mesoderm"]
        if np.isfinite(sim_data["t_mesoderm"]) and np.isfinite(sim_data["t_dewetting"])
        else np.nan
    )
    return sim_data


def _load_simulation_sweep(sim_root: str | Path) -> pd.DataFrame:
    """Recursively search root dir for simulation_results.npz files, load each,
    and return a dataframe Skips files that raise exceptions or contain NaNs
    everywhere.
    """
    sim_root = Path(sim_root)
    rows: List[dict] = []

    for npz_path in sim_root.rglob("simulation_results.npz"):
        try:
            rows.append(_load_simulation_data(npz_path))
        except Exception as exc:
            print(f"[skip] {npz_path} ({exc})")

    df = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)
    return df


def load_experimental_onset(
    npz_path: str,
    frac_of_peak: float = FRAC_OF_PEAK,
    slope_threshold: float = SLOPE_THRESHOLD,
) -> Dict[str, float]:
    """Load experimental reference time series and compute mesoderm and
    dewetting onset.
    """
    data = np.load(npz_path, allow_pickle=True)
    times = data["time"]
    meso_frac = data["mesoderm_fraction"]
    radius = data["tissue_size"]

    peak = meso_frac.max() if meso_frac.size else 0.0
    if peak > 0:
        thresh = frac_of_peak * peak
        idx_m = np.argmax(meso_frac >= thresh)
        t_m = float(times[idx_m])
    else:
        t_m = np.nan

    if radius.size > 1:
        dRdt = np.gradient(radius, times)
        mask = dRdt < slope_threshold
        if np.any(mask):
            idx_d = np.argmax(mask)
            t_d = float(times[idx_d])
        else:
            t_d = np.nan
    else:
        t_d = np.nan

    delay = float(t_d - t_m) if np.isfinite(t_m) and np.isfinite(t_d) else np.nan
    return {"t_mesoderm": t_m, "t_dewetting": t_d, "delay": delay}


def plot_mesoderm_dewetting_timing(
    df: pd.DataFrame,
    annotate: bool = True,
    experimental_onset: Dict[str, float] | None = None,
) -> Figure:
    """
    Scatter plot of dewetting vs. mesoderm delay.
    X = run index, Y = t_d - t_m (min).
    Positive → mesoderm onset before dewetting.
    """
    color_map = {
        "beta": "tab:red",
        "gamma": "tab:blue",
        "r": "tab:green",
        r"$x_0$": "k",
    }
    fig, ax = plt.subplots(figsize=(2.8, 1.85))

    types, vals = [], []
    for rn in df["run_name"]:
        parts = rn.split("_")
        if parts[0] == "sweep":
            types.append(parts[1])
            vals.append(float(parts[-1]))
        else:
            types.append(r"$x_0$")
            vals.append(np.nan)
    df = df.assign(_type=types, _val=vals)  # type: ignore

    texts = []
    for ptype, group in df.groupby("_type"):
        c = color_map.get(ptype, "gray")  # type: ignore
        ax.scatter(
            group.index,
            group["delay"],
            c=c,
            s=10,
            edgecolor="none",
            label=ptype,
            zorder=3,
        )
        if annotate and ptype != r"$x_0$":
            for x, y, v in zip(group.index, group["delay"], group["_val"]):
                txt = ax.text(
                    x,
                    y + 5,
                    f"{v:g}",
                    ha="center",
                    va="bottom",
                )
                texts.append(txt)

    # add experimental point
    if experimental_onset is not None:
        idx_exp = df.index.max() + 1
        delay_exp = experimental_onset.get("delay", np.nan)
        ax.scatter(
            idx_exp,
            delay_exp,
            c="purple",
            s=10,
            edgecolor="none",
            marker="^",
            label="experimental",
            zorder=4,
        )
        if annotate:
            ax.text(
                idx_exp,
                delay_exp + 5,
                "exp",
                ha="center",
                va="bottom",
            )

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_xlabel("Simulation index")
    ax.set_ylabel(r"$t_d - t_m$ (min)" "\nDewetting onset - Mesoderm onset")

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, handletextpad=0.1
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _set_matplotlib_publication_parameters()
    experimental = "/Users/steveho/PescoidProject/data/reference_timeseries.npz"
    df = _load_simulation_sweep(".")
    experimental_onset = load_experimental_onset(experimental)

    fig = plot_mesoderm_dewetting_timing(df, experimental_onset=experimental_onset)
    fig.tight_layout()
    fig.savefig("mesoderm_dewetting_timing.pdf", bbox_inches="tight")
    plt.close(fig)
