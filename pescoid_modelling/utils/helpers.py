"""Utility helper functions for pescoid modelling."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd  # type: ignore
from scipy.signal import savgol_filter  # type: ignore


def make_reference_timeseries(
    stats_df: pd.DataFrame,
    time_key: str = "TIME",
    radius_key: str = "mean_radius_norm",
    frac_key: str = "mean_fraction",
    window_length: int = 7,
    polyorder: int = 1,
) -> None:
    """Create a reference timeseries from experimental data and save as .npz for
    use. Smooths the trajectories, finds the max value, and populates the rest
    of the array with the max to produce a sigmoidal-like plateau.
    """
    time: np.ndarray = stats_df[time_key].to_numpy(dtype=float)
    radius: np.ndarray = stats_df[radius_key].to_numpy(dtype=float)
    fraction: np.ndarray = stats_df[frac_key].to_numpy(dtype=float)

    radius_sm = savgol_filter(radius, window_length, polyorder)
    meso_sm = savgol_filter(fraction, window_length, polyorder)

    max_idx = int(np.argmax(meso_sm))
    mesoderm_fraction = meso_sm.copy()
    mesoderm_fraction[max_idx:] = meso_sm[max_idx]

    np.savez(
        "reference_timeseries.npz",
        time=time,
        tissue_size=radius_sm,
        mesoderm_fraction=mesoderm_fraction,
    )
