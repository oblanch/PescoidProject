"""Utility helper functions for pescoid modelling."""

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
import psutil  # type: ignore
from scipy.signal import savgol_filter  # type: ignore

from pescoid_modelling.utils.config import _load_yaml
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.parameter_scaler import ParamScaler


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    cores = psutil.cpu_count(logical=False)
    if cores is None or cores <= 1:
        return 1
    return cores - 1


def make_reference_timeseries(
    stats_df: pd.DataFrame,
    time_key: str = "TIME",
    radius_key: str = "mean_radius_norm",
    frac_key: str = "mean_fraction",
    window_length: int = 7,
    polyorder: int = 1,
    outfile: str = "reference_timeseries.npz",
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
        outfile,
        time=time,
        tissue_size=radius_sm,
        mesoderm_fraction=mesoderm_fraction,
    )


def _least_squares_rescale(
    sim: np.ndarray,
    ref: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Compute & apply the least-squares amplitude scale to match reference.

    Finds the scalar s minimizing ‖reference - s·simulation‖₂, then returns
    (simulation * s, s).
    """
    denom: float = float(np.dot(sim, sim)) + eps
    if denom < eps:
        return sim, 1.0

    scale: float = float(np.dot(sim, ref) / denom)
    return scale * sim, scale


def _vecs_to_simulation_config(vec_str: str, config_path: Union[str, Path]) -> None:
    """Parse a normalized-vec string, convert to back to physical parameters and
    emit the simulation config block.

    Examples:
        >>> vec_str = (
        ...     "0.28525489144682264 0.00909736056224767 "
        ...     "0.774647354051897 0.9000926897671878 "
        ...     "0.20953666405892174 0.006077833681571175 "
        ...     "0.034680941325624694 0.9964689742010449"
        ... )
        >>> result = _vecs_to_simulation_config(
        ...     vec_str,
        ...     "PescoidProject/configs/optimization_config.yaml",
        ...     scaler
        ... )
        >>> print(result)
        ...   delta_t: 0.01
        ...   total_hours: 12.0
        ...   domain_length: 10.0
        ...   dx_interval: 0.001
        ...   diffusivity: 7.174054545113444e-05
        ...   m_diffusivity: 1e-3
        ...   tau_m: 6.2197140970099865
        ...   flow: 0.09097360562247671
        ...   activity: 1.0476833202946088
        ...   beta: 0.030389168407855875
        ...   gamma: 0.39844718147251246
        ...   sigma_c: 0.0
        ...   r: 0.6936188265124938
        ...   rho_sensitivity: 0.0
        ...   m_sensitivity: 0.09838705211988935
        ...   feedback_mode: active_stress
    """
    config = _load_yaml(config_path)
    cma_params = config.get("cma")

    log_axes = {"diffusivity", "gamma", "m_sensitivity"}
    lower_vec = [float(cma_params["bounds"]["lower"][k]) for k in _ORDER]  # type: ignore
    upper_vec = [float(cma_params["bounds"]["upper"][k]) for k in _ORDER]  # type: ignore

    scaler = ParamScaler(
        lower=lower_vec,
        upper=upper_vec,
        log_mask=[name in log_axes for name in _ORDER],
    )
    normalized = [float(x) for x in vec_str.strip().split()]
    physical = scaler.to_physical(normalized)

    params = dict(zip(_ORDER, physical))

    lines = [
        "simulation:",
        "  delta_t: 0.01",
        "  total_hours: 12.0",
        "  domain_length: 10.0",
        "  dx_interval: 0.001",
        f"  diffusivity: {params['diffusivity']}",
        "  m_diffusivity: 1e-3",
        f"  tau_m: {params['tau_m']}",
        f"  flow: {params['flow']}",
        f"  activity: {params['activity']}",
        f"  beta: {params['beta']}",
        f"  gamma: {params['gamma']}",
        "  sigma_c: 0.0",
        f"  r: {params['r']}",
        "  rho_sensitivity: 0.0",
        f"  m_sensitivity: {params['m_sensitivity']}",
        "  feedback_mode: active_stress",
    ]

    print("\n".join(lines))
