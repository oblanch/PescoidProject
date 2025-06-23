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

_LOG_AXES_PARAMS = {"diffusivity", "gamma", "m_sensitivity"}


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    cores = psutil.cpu_count(logical=False)
    if cores is None or cores <= 1:
        return 1
    return cores - 1


def get_dewetting_onset(
    times: np.ndarray,
    boundary_positions: np.ndarray,
    slope_threshold: float = -1e-1,
) -> float:
    """Time when dR/dt first drops below slope_threshold."""
    if boundary_positions.size < 2:
        return np.nan
    dRdt = np.gradient(boundary_positions, times)
    mask = dRdt < slope_threshold
    return float(times[np.argmax(mask)]) if np.any(mask) else np.nan


def make_reference_timeseries(
    stats_df: pd.DataFrame,
    time_key: str = "TIME",
    radius_key: str = "mean_radius_norm",
    frac_key: str = "mean_fraction",
    window_length: int = 7,
    polyorder: int = 1,
    frac_of_peak: float = 0.10,
    plateau_frac: float = 1.00,
    slope_threshold: float = -1e-1,
    outfile: str = "reference_timeseries.npz",
) -> None:
    """Build a rigid cubic [0→1] reference curve that crosses frac_of_peak at
    dewetting onset and reaches 1.0 at plateau time.
    """
    times = stats_df[time_key].to_numpy(dtype=float)
    raw_radius = stats_df[radius_key].to_numpy(dtype=float)
    raw_frac = stats_df[frac_key].to_numpy(dtype=float)

    smoothed_radius = savgol_filter(raw_radius, window_length, polyorder)
    smoothed_frac = savgol_filter(raw_frac, window_length, polyorder)

    dewetting_time = get_dewetting_onset(times, smoothed_radius, slope_threshold)
    if np.isnan(dewetting_time):
        raise RuntimeError("Could not detect dewetting onset in the radius trace.")

    peak_frac_value = float(np.nanmax(smoothed_frac))
    plateau_threshold = plateau_frac * peak_frac_value
    plateau_indices = np.where(smoothed_frac >= plateau_threshold)[0]
    if plateau_indices.size:
        plateau_time = float(times[plateau_indices[0]])
    else:
        plateau_time = float(times[-1])

    if plateau_time <= dewetting_time:
        raise RuntimeError("Plateau time must be after dewetting onset.")

    cubic_coeffs = [2.0, -3.0, 0.0, frac_of_peak]
    all_roots = np.roots(cubic_coeffs)
    valid_roots = [
        root.real
        for root in all_roots
        if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0
    ]
    if not valid_roots:
        raise RuntimeError(f"No valid cubic root for frac_of_peak={frac_of_peak}")
    s_onset = valid_roots[0]

    t_start = (dewetting_time - s_onset * plateau_time) / (1.0 - s_onset)

    normalized_time = (times - t_start) / (plateau_time - t_start)
    normalized_time = np.clip(normalized_time, 0.0, 1.0)
    mesoderm_fraction = 3.0 * normalized_time**2 - 2.0 * normalized_time**3

    np.savez(
        outfile,
        time=times,
        tissue_size=smoothed_radius,
        mesoderm_fraction=mesoderm_fraction,
        dewetting_time=np.asarray([dewetting_time]),
        plateau_time=np.asarray([plateau_time]),
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

    lower_vec = [float(cma_params["bounds"]["lower"][k]) for k in _ORDER]  # type: ignore
    upper_vec = [float(cma_params["bounds"]["upper"][k]) for k in _ORDER]  # type: ignore

    scaler = ParamScaler(
        lower=lower_vec,
        upper=upper_vec,
        log_mask=[name in _LOG_AXES_PARAMS for name in _ORDER],
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
        f"  c_diffusivity: {params['c_diffusivity']}",
        "  morphogen_decay: 0.05",
        "  gaussian_width: 0.15",
        f"  morphogen_feedback: {params['morphogen_feedback']}",
        "  feedback_mode: active_stress",
    ]

    print("\n".join(lines))
