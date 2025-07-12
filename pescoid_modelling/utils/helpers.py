"""Utility helper functions for pescoid modelling."""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
import psutil  # type: ignore
from scipy.signal import savgol_filter  # type: ignore
import yaml  # type: ignore

from pescoid_modelling.utils.config import _load_yaml
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.constants import ONSET_THRESH
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


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        sys.exit(f"YAML not found: {path}")


def calculate_onset_time(
    time_array: np.ndarray,
    mesoderm_array: np.ndarray,
    threshold: float = ONSET_THRESH,
) -> Optional[float]:
    """Calculate time when mesoderm reaches threshold of its maximum."""
    max_meso = np.max(mesoderm_array)
    if max_meso <= 0:
        return None

    onset_value = threshold * max_meso
    onset_mask = mesoderm_array >= onset_value

    if not np.any(onset_mask):
        return None

    onset_idx = np.where(onset_mask)[0][0]
    return float(time_array[onset_idx])


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
    plateau_frac: float = 1.00,
    outfile: str = "control_reference_timeseries.npz",
) -> None:
    """Builds a reference timeseries from experimental trajectories."""
    times = stats_df[time_key].to_numpy(dtype=float)
    raw_radius = stats_df[radius_key].to_numpy(dtype=float)
    raw_frac = stats_df[frac_key].to_numpy(dtype=float)

    smoothed_radius = savgol_filter(raw_radius, window_length, polyorder)
    smoothed_frac = savgol_filter(raw_frac, window_length, polyorder)

    peak_frac_value = float(np.nanmax(smoothed_frac))
    plateau_threshold = plateau_frac * peak_frac_value
    plateau_indices = np.where(smoothed_frac >= plateau_threshold)[0]
    if plateau_indices.size:
        plateau_time = float(times[plateau_indices[0]])
    else:
        plateau_time = float(times[-1])

    np.savez(
        outfile,
        time=times,
        tissue_size=smoothed_radius,
        mesoderm_fraction=smoothed_frac,
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
        >>> vec_str = "
        ...    0.00994556580728923 0.15341494818583273
        ...    0.7967177922215253 0.45192053907278784
        ...    0.7044615676175338 0.5995302805358877
        ...    0.30068249193679086 0.2766307616670052
        ...    0.6307550490189227
        ...    "
        >>> results = _vecs_to_simulation_config(
        ...     vec_str,
        ...     "PescoidProject/configs/optimization_config.yaml",
        ...     scaler
        ... )
        ... simulation:
        ...   delta_t: 0.01
        ...   total_hours: 12.0
        ...   domain_length: 10.0
        ...   dx_interval: 0.001
        ...   diffusivity: 0.00010161355653022654
        ...   m_diffusivity: 2e-3
        ...   tau_m: 8.577024545550678
        ...   flow: 0.15341494818583273
        ...   activity: 0.7340154108557804
        ...   beta: 1.1990605610717755
        ...   gamma: 0.1936119599257585
        ...   sigma_c: 0.1
        ...   r: 1.5034124596839544
        ...   rho_sensitivity: 0.0
        ...   m_sensitivity: 0.11451830806904184
        ...   morphogen_feedback: 1.6307550490189227
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
        "  m_diffusivity: 2e-3",
        f"  tau_m: {params['tau_m']}",
        f"  flow: {params['flow']}",
        f"  activity: {params['activity']}",
        f"  beta: {params['beta']}",
        f"  gamma: {params['gamma']}",
        "  sigma_c: 0.1",
        f"  r: {params['r']}",
        "  rho_sensitivity: 0.0",
        f"  m_sensitivity: {params['m_sensitivity']}",
        f"  morphogen_feedback: {params['morphogen_feedback']}",
        "  feedback_mode: active_stress",
    ]

    print("\n".join(lines))
