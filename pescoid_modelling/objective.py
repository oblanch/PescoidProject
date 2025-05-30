"""Time series optimization objective for the pescoid model."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ExperimentalTrajectories:
    """Dense trajectories that drive the optimizer."""

    time: np.ndarray
    tissue_size: np.ndarray  # L_exp(t)
    mesoderm_signal: np.ndarray  # M_exp(t)

    def __post_init__(self):
        """Validate that all arrays have the same length."""
        if not (len(self.time) == len(self.tissue_size) == len(self.mesoderm_signal)):
            raise ValueError("All trajectory arrays must have the same length")

        if not np.all(np.diff(self.time) > 0):
            raise ValueError("Time array must be monotonically increasing")


def interpolate_simulation_to_experimental_timepoints(
    sim_time_generations: np.ndarray,
    sim_values: np.ndarray,
    exp_time_minutes: np.ndarray,
    minutes_per_generation: float = 30.0,
) -> np.ndarray:
    """Interpolate simulation data onto experimental time grid in minutes.

    Returns:
      Interpolated simulation values at experimental time points

    Raises:
      ValueError: If no experimental time points are within simulation range
    """
    sim_time_minutes = sim_time_generations * minutes_per_generation
    valid_exp_mask = exp_time_minutes <= sim_time_minutes[-1]

    if not np.any(valid_exp_mask):
        raise ValueError(
            f"No experimental time points within simulation range "
            f"(sim: 0-{sim_time_minutes[-1]:.1f} min, "
            f"exp: {exp_time_minutes[0]:.1f}-{exp_time_minutes[-1]:.1f} min)"
        )

    exp_time_valid = exp_time_minutes[valid_exp_mask]
    return np.interp(exp_time_valid, sim_time_minutes, sim_values)


def calculate_trajectory_mismatch(
    sim_interpolated: np.ndarray,
    exp_values: np.ndarray,
) -> float:
    """Calculate L2 norm between interpolated simulation and experimental
    values.
    """
    return float(np.linalg.norm(sim_interpolated - exp_values) ** 2)


def optimization_objective(
    results: Dict[str, np.ndarray],
    experimental_data: ExperimentalTrajectories,
    minutes_per_generation: float = 30.0,
) -> float:
    """Fitness cost function based on L2 norm of the mismatch between simulated
    and experimental trajectories of tissue size L(t) and mesoderm signal M(t).

    Args:
      results: Dictionary containing simulation results with keys:
        - "time": time points (in generation units)
        - "tissue_size": L_sim(t)
        - "mesoderm_signal": M_sim(t)
      experimental_data: Experimental trajectories to match (time in minutes).
      minutes_per_generation: Conversion factor from generation units to
      minutes.

    Returns:
      L2 error value. Returns 1e9 for invalid/failed simulations.
    """
    if not results:
        return 1e9

    t_sim = results.get("time", np.array([]))
    L_sim = results.get("tissue_size", np.array([]))
    M_sim = results.get("mesoderm_signal", np.array([]))

    if not _check_simulation_results(t_sim, L_sim, M_sim):
        return 1e9

    try:
        L_sim_on_exp = interpolate_simulation_to_experimental_timepoints(
            t_sim, L_sim, experimental_data.time, minutes_per_generation
        )
        M_sim_on_exp = interpolate_simulation_to_experimental_timepoints(
            t_sim, M_sim, experimental_data.time, minutes_per_generation
        )

        # Get corresponding experimental values
        sim_time_minutes = t_sim * minutes_per_generation
        valid_exp_mask = experimental_data.time <= sim_time_minutes[-1]
        exp_tissue_valid = experimental_data.tissue_size[valid_exp_mask]
        exp_meso_valid = experimental_data.mesoderm_signal[valid_exp_mask]

        # Loss
        l2_tissue = calculate_trajectory_mismatch(L_sim_on_exp, exp_tissue_valid)
        l2_meso = calculate_trajectory_mismatch(M_sim_on_exp, exp_meso_valid)

        return l2_tissue + l2_meso

    except ValueError as e:
        raise ValueError(
            "Simulation results are not compatible with experimental data."
        ) from e


def _check_simulation_results(
    t_sim: np.ndarray,
    L_sim: np.ndarray,
    M_sim: np.ndarray,
) -> bool:
    """Ensure simulation results are valid."""
    if (
        len(t_sim) == 0
        or len(L_sim) == 0
        or len(M_sim) == 0
        or len(t_sim) != len(L_sim)
        or len(t_sim) != len(M_sim)
    ):
        return False

    if (
        np.any(~np.isfinite(t_sim))
        or np.any(~np.isfinite(L_sim))
        or np.any(~np.isfinite(M_sim))
    ):
        return False

    return True
