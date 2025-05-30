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


def optimization_objective(
    results: Dict[str, np.ndarray],
    experimental_data: ExperimentalTrajectories,
) -> float:
    """Fitness cost function based on L2 norm of the mismatch between simulated
    and experimental trajectories of tissue size L(t) and mesoderm signal M(t).

    Args:
      results: Dictionary containing simulation results with keys:
        - "time": time points
        - "tissue_size": L_sim(t)
        - "mesoderm_signal": M_sim(t)
      experimental_data: Experimental trajectories to match.

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

    # Interpolate simulation data onto experimental time grid
    L_sim_on_exp = np.interp(experimental_data.time, t_sim, L_sim)
    M_sim_on_exp = np.interp(experimental_data.time, t_sim, M_sim)

    l2_tissue = np.linalg.norm(L_sim_on_exp - experimental_data.tissue_size) ** 2
    l2_meso = np.linalg.norm(M_sim_on_exp - experimental_data.mesoderm_signal) ** 2

    return float(l2_tissue + l2_meso)


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
