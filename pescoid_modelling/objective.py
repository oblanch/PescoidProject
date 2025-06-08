"""Time series optimization objective for the pescoid model."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, MutableMapping, Optional, Tuple

import numpy as np

from pescoid_modelling.utils.constants import EMA_ALPHA
from pescoid_modelling.utils.constants import EPS
from pescoid_modelling.utils.constants import JITTER
from pescoid_modelling.utils.constants import OPTIM_PENALTY

_RUNNING_EMA: Dict[str, Optional[float]] = defaultdict(lambda: None)


@dataclass
class ReferenceTrajectories:
    """Dense trajectories that drive the optimizer."""

    time: np.ndarray
    tissue_size: np.ndarray  # L_exp(t)
    mesoderm_fraction: np.ndarray  # M_exp(t)

    def __post_init__(self):
        """Validate that all arrays have the same length."""
        if not (len(self.time) == len(self.tissue_size) == len(self.mesoderm_fraction)):
            raise ValueError("All trajectory arrays must have the same length")

        if not np.all(np.diff(self.time) > 0):
            raise ValueError("Time array must be monotonically increasing")


def _require_exponential_moving_average(
    ema_dict: MutableMapping[str, Optional[float]] | None = None,
) -> MutableMapping[str, Optional[float]]:
    """Ensure an EMA dictionary is available, creating a new one if necessary."""
    if ema_dict is None:
        ema_dict = defaultdict(lambda: None)
    return ema_dict


def _invalid_fitness() -> float:
    """Return a fitness for failed or too-short simulations. Adds a small jitter
    term to avoid premature termination.
    """
    return OPTIM_PENALTY * (1.0 + JITTER * np.random.randn())


def _calculate_normalization_scales(
    experimental_data: ReferenceTrajectories,
) -> Tuple[float, float]:
    """Calculate scales for standard deviation-based normalization.

    Returns:
      Tuple of (tissue_std, mesoderm_std)
    """
    tissue_std = float(np.std(experimental_data.tissue_size))
    mesoderm_std = float(np.std(experimental_data.mesoderm_fraction))

    # Protect against zero std (constant signals)
    tissue_std = tissue_std if tissue_std > 0 else 1.0
    mesoderm_std = mesoderm_std if mesoderm_std > 0 else 1.0

    return tissue_std, mesoderm_std


def _calculate_trajectory_mismatch(
    sim_interpolated: np.ndarray,
    exp_values: np.ndarray,
    normalization_scale: float,
) -> float:
    """Calculate L2 norm squared between interpolated simulation and
    experimental values with normalization.
    """
    sim_normalized = sim_interpolated / normalization_scale
    exp_normalized = exp_values / normalization_scale
    return float(np.linalg.norm(sim_normalized - exp_normalized) ** 2)


def _interpolate_simulation_to_experimental_timepoints(
    sim_time_generations: np.ndarray,
    sim_values: np.ndarray,
    exp_time_minutes: np.ndarray,
    minutes_per_generation: float = 30.0,
) -> np.ndarray:
    """Interpolate simulation data onto reference time grid in minutes.

    Returns:
      Interpolated simulation values at reference time points

    Raises:
      ValueError: If no reference time points are within simulation range
    """
    sim_time_minutes = sim_time_generations * minutes_per_generation
    valid_exp_mask = exp_time_minutes <= sim_time_minutes[-1]

    if not np.any(valid_exp_mask):
        raise ValueError(
            f"No reference time points within simulation range "
            f"(sim: 0-{sim_time_minutes[-1]:.1f} min, "
            f"exp: {exp_time_minutes[0]:.1f}-{exp_time_minutes[-1]:.1f} min)"
        )

    exp_time_valid = exp_time_minutes[valid_exp_mask]
    return np.interp(exp_time_valid, sim_time_minutes, sim_values)


def _ema_update(
    tag: str,
    value: float,
    ema_dict: MutableMapping[str, Optional[float]],
    alpha: float = EMA_ALPHA,
) -> float:
    """Exponential-moving-average update."""
    prev = ema_dict.get(tag)
    ema_dict[tag] = value if prev is None else (1.0 - alpha) * prev + alpha * value
    return ema_dict[tag]  # type: ignore


def _compute_dynamic_weights(
    losses: Dict[str, float],
    ema_dict: MutableMapping[str, Optional[float]],
    eps: float = EPS,
    w_clip: Tuple[float, float] = (0.05, 1.95),
) -> Dict[str, float]:
    """Inverse-EMA weighting."""
    for tag, val in losses.items():
        _ema_update(tag, val, ema_dict)

    inv: Dict[str, float] = {}
    for tag, cur_loss in losses.items():
        ema_val = ema_dict[tag]
        baseline = cur_loss if ema_val is None else ema_val
        inv[tag] = 1.0 / max(baseline, eps)

    total_inv = sum(inv.values())
    if total_inv < eps:
        return {k: 1.0 for k in losses}

    scale = 2.0 / total_inv
    weights = {k: np.clip(v * scale, *w_clip) for k, v in inv.items()}
    return weights


def optimization_objective(
    results: Dict[str, np.ndarray],
    experimental_data: ReferenceTrajectories,
    tissue_std: Optional[float] = None,
    mesoderm_std: Optional[float] = None,
    minutes_per_generation: float = 30.0,
    *,
    ema_dict: MutableMapping[str, Optional[float]] | None = None,
) -> float:
    """Fitness cost function based on L2 norm of the mismatch between simulated
    and reference trajectories of tissue size L(t) and mesoderm signal M(t).

    Args:
      results: Dictionary containing simulation results with keys:
        - "time": time points (in generation units)
        - "tissue_size": L_sim(t)
        - "mesoderm_fraction": M_sim(t)
      experimental_data: Experimental trajectories to match (time in minutes).
      minutes_per_generation: Conversion factor from generation units to
      minutes.

    Returns:
      L2 error value or calls `_invalid_fitness()` if results are invalid or
      simulation is too short.
    """
    if not results:
        return _invalid_fitness()

    ema_dict = _require_exponential_moving_average(ema_dict)

    t_sim = results.get("time", np.array([]))
    L_sim = results.get("tissue_size", np.array([]))
    M_sim = results.get("mesoderm_fraction", np.array([]))

    if not _check_simulation_results(t_sim, L_sim, M_sim):
        return _invalid_fitness()

    try:
        if tissue_std is None or mesoderm_std is None:
            tissue_std, mesoderm_std = _calculate_normalization_scales(
                experimental_data
            )

        L_sim_on_exp = _interpolate_simulation_to_experimental_timepoints(
            t_sim, L_sim, experimental_data.time, minutes_per_generation
        )
        M_sim_on_exp = _interpolate_simulation_to_experimental_timepoints(
            t_sim, M_sim, experimental_data.time, minutes_per_generation
        )

        # Get reference values
        sim_time_minutes = t_sim * minutes_per_generation
        if not _check_simulation_length(sim_time_minutes, experimental_data):
            return _invalid_fitness()

        valid_exp_mask = experimental_data.time <= sim_time_minutes[-1]
        exp_tissue_valid = experimental_data.tissue_size[valid_exp_mask]
        exp_meso_valid = experimental_data.mesoderm_fraction[valid_exp_mask]

        # Loss
        l2_tissue = _calculate_trajectory_mismatch(
            sim_interpolated=L_sim_on_exp,
            exp_values=exp_tissue_valid,
            normalization_scale=tissue_std,
        )
        l2_meso = _calculate_trajectory_mismatch(
            sim_interpolated=M_sim_on_exp,
            exp_values=exp_meso_valid,
            normalization_scale=mesoderm_std,
        )

        losses = {"tissue": l2_tissue, "mesoderm": l2_meso}
        weights = _compute_dynamic_weights(losses=losses, ema_dict=ema_dict)  # type: ignore
        return weights["tissue"] * l2_tissue + weights["mesoderm"] * l2_meso

    except ValueError as e:
        raise ValueError(
            "Simulation results are not compatible with reference data."
        ) from e


def _check_simulation_length(
    sim_time_minutes: np.ndarray,
    experimental_data: ReferenceTrajectories,
) -> bool:
    """Check that simulation runs long enough to avoid comparing premature
    curves.
    """
    if experimental_data.time[-1] - sim_time_minutes[-1] > 1e-3:
        return False
    return True


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
