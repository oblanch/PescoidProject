"""Time series optimization objective for the pescoid fluid dynamics model."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, MutableMapping, Optional, Tuple

import numpy as np

from pescoid_modelling.utils.constants import ALIGN_TOL
from pescoid_modelling.utils.constants import EMA_ALPHA
from pescoid_modelling.utils.constants import EPS
from pescoid_modelling.utils.constants import GROWTH_MIN
from pescoid_modelling.utils.constants import JITTER
from pescoid_modelling.utils.constants import MIN_PEAK_TIME
from pescoid_modelling.utils.constants import ONSET_THRESH
from pescoid_modelling.utils.constants import ONSET_TIME_SCALE
from pescoid_modelling.utils.constants import OPTIM_PENALTY
from pescoid_modelling.utils.constants import PEAK_DROP_MIN
from pescoid_modelling.utils.constants import SLOPE_THRESHOLD
from pescoid_modelling.utils.constants import WALL_TOL
from pescoid_modelling.utils.helpers import calculate_onset_time

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
    """Ensure an EMA dictionary is available, creating a new one if
    necessary.
    """
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


def calculate_onset_loss(
    sim_onset_time: Optional[float],
    ref_onset_time: Optional[float],
    time_scale: float = ONSET_TIME_SCALE,
) -> float:
    """Calculate loss based on onset time difference."""
    if ref_onset_time is None or sim_onset_time is None:
        return 1e9

    time_diff = (sim_onset_time - ref_onset_time) / time_scale
    return time_diff**2


def extract_simulation_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract key metrics from simulation results to determine if simulation is
    exhibiting system behavior.
    """
    t = results["time"] * 30.0
    radius = results["tissue_size"]
    mezzo = results["mesoderm_fraction"]
    edge_x = results.get("boundary_positions", radius)

    # Peak
    peak_idx = int(radius.argmax())
    peak_time = t[peak_idx]
    peak_radius = float(radius[peak_idx])

    # Wetting and dewetting
    growth_factor = peak_radius / radius[0] if radius[0] > 0 else 0.0
    post_peak_min = (
        float(radius[peak_idx:].min()) if peak_idx < len(radius) else peak_radius
    )
    drop_frac = (peak_radius - post_peak_min) / peak_radius if peak_radius else 0.0

    # Boundary position
    domain_length = (
        results.get("domain_length", [10.0])[0] if "domain_length" in results else 10.0
    )
    edge_at_peak = edge_x[peak_idx] if peak_idx < len(edge_x) else edge_x[-1]
    half_domain = domain_length / 2
    wall_position = edge_at_peak / half_domain

    # Slope over entire post-peak period
    if peak_idx < len(radius) - 1:
        a, _ = np.polyfit(t[peak_idx:], radius[peak_idx:], 1)
        slope = float(a)
    else:
        slope = 0.0

    final_drop_frac = (peak_radius - radius[-1]) / peak_radius if peak_radius else 0.0
    onset_time = calculate_onset_time(t, mezzo, ONSET_THRESH)

    return {
        "tissue_size": results["tissue_size"],  # type: ignore
        "peak_time": peak_time,
        "peak_idx": peak_idx,
        "growth_factor": growth_factor,
        "drop_frac": drop_frac,
        "final_drop_frac": final_drop_frac,
        "wall_position": wall_position,
        "onset_time": onset_time if onset_time is not None else np.nan,
        "post_peak_slope": slope,
    }


def check_acceptance_criteria(
    results: Optional[Dict[str, np.ndarray]],
    metrics: Dict[str, float],
) -> bool:
    """Check if simulation meets all acceptance criteria."""
    if results is None:
        return False

    growth_ok = metrics["growth_factor"] >= GROWTH_MIN
    drop_ok = metrics["drop_frac"] >= PEAK_DROP_MIN
    final_ok = metrics["final_drop_frac"] >= PEAK_DROP_MIN
    wall_ok = metrics["wall_position"] <= WALL_TOL
    slope_ok = metrics["post_peak_slope"] < SLOPE_THRESHOLD
    onset_ok = not np.isnan(metrics["onset_time"])

    align_ok = (
        onset_ok and abs(metrics["peak_time"] - metrics["onset_time"]) <= ALIGN_TOL
    )

    return (
        growth_ok
        and drop_ok
        and final_ok
        and wall_ok
        and onset_ok
        and align_ok
        and slope_ok
    )


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
    minutes_per_generation: float = 30.0,
    optimization_target: str = "tissue_and_mesoderm",
    tissue_std: Optional[float] = None,
    mesoderm_std: Optional[float] = None,
    *,
    ema_dict: MutableMapping[str, Optional[float]] | None = None,
) -> float:
    """Fitness cost function based on L2 norm of tissue trajectory and mesoderm
    onset timing.

    Args:
      results: Dictionary containing simulation results with keys:
        - "time": time points (in generation units)
        - "tissue_size": L_sim(t)
        - "mesoderm_fraction": M_sim(t)
      experimental_data: Experimental trajectories to match (time in minutes).
      minutes_per_generation: Conversion factor from generation units to
        minutes
      optimization_target: What to optimize over - "tissue", "mesoderm", or
        "tissue_and_mesoderm". If "tissue_and_mesoderm", both tissue and
        mesoderm losses are computed and weighted dynamically.

    Returns:
      L2 error value or calls _invalid_fitness() if results are invalid or
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
        metrics = extract_simulation_metrics(results)
        if not check_acceptance_criteria(results, metrics):
            return _invalid_fitness()

        l2_tissue = None
        l2_onset = None

        if tissue_std is None or mesoderm_std is None:
            tissue_std, mesoderm_std = _calculate_normalization_scales(
                experimental_data
            )

        sim_time_minutes = t_sim * minutes_per_generation
        if not _check_simulation_length(sim_time_minutes, experimental_data):
            return _invalid_fitness()

        if optimization_target in ["tissue", "tissue_and_mesoderm"]:
            L_sim_on_exp = _interpolate_simulation_to_experimental_timepoints(
                t_sim, L_sim, experimental_data.time, minutes_per_generation
            )
            valid_exp_mask = experimental_data.time <= sim_time_minutes[-1]
            exp_tissue_valid = experimental_data.tissue_size[valid_exp_mask]
            l2_tissue = _calculate_trajectory_mismatch(
                sim_interpolated=L_sim_on_exp,
                exp_values=exp_tissue_valid,
                normalization_scale=tissue_std,
            )

        if optimization_target in ["onset", "tissue_and_mesoderm"]:
            ref_onset_time = calculate_onset_time(
                experimental_data.time, experimental_data.mesoderm_fraction
            )
            l2_onset = calculate_onset_loss(
                metrics.get("onset_time"), ref_onset_time, ONSET_TIME_SCALE
            )

        if optimization_target == "tissue":
            assert l2_tissue is not None
            return l2_tissue

        if optimization_target == "onset":
            assert l2_onset is not None
            return l2_onset

        assert l2_tissue is not None and l2_onset is not None
        losses = {"tissue": l2_tissue, "onset": l2_onset}
        weights = _compute_dynamic_weights(losses=losses, ema_dict=ema_dict)

        slope = metrics.get("post_peak_slope", 0.0)
        slope_penalty = 1e9 if slope >= SLOPE_THRESHOLD else 0.0

        return (
            weights["tissue"] * l2_tissue + weights["onset"] * l2_onset + slope_penalty
        )

    except ValueError as e:
        return _invalid_fitness()


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
