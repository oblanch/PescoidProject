"""Optimization objective for the pescoid model."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, cast, Dict, Optional, Tuple

import numpy as np


class SimulationState(IntEnum):
    """Classification of simulation outcomes."""

    LATE_TRANSITION = 0  # Late or minimal transition
    NORMAL_TRANSITION = 1  # Normal transition
    EARLY_TRANSITION = 2  # Early or minimal transition
    INVALID = 3  # Invalid or no transition


@dataclass
class ExperimentalData:
    """Reference experimental data for optimization comparison."""

    # Boundary/transition metrics
    transition_time: Optional[float] = None
    transition_position: Optional[float] = None
    final_position: Optional[float] = None

    # Mesoderm metrics
    max_mesoderm_time: Optional[float] = None
    mesoderm_lag: Optional[float] = None

    # Expected state (0-3)
    expected_state: Optional[int] = None


@dataclass
class SimulationMetrics:
    """Container for metrics extracted from simulation results."""

    # Basic metrics
    initial_position: float
    final_position: float
    time_bound: float

    # Transition metrics
    transition_time: float
    transition_position: float

    # Mesoderm metrics
    mesoderm_lag: float
    max_mesoderm_time: float
    max_mesoderm_position: float

    # State classification
    state: SimulationState

    @classmethod
    def from_results(cls, results: Dict[str, np.ndarray]) -> "SimulationMetrics":
        """Create SimulationMetrics from raw simulation results.

        Args:
          results: Dictionary containing simulation results

        Returns:
          SimulationMetrics object with calculated metrics
        """
        boundary_positions = results.get("boundary_positions", np.array([0.0]))
        boundary_time_data = results.get("boundary_time_data", np.array([0.0]))
        meso_frac_data = results.get("meso_frac_data", np.array([0.0]))

        if len(boundary_positions) == 0 or len(boundary_time_data) == 0:
            return cls._populate_invalid_metrics()

        initial_position = boundary_positions[0]
        final_position = boundary_positions[-1]
        time_bound = boundary_time_data[-1]

        transition_time, transition_position = find_transition(
            boundary_time_data, boundary_positions
        )

        mesoderm_metrics = calculate_mesoderm_metrics(
            boundary_time_data, transition_time, meso_frac_data
        )

        state = determine_state(
            transition_time,
            transition_position,
            final_position,
            initial_position,
            time_bound,
        )

        return cls(
            initial_position=initial_position,
            final_position=final_position,
            time_bound=time_bound,
            transition_time=transition_time,
            transition_position=transition_position,
            mesoderm_lag=mesoderm_metrics[0],
            max_mesoderm_time=mesoderm_metrics[1],
            max_mesoderm_position=mesoderm_metrics[2],
            state=state,
        )

    @classmethod
    def _populate_invalid_metrics(cls) -> "SimulationMetrics":
        """Create a SimulationMetrics object for invalid results."""
        return cls(
            initial_position=0.0,
            final_position=0.0,
            time_bound=0.0,
            transition_time=np.nan,
            transition_position=np.nan,
            mesoderm_lag=np.nan,
            max_mesoderm_time=np.nan,
            max_mesoderm_position=np.nan,
            state=SimulationState.INVALID,
        )


def optimization_objective(
    results: Dict[str, np.ndarray], experimental_data: Optional[ExperimentalData] = None
) -> float:
    """Objective function that evaluates simulation results using combined error
    metrics.

    Args:
      results: Dictionary containing simulation results
      experimental_data: Reference experimental data for comparison

    Returns:
      Combined error value (float)
    """
    # Check for empty or failed simulations
    if not results:
        return 1e9  # Large penalty

    if experimental_data is None:
        experimental_data = get_default_experimental_data()

    metrics = SimulationMetrics.from_results(results)
    return calculate_error_score(metrics, experimental_data)


def get_default_experimental_data() -> ExperimentalData:
    """Provide default experimental data for testing.

    Returns:
      ExperimentalData with default values
    """
    return ExperimentalData(
        transition_time=204.0,
        transition_position=4.0,
        final_position=2.0,
        max_mesoderm_time=204.0,
        mesoderm_lag=0.0,
        expected_state=1,
    )


def calculate_error_score(
    metrics: SimulationMetrics, experimental_data: ExperimentalData
) -> float:
    """Calculate the error score between simulation metrics and experimental
    data based on normalized squared differences.
    """
    comparisons = [
        (metrics.final_position, experimental_data.final_position),
        (metrics.transition_time, experimental_data.transition_time),
        (metrics.transition_position, experimental_data.transition_position),
        (metrics.mesoderm_lag, experimental_data.mesoderm_lag),
        (metrics.max_mesoderm_position, experimental_data.max_mesoderm_time),
    ]

    total_error = 0.0
    error_count = 0

    for sim_value, exp_value in comparisons:
        if _valid_comparison(sim_value, exp_value):
            error = _calculate_normalized_error(sim_value, cast(float, exp_value))
            total_error += error
            error_count += 1

    if experimental_data.expected_state is not None:
        state_error = _calculate_state_error(
            metrics.state, experimental_data.expected_state
        )
        total_error += state_error
        error_count += 1

    if error_count > 0:
        return float(total_error / error_count)
    else:
        return 1e9


def _valid_comparison(sim_value: Any, exp_value: Any) -> bool:
    """Ensure that the values are valid for comparison, not NaN, and not
    zero.
    """
    return (
        exp_value is not None
        and not (isinstance(sim_value, float) and np.isnan(sim_value))
        and exp_value != 0
    )


def _calculate_normalized_error(sim_value: float, exp_value: float) -> float:
    """Calculate normalized squared error between two values."""
    return ((sim_value - exp_value) / exp_value) ** 2


def _calculate_state_error(sim_state: SimulationState, exp_state: int) -> float:
    """Calculate error for state classification normalizing by the number of
    states.
    """
    normalized_error = abs(int(sim_state) - exp_state) / 3.0
    return normalized_error**2


def find_transition(
    time_data: np.ndarray, boundary_positions: np.ndarray
) -> Tuple[float, float]:
    """Find the transition time and position when boundary reaches maximum."""
    if len(boundary_positions) == 0 or len(time_data) == 0:
        return np.nan, np.nan

    max_position = np.max(boundary_positions)
    max_index = np.argmax(boundary_positions)

    if max_index >= len(time_data):
        max_index = type(max_index)(len(time_data) - 1)

    transition_time = time_data[max_index]
    return transition_time, max_position


def calculate_mesoderm_metrics(
    time_data: np.ndarray, transition_time: float, meso_frac_data: np.ndarray
) -> Tuple[float, float, float]:
    """Calculate key metrics related to mesoderm formation.

    Args:
      time_data: Array of time points
      transition_time: Time of maximum boundary position
      meso_frac_data: Array of mesoderm fraction values

    Returns:
      (mesoderm_lag, max_mesoderm_time, max_mesoderm_position)
    """
    if len(meso_frac_data) == 0 or len(time_data) == 0 or np.isnan(transition_time):
        return np.nan, np.nan, np.nan

    max_position = np.max(meso_frac_data)
    max_index = np.argmax(meso_frac_data)

    if max_index >= len(time_data):
        max_index = type(max_index)(len(time_data) - 1)

    max_time = time_data[max_index]

    # Calculate lag (time difference between max mesoderm and transition)
    if max_position <= 0:
        lag = np.nan
    else:
        lag = max_time - transition_time

    return lag, max_time, max_position


def determine_state(
    transition_time: float,
    transition_position: float,
    final_position: float,
    initial_position: float,
    time_bound: float,
) -> SimulationState:
    """Determine the state classification based on transition dynamics.

    Args:
      transition_time: Time of maximum boundary position
      transition_position: Maximum boundary position
      final_position: Final boundary position
      initial_position: Initial boundary position
      time_bound: Maximum simulation time

    Returns:
      SimulationState classification
    """
    if np.isnan(transition_time) or np.isnan(transition_position):
        return SimulationState.INVALID
    elif transition_time >= time_bound or transition_position <= 1.1 * final_position:
        return SimulationState.LATE_TRANSITION
    elif transition_time < 60 or transition_position <= 1.1 * initial_position:
        return SimulationState.EARLY_TRANSITION
    else:
        return SimulationState.NORMAL_TRANSITION
