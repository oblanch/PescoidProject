"""Utility for scaling of simulation parameters."""

from typing import Sequence

import numpy as np


class ParamScaler:
    """Class for bidirectional transformation of simulation parameters.

    Maps all parameters between [0, 1] using both linear and log transforms to
    improve optimization performance.

    Attributes:
      _lower_bounds: Lower bounds for each parameter in physical units
      _upper_bounds: Upper bounds for each parameter in physical units
      _use_log_scale: Boolean mask indicating log vs linear scaling

    Examples::
      # Mix of parameters with different natural scales
      >>> scaler = ParamScaler(
      ...     lower=[1e-5, 0.0, 0.1],
      ...     upper=[1e-2, 10.0, 8.0],
      ...     log_mask=[True, False, False]
      ... )

      # Convert physical parameters to normalized [0,1] space
      >>> normalized = scaler.to_normalized(physical)

      # Convert optimizer output back to physical space
      >>> physical_result = scaler.to_physical(optimized_normalized)
    """

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        log_mask: Sequence[bool],
    ) -> None:
        """Initialize parameter bounds and scaling types."""
        self._lower_bounds = np.asarray(lower, dtype=float)
        self._upper_bounds = np.asarray(upper, dtype=float)
        self._use_log_scale = np.asarray(log_mask, dtype=bool)

        if np.any(self._upper_bounds <= self._lower_bounds):
            raise ValueError("All upper bounds must exceed corresponding lower bounds")

        if np.any(self._lower_bounds[self._use_log_scale] <= 0):
            raise ValueError(
                "Parameters with log scaling must have positive lower bounds"
            )

    def to_normalized(self, physical_params: Sequence[float]) -> np.ndarray:
        """Transform physical parameters to normalized [0, 1] space."""
        physical = np.asarray(physical_params, dtype=float)
        normalized = np.empty_like(physical)

        log_mask = self._use_log_scale
        log_num = np.log10(physical[log_mask]) - np.log10(self._lower_bounds[log_mask])
        log_den = np.log10(self._upper_bounds[log_mask]) - np.log10(
            self._lower_bounds[log_mask]
        )
        normalized[log_mask] = log_num / log_den

        lin_mask = ~log_mask
        normalized[lin_mask] = (physical[lin_mask] - self._lower_bounds[lin_mask]) / (
            self._upper_bounds[lin_mask] - self._lower_bounds[lin_mask]
        )

        return normalized

    def to_physical(self, normalized_params: Sequence[float]) -> np.ndarray:
        """Transform normalized [0, 1] parameters back to physical space."""
        normalized = np.asarray(normalized_params, dtype=float)
        physical = np.empty_like(normalized)

        log_mask = self._use_log_scale
        physical[log_mask] = 10 ** (
            np.log10(self._lower_bounds[log_mask])
            + normalized[log_mask]
            * (
                np.log10(self._upper_bounds[log_mask])
                - np.log10(self._lower_bounds[log_mask])
            )
        )

        lin_mask = ~log_mask
        physical[lin_mask] = self._lower_bounds[lin_mask] + normalized[lin_mask] * (
            self._upper_bounds[lin_mask] - self._lower_bounds[lin_mask]
        )

        return physical
