"""Structures + loader for the non-dimensional PESC model. Dataclasses for
solver parameters and optimization settings."""

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml  # type: ignore

_ORDER = [
    "diffusivity",
    "flow",
    "tau_m",
    "gamma",
    "activity",
    "beta",
    "r",
    "m_sensitivity",
    "growth_inhibition",
]


@dataclass(frozen=True)
class SimulationParams:
    """Class representing parameters for the pescoid model.

    Attributes:
      delta_t:
        Time step (t_g units) & must be < the tau_m lower bound (the fastest
        time-scale)
      total_hours:
        Total biological time to simulate [h].
      domain_length:
        Physical domain length (x ∈ [-L/2, L/2]) in nondim units
      dx_interval:
        IntervalMesh spacing.
      diffusivity:
        δ - nondim diffusivity
      m_diffusivity:
        Artificial diffusivity on mesoderm for numerical stability.
      tau_m:
        Mesoderm growth time-scale.
      flow:
        F - nondimensional advection or flow.
      activity:
        A - activity parameter.
      beta:
        β - contribution of mesoderm fate to cell's contractility
      gamma:
        Γ non friction coefficient.
      sigma_c:
        σ_c -critical amount of mechanical feedback)
      r:
        Sensitivity of cells to mechanical feedback
      rho_sensitivity:
        Saturation of active stress at high density
      m_sensitivity:
        Sensitivity of the increase in contractility when cells become mesoderm
      c_diffusivity:
        D_c - morphogen diffusion coefficient
      morphogen_decay:
        k - morphogen decay/consumption rate
      gaussian_width:
        σ - width of Gaussian source
      morphogen_feedback:
        R - morphogen sensitivity for chemical feedback
      feedback_mode:
        Mode of mechanical feedback, either "active_stress" or
        "strain_rate".
    """

    delta_t: float = 0.01
    total_hours: float = 12.0
    domain_length: float = 10.0
    dx_interval: float = 0.001
    diffusivity: float = 8.980959167540726e-05
    m_diffusivity: float = 2e-3
    tau_m: float = 5.96945686472471
    flow: float = 0.14313330708373756
    activity: float = 0.8177242457000748
    beta: float = 0.6359440258892959
    gamma: float = 0.19664898263383435
    sigma_c: float = 0.1
    r: float = 1.4355095309012016
    rho_sensitivity: float = 0.0
    m_sensitivity: float = 0.09628726199197271
    feedback_mode: str = "active_stress"
    morphogen_feedback: float = 1.5


@dataclass(frozen=True)
class CMAConfig:
    """Class representing CMA-ES optimization parameters."""

    x0: List[float]
    sigma0: float
    popsize: int
    bounds: Tuple[List[float], List[float]]
    max_evals: int
    n_restarts: int


def _load_yaml(yaml_file: Union[Path, str]) -> Dict[str, Any]:
    """Load a YAML file and return the contents as a dictionary."""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


def load_config(
    path: Union[str, Path], require_cma: bool = True
) -> Tuple[SimulationParams, CMAConfig | None]:
    """Return parameters from YAML."""
    params = _load_yaml(path)
    sim = SimulationParams(**params.get("simulation", {}))

    cma_raw = params.get("cma")
    if cma_raw is None:
        if require_cma:
            raise ValueError("Missing 'cma' section in config.")
        return sim, None

    x0_vec = [float(cma_raw["x0"][k]) for k in _ORDER]
    lower_vec = [float(cma_raw["bounds"]["lower"][k]) for k in _ORDER]
    upper_vec = [float(cma_raw["bounds"]["upper"][k]) for k in _ORDER]

    cma = CMAConfig(
        x0=x0_vec,
        sigma0=float(cma_raw["sigma0"]),
        popsize=cma_raw.get("popsize"),
        bounds=(lower_vec, upper_vec),
        max_evals=int(cma_raw["max_evals"]),
        n_restarts=int(cma_raw.get("n_restarts", 0)),
    )

    return sim, cma


def load_config_with_overrides(
    path: Union[str, Path], overrides: Dict[str, Any], require_cma: bool = True
) -> Tuple[SimulationParams, CMAConfig | None]:
    """Load configuration from YAML and apply command-line overrides.

    Args:
        path: Path to YAML configuration file
        overrides: Dictionary of parameter overrides from command line
        require_cma: Whether CMA configuration is required

    Returns:
        Tuple of (SimulationParams with overrides applied, CMAConfig or None)
    """
    sim_params, cma_config = load_config(path, require_cma)
    sim_params_with_overrides = apply_parameter_overrides(sim_params, overrides)
    return sim_params_with_overrides, cma_config


def apply_parameter_overrides(
    params: SimulationParams, overrides: Dict[str, Any]
) -> SimulationParams:
    """Apply command-line parameter overrides to SimulationParams."""
    valid_fields = {field.name for field in params.__dataclass_fields__.values()}

    invalid_keys = set(overrides.keys()) - valid_fields
    if invalid_keys:
        raise ValueError(
            f"Invalid parameter override(s): {invalid_keys}. "
            f"Valid parameters are: {sorted(valid_fields)}"
        )

    filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
    if not filtered_overrides:
        return params

    return replace(params, **filtered_overrides)


def extract_simulation_overrides(args) -> Dict[str, Any]:
    """Extract simulation parameter overrides from command-line arguments."""
    param_names = {
        field.name for field in SimulationParams.__dataclass_fields__.values()
    }

    overrides = {}
    for param_name in param_names:
        if hasattr(args, param_name):
            value = getattr(args, param_name)
            if value is not None:
                overrides[param_name] = value

    return overrides
