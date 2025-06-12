"""Structures + loader for the non-dimensional PESC model. Dataclasses for
solver parameters and optimization settings."""

from dataclasses import dataclass
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
    m_diffusivity: float = 1e-3
    tau_m: float = 5.96945686472471
    flow: float = 0.14313330708373756
    activity: float = 0.8177242457000748
    beta: float = 0.6359440258892959
    gamma: float = 0.19664898263383435
    sigma_c: float = 0.0
    r: float = 1.4355095309012016
    rho_sensitivity: float = 0.0
    m_sensitivity: float = 0.09628726199197271
    feedback_mode: str = "active_stress"
    c_diffusivity: float = 5e-4
    morphogen_decay: float = 0.05
    gaussian_width: float = 0.2
    morphogen_feedback: float = 0.0


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
