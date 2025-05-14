"""Structures + loader for the non-dimensional PESC model. Dataclasses for
solver parameters and optimization settings."""

from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml  # type: ignore

from pescoid_modelling.utils.constants import _ORDER

_ORDER = [
    "length_scale",
    "diffusivity",
    "flow",
    "tau_m",
    "gamma",
    "activity",
    "beta",
    "sigma_c",
    "r",
]


@dataclass(frozen=True)
class SimulationParams:
    """Class representing parameters for the pescoid model."""

    delta_t: float
    total_hours: float
    domain_length: float
    dx_interval: float
    diffusivity: float
    tau_m: float
    length_scale: float
    flow: float
    activity: float
    beta: float
    gamma: float
    sigma_c: float
    r: float
    feedback_mode: str = "strain_rate"


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


def load_config(path: Union[str, Path]) -> Tuple[SimulationParams, CMAConfig]:
    """Return parameters from YAML."""
    data = _load_yaml(path)
    sim = SimulationParams(**data["simulation"])

    cma_raw = data.get("cma")
    if cma_raw is None:
        raise ValueError("Missing 'cma' configuration in YAML file")

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
