"""Structures + loader for the non-dimensional PESC model. Dataclasses for
solver parameters and optimization settings."""

from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml  # type: ignore


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
    popsize: Optional[int]
    bounds: Tuple[List[float], List[float]]
    max_evals: int
    n_restarts: int


def _load_yaml(yaml_file: Union[Path, str]) -> Dict[str, Any]:
    """Load a YAML file and return the contents as a dictionary."""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


def _ordered(mapping: Dict[str, float], cls) -> List[float]:
    """Return a list of values from the mapping ordered by the dataclass
    fields.
    """
    return [mapping[name] for name in (f.name for f in fields(cls)) if name in mapping]


def load_config(path: Union[str, Path]) -> Tuple[SimulationParams, CMAConfig]:
    """Return parameters from YAML."""
    data = _load_yaml(path)
    sim = SimulationParams(**data["simulation"])

    cma_raw = data.get("cma")
    if cma_raw is None:
        raise ValueError("Missing 'cma' configuration in YAML file")

    x0_vec = _ordered(cma_raw["x0"], SimulationParams)
    lower_vec = _ordered(cma_raw["bounds"]["lower"], SimulationParams)
    upper_vec = _ordered(cma_raw["bounds"]["upper"], SimulationParams)

    cma = CMAConfig(
        x0=x0_vec,
        sigma0=float(cma_raw["sigma0"]),
        popsize=cma_raw.get("popsize") or None,
        bounds=(lower_vec, upper_vec),
        max_evals=int(cma_raw["max_evals"]),
        n_restarts=int(cma_raw.get("n_restarts", 0)),
    )

    return sim, cma
