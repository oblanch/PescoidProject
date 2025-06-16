"""Visualization module init."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
            "figure.titlesize": 5,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "xtick.minor.width": 0.25,
            "ytick.minor.width": 0.25,
        }
    )


def _load_simulation_data(file_path: str) -> Dict[str, Any]:
    """Load simulation NPZ."""
    data = np.load(file_path)
    sim = {}

    # 2D fields
    sim["density"] = np.flip(data["density"], axis=1)
    sim["mesoderm"] = np.flip(data["mesoderm"], axis=1)
    sim["velocity"] = np.flip(data["velocity"], axis=1)
    sim["stress"] = np.flip(data["stress"], axis=1)
    sim["morphogen"] = np.flip(data["morphogen"], axis=1)

    # Time and tissue properties
    sim["time"] = data["time"]
    sim["tissue_size"] = data["tissue_size"]
    sim["boundary_positions"] = data["boundary_positions"]
    sim["boundary_times"] = data["boundary_times"]
    sim["boundary_velocity"] = data["boundary_velocity"]

    # Mesoderm
    sim["mesoderm_mean"] = data["mesoderm_mean"]
    sim["mesoderm_center"] = data["mesoderm_center"]
    sim["mesoderm_average"] = data["mesoderm_average"]
    sim["mesoderm_fraction"] = data["mesoderm_fraction"]
    sim["max_mesoderm"] = data["max_mesoderm"]

    # Morphogen
    sim["morphogen_mean"] = data["morphogen_mean"]
    sim["morphogen_center"] = data["morphogen_center"]
    sim["max_morphogen"] = data["max_morphogen"]
    sim["morphogen_edge"] = data["morphogen_edge"]
    sim["morphogen_gradient_max"] = data["morphogen_gradient_max"]
    sim["morphogen_gradient_center"] = data["morphogen_gradient_center"]

    # Coordinates
    sim["x_coords"] = data["x_coords"]

    # Norms
    sim["rho_norm"] = data["rho_norm"]
    sim["m_norm"] = data["m_norm"]
    sim["u_norm"] = data["u_norm"]
    sim["c_norm"] = data["c_norm"]

    _validate_data_consistency(sim)
    return sim


def _validate_data_consistency(data: Dict[str, np.ndarray]) -> None:
    """Validate the consistency of simulation data ensuring array lengths
    match.
    """
    if not (len(data["density"]) == len(data["velocity"]) == len(data["time"])):
        raise ValueError("Mismatched lengths of density, velocity, or time data arrays")

    if not (
        len(data["boundary_positions"])
        == len(data["boundary_times"])
        == len(data["boundary_velocity"])
    ):
        raise ValueError(
            f"Mismatched lengths for boundary data: "
            f"positions ({len(data['boundary_positions'])}), "
            f"times ({len(data['boundary_times'])}), "
            f"velocity ({len(data['boundary_velocity'])})"
        )
