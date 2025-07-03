"""Scan Sobol-sampled parameter space."""

import csv
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d  # type: ignore
from scipy.stats import qmc  # type: ignore
import yaml  # type: ignore

from pescoid_modelling.objective import check_acceptance_criteria
from pescoid_modelling.objective import extract_simulation_metrics
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import apply_parameter_overrides
from pescoid_modelling.utils.config import SimulationParams
from pescoid_modelling.utils.constants import ONSET_TIME_SCALE
from pescoid_modelling.utils.constants import SLOPE_THRESHOLD
from pescoid_modelling.utils.helpers import calculate_onset_time
from pescoid_modelling.utils.helpers import load_yaml_config

PARAM_KEYS: List[str] = [
    "diffusivity",
    "flow",
    "tau_m",
    "gamma",
    "activity",
    "beta",
    "r",
    "m_sensitivity",
    "morphogen_feedback",
]
LOG_SCALE = {"diffusivity", "gamma", "activity", "beta", "r"}
FAST_PATCH = dict(
    total_hours=12.0,
    dx_interval=0.01,
    delta_t=0.01,
    domain_length=10.0,
)

# Optimization settings
MAX_TRIES = 1000
BATCH_POW2 = 5
PROGRESS_EVERY = 32
KEEP_TOP = 5
EPS = 1e-12

# Time conversion
TOTAL_MIN = FAST_PATCH["total_hours"] * 60.0

data_ctrl_t = np.arange(30, 630 + 1, 30)
data_ctrl_r = np.array(
    [
        0.99345961,
        1.08662893,
        1.17979825,
        1.27296758,
        1.34934004,
        1.41669926,
        1.46607894,
        1.4927817,
        1.49463536,
        1.48436569,
        1.46173626,
        1.42968886,
        1.39095644,
        1.34058284,
        1.28913355,
        1.23059502,
        1.16750006,
        1.11313755,
        1.05040067,
        0.98766379,
        0.9249269,
    ]
)
data_ctrl_m = np.array(
    [
        0.00701762,
        0.00569433,
        0.00437104,
        0.00304775,
        0.00254969,
        0.00120534,
        0.0035897,
        0.02141178,
        0.06153418,
        0.11990535,
        0.20262491,
        0.29699181,
        0.3787716,
        0.46077626,
        0.50012072,
        0.51019693,
        0.51019693,
        0.51019693,
        0.51019693,
        0.51019693,
        0.51019693,
    ]
)


def initialize_csv_log(path: Path) -> None:
    """Initialize CSV file with headers for parameter logging."""
    headers = (
        [
            "trial",
            "score",
            "passes",
            "tissue_loss",
            "onset_loss",
            "peak_time",
            "drop_frac",
            "final_drop_frac",
            "onset_time",
            "growth_factor",
            "wall_position",
        ]
        + PARAM_KEYS
        + [
            "delta_t",
            "total_hours",
            "domain_length",
            "dx_interval",
            "m_diffusivity",
            "sigma_c",
            "rho_sensitivity",
            "feedback_mode",
        ]
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def log_trial_to_csv(
    path: Path,
    trial_num: int,
    params: SimulationParams,
    score: float,
    passes: bool,
    metrics: Dict[str, float],
) -> None:
    """Append trial results to CSV file."""
    row = [trial_num, score, int(passes)]
    row.extend(
        [
            metrics.get("tissue_loss", np.nan),
            metrics.get("onset_loss", np.nan),
            metrics.get("peak_time", np.nan),
            metrics.get("drop_frac", np.nan),
            metrics.get("final_drop_frac", np.nan),
            metrics.get("onset_time", np.nan),
            metrics.get("growth_factor", np.nan),
            metrics.get("wall_position", np.nan),
        ]
    )

    for k in PARAM_KEYS:
        row.append(getattr(params, k))

    row.extend(
        [
            params.delta_t,
            params.total_hours,
            params.domain_length,
            params.dx_interval,
            params.m_diffusivity,
            params.sigma_c,
            params.rho_sensitivity,
            params.feedback_mode,
        ]
    )

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def vector_to_simulation_params(
    vec: np.ndarray, base: SimulationParams
) -> SimulationParams:
    """Convert parameter vector to SimulationParams object."""
    return replace(base, **{k: float(v) for k, v in zip(PARAM_KEYS, vec)})


def run_fast_simulation(
    params: SimulationParams,
) -> Tuple[Optional[Dict[str, np.ndarray]], SimulationParams]:
    """Run simulation with fast settings for parameter scanning."""
    try:
        fast_params = apply_parameter_overrides(params, FAST_PATCH)
        with tempfile.TemporaryDirectory() as tmp:
            sim = PescoidSimulator(fast_params, tmp, log_residuals=False)
            sim.run()
            return (None if sim.aborted else sim.results, fast_params)
    except Exception:
        return (None, apply_parameter_overrides(params, FAST_PATCH))


def calculate_tissue_loss(
    sim_results: Dict[str, np.ndarray],
    ref_time: np.ndarray,
    ref_tissue: np.ndarray,
) -> float:
    """Calculate L2 loss for tissue size trajectory."""
    sim_time = sim_results["time"] * 30.0
    sim_tissue = sim_results["tissue_size"]

    valid_ref_mask = ref_time <= sim_time[-1]
    if not np.any(valid_ref_mask):
        return float("inf")

    ref_time_truncated = ref_time[valid_ref_mask]
    ref_tissue_truncated = ref_tissue[valid_ref_mask]

    sim_interp = np.interp(ref_time_truncated, sim_time, sim_tissue)
    tissue_std = np.std(ref_tissue_truncated)
    if tissue_std > 0:
        sim_norm = sim_interp / tissue_std
        ref_norm = ref_tissue_truncated / tissue_std
        return float(np.sum((sim_norm - ref_norm) ** 2))
    else:
        return float(np.sum((sim_interp - ref_tissue_truncated) ** 2))


def calculate_onset_loss(
    sim_onset_time: Optional[float],
    ref_onset_time: Optional[float],
    time_scale: float = ONSET_TIME_SCALE,
) -> float:
    """Calculate loss based on onset time difference."""
    if ref_onset_time is None:
        ref_onset_time = calculate_onset_time(data_ctrl_t, data_ctrl_m)

    if ref_onset_time is None and sim_onset_time is None:
        raise ValueError("Both reference and simulated onset times are None.")

    elif ref_onset_time is None or sim_onset_time is None:
        return 1e9

    time_diff = (sim_onset_time - ref_onset_time) / time_scale
    return time_diff**2


def calculate_combined_score(
    results: Optional[Dict[str, np.ndarray]],
    metrics: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Calculate combined score with tissue L2 and onset timing."""
    if results is None:
        return float("inf"), {}

    tissue_loss = calculate_tissue_loss(results, data_ctrl_t, data_ctrl_r)
    onset_loss = calculate_onset_loss(
        metrics.get("onset_time"), calculate_onset_time(data_ctrl_t, data_ctrl_m)
    )

    slope = metrics.get("post_peak_slope", 0.0)
    slope_penalty = 0.0 if slope < SLOPE_THRESHOLD else 1e9

    tissue_weight = 1.0 / (1.0 + tissue_loss)
    onset_weight = 1.0 / (1.0 + onset_loss)
    total_weight = tissue_weight + onset_weight
    tissue_weight /= total_weight
    onset_weight /= total_weight

    score = tissue_weight * tissue_loss + onset_weight * onset_loss + slope_penalty

    loss_components = {
        "tissue_loss": tissue_loss,
        "onset_loss": onset_loss,
        "tissue_weight": tissue_weight,
        "onset_weight": onset_weight,
    }

    return score, loss_components


def grid_search(
    config_path: Path,
    log_csv: Path = Path("parameter_scan_log.csv"),
    max_tries: int = MAX_TRIES,
) -> None:
    """Local Sobol scan around an existing seed (x naught)."""

    best_pass: Optional[
        Tuple[float, np.ndarray, SimulationParams, Dict[str, float]]
    ] = None

    cfg = load_yaml_config(config_path)
    sim_block = cfg.get("simulation", cfg)
    sim_block = {k: v for k, v in sim_block.items() if k in SimulationParams().__dict__}
    base_params_dict = SimulationParams(**sim_block).__dict__

    LOCAL_FACTORS = {
        "diffusivity": (0.5, 2.0),
        "gamma": (0.5, 2.0),
        "activity": (0.5, 2.0),
        "beta": (0.5, 2.0),
        "r": (0.5, 2.0),
        "flow": (0.7, 1.3),
        "tau_m": (0.7, 1.3),
        "m_sensitivity": (0.7, 1.3),
        "morphogen_feedback": (0.7, 1.3),
    }
    local_keys = list(LOCAL_FACTORS)

    sampler = qmc.Sobol(d=len(local_keys), scramble=True)

    initialize_csv_log(log_csv)
    tested = 0
    best_candidates: List[Tuple[float, np.ndarray, SimulationParams]] = []

    ref_onset_time = calculate_onset_time(data_ctrl_t, data_ctrl_m)
    print(f"Local Sobol search around seed with max {max_tries} trials …")
    print(f"Reference onset time: {ref_onset_time:.1f} minutes")

    while tested < max_tries:
        batch_size = min(2**BATCH_POW2, max_tries - tested)
        for u in sampler.random(batch_size):
            tested += 1
            p_dict = base_params_dict.copy()
            for ui, k in zip(u, local_keys):
                lo, hi = LOCAL_FACTORS[k]
                p_dict[k] = p_dict[k] * (lo + ui * (hi - lo))

            params = SimulationParams(**p_dict)
            vec = np.array([getattr(params, k) for k in PARAM_KEYS], dtype=float)

            results, params_used = run_fast_simulation(params)
            if results is not None:
                metrics = extract_simulation_metrics(results)
                score, loss_comps = calculate_combined_score(
                    results,
                    metrics,
                )
                metrics.update(loss_comps)
            else:
                metrics = {}
                score = float("inf")

            passes = check_acceptance_criteria(results, metrics)
            log_trial_to_csv(log_csv, tested, params_used, score, passes, metrics)

            if np.isfinite(score):
                best_candidates.append((score, vec, params))
                best_candidates.sort(key=lambda x: x[0])
                best_candidates = best_candidates[:KEEP_TOP]

            if passes and np.isfinite(score):
                if best_pass is None or score < best_pass[0]:  # type: ignore
                    best_pass = (score, vec, params, metrics)

            if tested % PROGRESS_EVERY == 0:
                best_overall = best_candidates[0][0] if best_candidates else np.inf
                best_pass_score = best_pass[0] if best_pass else np.inf  # type: ignore
                print(
                    f"Tested {tested:>4}/{max_tries}  |  "
                    f"best score: {best_overall:.6g}  |  "
                    f"best PASS: {best_pass_score if best_pass else '–'}"  # type: ignore
                )

    print("\nSearch completed.  Results summary:\n" + "-" * 60)
    if best_pass is not None:  # type: ignore
        score, vec, params, metrics = best_pass  # type: ignore
        print(f"Best *passing* candidate (score = {score:.6g}):")
        for k, v in zip(PARAM_KEYS, vec):
            print(f"  {k:>15}: {v:.6g}")
        print(
            "\nTissue loss: {tissue_loss:.3g} | Onset loss: {onset_loss:.3g}".format(
                **metrics
            )
        )
    else:
        print("No parameter vector satisfied all acceptance criteria.\n")

    print(f"All {tested} trials logged in {log_csv.resolve()}")


def main() -> None:
    """Entry point for parameter scanning."""
    config_path = Path("/Users/steveho/PescoidProject/configs/optimization_config.yaml")
    grid_search(config_path)


if __name__ == "__main__":
    main()
