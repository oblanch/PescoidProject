"""Scan Sobol-sampled parameter space for a stable simulation seed."""

import csv
from dataclasses import replace
import pathlib
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.interpolate import interp1d  # type: ignore
from scipy.stats import qmc  # type: ignore
import yaml  # type: ignore

from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import apply_parameter_overrides
from pescoid_modelling.utils.config import SimulationParams

PARAM_KEYS: List[str] = [
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
LOG_SCALE = {"diffusivity", "gamma", "activity", "beta", "r"}

FAST_PATCH = dict(total_hours=12.0, dx_interval=0.01, delta_t=0.01, domain_length=10.0)
ALIGN_TOL = 30.0
ONSET_THRESH = 0.05
PEAK_DROP_MIN = 0.05
MAX_TRIES = 200
BATCH_POW2 = 5
PROGRESS_EVERY = 32
KEEP_TOP = 5
EPS = 1e-12
MIN_PEAK_TIME = 180.0
GROWTH_MIN = 1.25
WALL_TOL = 0.98
TOTAL_MIN = FAST_PATCH["total_hours"] * 60.0

data_ctrl_t = np.linspace(0, 600, 21)
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


def safe_yaml(path: pathlib.Path) -> Dict[str, Any]:
    """Load configuration YAML."""
    try:
        return yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        sys.exit(f"YAML not found: {path}")


def build_bounds(
    base: SimulationParams, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Make arrays for upper and lower bounds."""
    lower = cfg.get("cma", {}).get("bounds", {}).get("lower", {})
    upper = cfg.get("cma", {}).get("bounds", {}).get("upper", {})
    lb, ub = [], []
    for k in PARAM_KEYS:
        lo = lower.get(k)
        hi = upper.get(k)
        lb.append(float(lo))
        ub.append(float(hi))

    lb, ub = np.array(lb), np.array(ub)  # type: ignore
    for idx, k in enumerate(PARAM_KEYS):
        if k in {"activity", "beta", "gamma", "r"}:
            mid = getattr(base, k)
            lb[idx] = max(lb[idx], 0.5 * mid)
            ub[idx] = min(ub[idx], 1.5 * mid)
    return lb, ub  # type: ignore


def cube_to_vec(u: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Convert unit cube vector to parameter vector."""
    v = lb + u * (ub - lb)
    for i, k in enumerate(PARAM_KEYS):
        if k in LOG_SCALE:
            lo = max(lb[i], EPS)
            hi = max(ub[i], lo * 1.001)
            v[i] = np.exp(np.log(lo) + u[i] * (np.log(hi) - np.log(lo)))
    return v


def vec_to_params(vec: np.ndarray, base: SimulationParams) -> SimulationParams:
    """Convert vector to SimulationParams."""
    return replace(base, **{k: float(v) for k, v in zip(PARAM_KEYS, vec)})  # type: ignore


def preview(vec: np.ndarray, base: SimulationParams) -> Dict[str, Any] | None:
    """Run partial sim for faster crawl through param space."""
    try:
        params = apply_parameter_overrides(vec_to_params(vec, base), FAST_PATCH)
        with tempfile.TemporaryDirectory() as tmp:
            sim = PescoidSimulator(params, tmp, log_residuals=False)
            sim.run()
            return None if sim.aborted else sim.results
    except BaseException:
        return None


def passes(res: Dict[str, Any] | None) -> bool:
    """Strict acceptance gate."""
    if res is None:
        return False
    t = np.linspace(0.0, TOTAL_MIN / 30.0, len(res["tissue_size"]))
    radius = res["tissue_size"]
    mezzo = res["mesoderm_fraction"]
    edge_x = res["boundary_positions"]

    peak_idx = int(radius.argmax())
    drop_ok = (radius[peak_idx] - radius[-1]) >= PEAK_DROP_MIN
    growth_ok = radius[peak_idx] >= GROWTH_MIN * radius[0]

    edge_at_peak = edge_x[peak_idx]
    half_domain = FAST_PATCH["domain_length"] / 2
    wall_ok = edge_at_peak <= WALL_TOL * half_domain

    onset_idx = np.where(mezzo >= ONSET_THRESH)[0]
    onset_ok = onset_idx.size > 0
    align_ok = onset_ok and abs(t[peak_idx] - t[onset_idx[0]]) <= ALIGN_TOL

    return wall_ok and growth_ok and drop_ok and onset_ok and align_ok


def post_peak_slope(r: np.ndarray, t: np.ndarray) -> float:
    """Check slope 40 minutes after peak."""
    peak = r.argmax()
    later = np.searchsorted(t, t[peak] + 40 / 30.0)
    return np.median(np.diff(r[peak : later + 1]) / np.diff(t[peak : later + 1]))  # type: ignore


def score(res: Dict[str, Any] | None, best_so_far: float = np.inf) -> float:
    """L2 distance for tissue size and mesoderm fraction along with a penalty
    for straight trajectories.
    """
    if res is None:
        return float("inf")
    tsim = np.linspace(0.0, TOTAL_MIN, len(res["tissue_size"]))

    slope = post_peak_slope(res["tissue_size"], tsim)
    slope_pen = 0 if slope < -0.002 else 1e9

    r_sim = interp1d(
        tsim,
        res["tissue_size"],
        kind="linear",
        bounds_error=False,
        fill_value=(res["tissue_size"][0], res["tissue_size"][-1]),  # type: ignore
    )(data_ctrl_t)

    m_sim = interp1d(
        tsim,
        res["mesoderm_fraction"],
        kind="linear",
        bounds_error=False,
        fill_value=(res["mesoderm_fraction"][0], res["mesoderm_fraction"][-1]),  # type: ignore
    )(data_ctrl_t)

    err_r = np.sum((r_sim - data_ctrl_r) ** 2)
    err_m = np.sum((m_sim - data_ctrl_m) ** 2)
    w_r = 1 / np.var(data_ctrl_r)
    w_m = 1 / np.var(data_ctrl_m)
    score = w_r * err_r + w_m * err_m + slope_pen
    return score if score < best_so_far else float("inf")


def main() -> None:
    """Entry point: find a good seed for the simulation."""
    out_yaml = pathlib.Path("good_seed.yaml")
    raw_cfg = safe_yaml(
        Path("/Users/steveho/PescoidProject/configs/optimization_config.yaml")
    )
    sim_block = raw_cfg.get("simulation", raw_cfg)
    sim_block = {k: v for k, v in sim_block.items() if k in SimulationParams().__dict__}
    base = SimulationParams(**sim_block)
    lb, ub = build_bounds(base, raw_cfg)

    sampler = qmc.Sobol(d=len(PARAM_KEYS), scramble=True)
    tested = 0
    best: list[tuple[float, np.ndarray]] = []
    metrics = []  # type: ignore

    while tested < MAX_TRIES:
        for row in sampler.random(2**BATCH_POW2):
            tested += 1
            vec = cube_to_vec(row, lb, ub)
            res = preview(vec, base)

            if res is not None:
                t = np.linspace(0, 600, len(res["tissue_size"]))
                radius = res["tissue_size"]
                mezzo = res["mesoderm_fraction"]
                peak_idx = int(radius.argmax())
                peak_time = t[peak_idx]
                drop_frac = (radius[peak_idx] - radius[-1]) / radius[peak_idx]
                onset_idxs = np.where(mezzo >= ONSET_THRESH)[0]
                onset_time = t[onset_idxs[0]] if onset_idxs.size else np.nan
                metrics.append((peak_time, drop_frac, onset_time))

            if passes(res):
                print(f"\nGood seed after {tested} candidates:\n")
                for k, v in zip(PARAM_KEYS, vec):
                    print(f"{k:>15}: {v:.6g}")
                out_yaml.write_text(
                    yaml.safe_dump(
                        {"simulation": {k: float(v) for k, v in zip(PARAM_KEYS, vec)}}
                    )
                )
                print(f"\nSaved to {out_yaml.resolve()}")
                sys.exit(0)

            sc = score(res, best[0][0] if best else np.inf)
            best.append((sc, vec))
            best.sort(key=lambda x: x[0])
            best = best[:KEEP_TOP]

            if tested % PROGRESS_EVERY == 0:
                print(f"…{tested} tested, none strict-pass yet…", flush=True)

            if tested >= MAX_TRIES:
                break

    if best and np.isfinite(best[0][0]):
        vec_best = best[0][1].copy()
        pidx_flow = PARAM_KEYS.index("flow")
        pidx_taum = PARAM_KEYS.index("tau_m")

        for flow_mul, taum_mul in [(0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)]:
            trial = vec_best.copy()
            trial[pidx_flow] *= flow_mul
            trial[pidx_taum] *= taum_mul
            if passes(preview(trial, base)):
                vec_best = trial
                print("Refined seed hit strict gate.")
                out_yaml.write_text(
                    yaml.safe_dump(
                        {
                            "simulation": {
                                k: float(v) for k, v in zip(PARAM_KEYS, vec_best)
                            }
                        }
                    )
                )
                print(f"Saved to {out_yaml.resolve()}")
                sys.exit(0)

    if best and np.isfinite(best[0][0]):
        vec = best[0][1]
        out_yaml.write_text(
            yaml.safe_dump(
                {"simulation": {k: float(v) for k, v in zip(PARAM_KEYS, vec)}}
            )
        )
        print(
            f"\nNo strict seed found; best candidate (score={best[0][0]:.3g}) to {out_yaml}"
        )
    else:
        print("\nAll previews aborted – tighten bounds or adjust PDEs.")

    with open("feasibility_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["peak_time", "drop_frac", "onset_time"])
        writer.writerows(metrics)


if __name__ == "__main__":
    main()
