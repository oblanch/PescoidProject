"""Parameter sweep for pescoid phase diagrams (A vs F and β vs R)."""

import argparse
import csv
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List

import numpy as np

from pescoid_modelling.objective import extract_simulation_metrics
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import apply_parameter_overrides
from pescoid_modelling.utils.config import SimulationParams
from pescoid_modelling.utils.helpers import get_physical_cores
from pescoid_modelling.utils.helpers import load_yaml_config

CORES = get_physical_cores()
mp.set_start_method("spawn", force=True)
CTX = mp.get_context("spawn")


def onset_time_from(metrics: Dict[str, float]) -> float:
    """Return mesoderm onset time (minutes) or NaN if unavailable."""
    return float(metrics.get("onset_time", np.nan))


def init_csv(path: Path, header: List[str]):
    """Initialize a CSV file with the given header."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_csv(path: Path, row: List):
    """Append a row to the CSV file at the given path."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def get_final_tissue_size(metrics: Dict[str, float]) -> float:
    """Return final tissue size or NaN if unavailable."""
    if "tissue_size" in metrics:
        size_array = np.asarray(metrics["tissue_size"], dtype=float)
        if size_array.size > 0:
            return float(size_array[-1])
    return np.nan


def range_from_list(lst: List[float]) -> np.ndarray:
    """Convert a list of 2 or 3 numbers into a numpy linspace."""
    if len(lst) == 3:
        return np.linspace(lst[0], lst[1], int(lst[2]))
    elif len(lst) == 2:
        return np.linspace(lst[0], lst[1], 9)
    else:
        raise ValueError("Sweep range must have 2 or 3 numbers (start, stop[, num])")


@dataclass(frozen=True)
class SweepConfig:
    """Container describing a 2-D parameter sweep."""

    tag: str  # "AF", "BR", or "RTm"
    p1_name: str
    p2_name: str
    p1_range: Iterable[float]
    p2_range: Iterable[float]

    def header(self) -> List[str]:
        """Return the header for the CSV file."""
        if self.tag in ["AF", "AB"]:  # AF or AB sweep
            return [
                "pair",
                self.p1_name,
                self.p2_name,
                "state",
                "final_tissue_size",
            ]
        else:  # BR or RTm or AR
            return [
                "pair",
                self.p1_name,
                self.p2_name,
                "onset_time",
                "final_tissue_size",
            ]


def run_simulation(base: SimulationParams, overrides: Dict[str, float]) -> Dict | None:
    """Run one simulation."""
    params = apply_parameter_overrides(base, overrides)
    with tempfile.TemporaryDirectory() as tmp:
        sim = PescoidSimulator(params, tmp, log_residuals=False)
        sim.run()
    return None if sim.aborted else sim.results


def classify_state(
    metrics: Dict[str, float],
    threshold: float = 0.05,
) -> int:
    """Classify trajectory into states using ≥ 5 % threshold crossings.

    0 – Wet + de-wet: grow ≥ 5 % above the start, then shrink ≥ 5 %
        below that peak (single up -> down cycle).
    1 – De-wet only: never grows ≥ 5 % above the start but shrinks ≥ 5 %
        below the start (monotonic down).
    2 – Wet only: grows ≥ 5 % above the start and never shrinks ≥ 5 % below
        its post-growth extrema (monotonic up).
    3 – Any additional ≥ 5 % reversal or < 3 points / ≤ 5 % flat.
    """
    size = np.asarray(metrics["tissue_size"], dtype=float)
    if size.size < 3:
        return 3

    start = size[0]
    extreme = start
    direction = None
    crossings = 0

    for val in size[1:]:
        if direction is None:
            if val > extreme * (1 + threshold):
                direction = "up"
                extreme = val
            elif val < extreme * (1 - threshold):
                direction = "down"
                extreme = val

        elif direction == "up":
            if val > extreme:
                extreme = val
            elif val < extreme * (1 - threshold):
                crossings += 1
                direction = "down"
                extreme = val

        elif direction == "down":
            if val < extreme:
                extreme = val
            elif val > extreme * (1 + threshold):
                crossings += 1
                direction = "up"
                extreme = val

    if crossings >= 2:
        return 3

    if direction == "up" and crossings == 0:
        return 2
    if direction == "down" and crossings == 0:
        return 1
    if crossings == 1:
        return 0

    return 3


def _worker(args: tuple) -> tuple[float, float, Dict | None]:
    """Run one (p1, p2) simulation and return results for the pool."""
    p1, p2, p1_name, p2_name, base = args
    res = run_simulation(base, {p1_name: p1, p2_name: p2})
    return p1, p2, res


def sweep(config: SweepConfig, base: SimulationParams, out_csv: Path) -> None:
    """Run a 2-D grid sweep (parallel with multiprocessing)."""
    init_csv(out_csv, config.header())

    param_grid = [
        (p1, p2, config.p1_name, config.p2_name, base)
        for p1 in config.p1_range
        for p2 in config.p2_range
    ]
    total = len(param_grid)

    print(f"[{config.tag}] using {CORES} workers for {total} jobs")
    with CTX.Pool(processes=CORES, maxtasksperchild=1) as pool:
        for idx, (p1, p2, results) in enumerate(
            pool.imap_unordered(_worker, param_grid), 1
        ):
            metrics = extract_simulation_metrics(results) if results else {}
            final_size = get_final_tissue_size(metrics)

            if config.tag in ["AF", "AB"]:  # AF or AB sweep
                state = classify_state(metrics) if metrics else 3
                append_csv(out_csv, ["AF", p1, p2, state, final_size])
            else:  # BR or RTm or AR sweep
                onset = onset_time_from(metrics)
                append_csv(out_csv, [config.tag, p1, p2, onset, final_size])

            if idx % 32 == 0 or idx == total:
                print(f"[{config.tag}] {idx}/{total} done …")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Pescoid parameter sweep.")
    parser.add_argument(
        "--sweep",
        choices=("AF", "BR", "RTm", "AR", "AB"),
        help="Which sweep to run",
        default="AF",
    )
    parser.add_argument(
        "--yaml",
        default="configs/current_best.yaml",
        type=Path,
        help="YAML file with base parameters",
    )
    parser.add_argument(
        "--out",
        default="phase_diagram_data.csv",
        type=Path,
        help="CSV file to append results into",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for phase diagram parameter sweep."""
    args = _parse_arguments()
    if args.out == Path("phase_diagram_data.csv"):
        args.out = Path(f"phase_diagram_data.{args.sweep.lower()}.csv")

    config_yaml = load_yaml_config(args.yaml)
    base = config_yaml.get("simulation", config_yaml)
    sweep_config = config_yaml.get("sweep", {})

    base_params = SimulationParams(
        **{k: v for k, v in base.items() if k in SimulationParams().__dict__}
    )
    activity_range = range_from_list(sweep_config.get("activity", [0.05, 2.0, 25]))
    flow_range = range_from_list(sweep_config.get("flow", [0.05, 0.35, 25]))
    beta_range = range_from_list(sweep_config.get("beta", [0.0, 5.0, 25]))
    r_range = range_from_list(sweep_config.get("r", [0.5, 3.0, 25]))
    tau_m_range = range_from_list(sweep_config.get("tau_m", [0.1, 8.0, 25]))

    sweep_configs = {
        "AF": ("activity", "flow", activity_range, flow_range),
        "BR": ("beta", "r", beta_range, r_range),
        "RTm": ("r", "tau_m", r_range, tau_m_range),
        "AR": ("activity", "r", activity_range, r_range),
        "AB": ("activity", "beta", activity_range, beta_range),
    }

    p1_name, p2_name, p1_range, p2_range = sweep_configs[args.sweep]
    sweep_config = SweepConfig(
        tag=args.sweep,
        p1_name=p1_name,
        p2_name=p2_name,
        p1_range=p1_range,
        p2_range=p2_range,
    )

    sweep(
        config=sweep_config,
        base=base_params,
        out_csv=args.out,
    )


if __name__ == "__main__":
    main()
