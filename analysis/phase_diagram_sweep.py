"""Parameter sweep for pescoid phase diagrams (A vs F and β vs R)."""

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

    tag: str  # "AF" or "BR"
    p1_name: str
    p2_name: str
    p1_range: Iterable[float]
    p2_range: Iterable[float]

    def header(self) -> List[str]:
        """Return the header for the CSV file."""
        return [
            "pair",
            self.p1_name,
            self.p2_name,
            "state",
            "final_tissue_size",
            "mesoderm_fraction",
        ]


def run_simulation(base: SimulationParams, overrides: Dict[str, float]) -> Dict | None:
    """Run one simulation."""
    params = apply_parameter_overrides(base, overrides)
    with tempfile.TemporaryDirectory() as tmp:
        sim = PescoidSimulator(params, tmp, log_residuals=False)
        sim.run()
    return None if sim.aborted else sim.results


def classify_state(metrics: Dict[str, float]) -> int:
    """Classify trajectory into states using sign changes in Δsize/Δt.

    0 – Grows and later shrinks - wetting and dewetting
    1 – Derivatives ≤ 0 (never grows) --> only de‑wet
    2 – Derivatives ≥ 0 (never shrinks) --> only wet
    3 – Everything else (e.g. oscillatory/noisy)
    """
    size = np.asarray(metrics["tissue_size"], dtype=float)
    if size.size < 3:
        return 3

    diff = np.diff(size)
    pos = diff > 0
    neg = diff < 0

    if pos.any() and neg.any():
        first_neg_idx = np.argmax(neg)
        if pos[:first_neg_idx].any():
            return 0
        return 3
    if pos.any() and not neg.any():
        return 2
    if neg.any() and not pos.any():
        return 1
    return 3


def onset_time_from(metrics: Dict[str, float]) -> float:
    """Return mesoderm onset time (minutes) or NaN if unavailable."""
    return float(metrics.get("onset_time", np.nan))


def init_csv(path: Path, header: List[str]):
    """Initialize a CSV file with the given header."""
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_csv(path: Path, row: List):
    """Append a row to the CSV file at the given path."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


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
    with mp.Pool(processes=CORES) as pool:
        for idx, (p1, p2, results) in enumerate(
            pool.imap_unordered(_worker, param_grid), 1
        ):
            metrics = extract_simulation_metrics(results) if results else {}

            if config.tag == "AF":
                state = classify_state(metrics) if metrics else 3
                append_csv(out_csv, ["AF", p1, p2, state])
            else:
                onset = onset_time_from(metrics)
                append_csv(out_csv, ["BR", p1, p2, onset])

            if idx % 32 == 0 or idx == total:
                print(f"[{config.tag}] {idx}/{total} done …")


def main(
    yaml_path: Path = Path("configs/current_best.yaml"),
    out_csv: Path = Path("phase_diagram_data.csv"),
) -> None:
    """Entry point for phase diagram parameter sweep."""
    config_yaml = load_yaml_config(yaml_path)
    base = config_yaml.get("simulation", config_yaml)
    sweep_config = config_yaml.get("sweep", {})

    base_params = SimulationParams(
        **{k: v for k, v in base.items() if k in SimulationParams().__dict__}
    )

    activity_range = range_from_list(sweep_config.get("activity", [0, 5.0, 15]))
    flow_range = range_from_list(sweep_config.get("flow", [0, 10.0, 15]))
    beta_range = range_from_list(sweep_config.get("beta", [0, 5.0, 15]))
    r_range = range_from_list(sweep_config.get("r", [0.0, 20.0, 15]))

    af_config = SweepConfig("AF", "activity", "flow", activity_range, flow_range)
    br_config = SweepConfig("BR", "beta", "r", beta_range, r_range)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_af = out_csv.with_suffix(".af.csv")
    tmp_br = out_csv.with_suffix(".br.csv")

    print("Running (activity, flow) sweep …")
    sweep(af_config, base_params, tmp_af)

    print("Running (beta, r) sweep …")
    sweep(br_config, base_params, tmp_br)

    with open(out_csv, "w") as fout:
        fout.write(Path(tmp_af).read_text())
        fout.write(Path(tmp_br).read_text())

    tmp_af.unlink()
    tmp_br.unlink()
    print(f"Phase‑diagram data written to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
