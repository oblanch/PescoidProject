"""Command-line interface for Pescoid simulation."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

from pescoid_modelling.objective import ReferenceTrajectories
from pescoid_modelling.optimizer import CMAOptimizer
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.config import extract_simulation_overrides
from pescoid_modelling.utils.config import load_config
from pescoid_modelling.utils.config import load_config_with_overrides
from pescoid_modelling.utils.parsers import build_parser
from pescoid_modelling.visualization.plot_trajectories import (
    visualize_simulation_results,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | " "%(name)s:%(lineno)d - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("pescoid.cli")


def _load_trajectories(path: Path) -> ReferenceTrajectories:
    """Load data from a .npz file and return ExperimentalTrajectories object."""
    data = np.load(path)
    try:
        return ReferenceTrajectories(
            time=data["time"],
            tissue_size=data["tissue_size"],
            mesoderm_fraction=data["mesoderm_fraction"],
        )
    except KeyError as e:
        raise ValueError(
            f"Missing required keys in {path}. Expected keys: "
            "'time', 'tissue_size', 'mesoderm_fraction'."
        ) from e


def _validate_experimental_data(experimental_npz: str) -> None:
    """Check that experimental data file exists and is readable."""
    exp_path = Path(experimental_npz)
    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experimental data file not found: {experimental_npz}\n"
            "Please specify a valid --experimental_npz path."
        )
    try:
        np.load(exp_path)
    except Exception as e:
        raise ValueError(f"Cannot read experimental data file {experimental_npz}: {e}")


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    """Prepare the output directory for results. Creates a subdirectory based on
    the prefix of the yaml.
    """
    output_path = Path(args.output_dir)

    if args.name:
        work_dir = output_path / args.name
    else:
        config_file = Path(args.config)
        work_dir = output_path / config_file.stem

    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _run_simulation(args: argparse.Namespace) -> None:
    """Entry point for running simulations."""
    work_dir = _prepare_output_dir(args)

    overrides = extract_simulation_overrides(args)
    sim_params, _ = load_config_with_overrides(
        args.config, overrides, require_cma=False
    )
    out_npz = work_dir / "simulation_results.npz"

    # Initialize and run simulator
    LOGGER.info(f"Running pescoid simulation --> {work_dir}")
    simulator = PescoidSimulator(
        parameters=sim_params,
        work_dir=work_dir,
        corrected_pressure=args.corrected_pressure,
    )
    simulator.run()
    simulator.save(out_npz)
    LOGGER.info(f"Simulation results saved --> {out_npz}")

    results = simulator.results
    if results.get("aborted", [False])[0]:
        LOGGER.warning("Simulation aborted before completion.")
    else:
        LOGGER.info("Simulation completed successfully.")
        if args.generate_figures:
            LOGGER.info("Visualizing simulation results.")
            visualize_simulation_results(
                data_path=str(out_npz),
                experimental_npz=(
                    args.experimental_npz if args.experimental_npz else None
                ),
                output_dir=str(work_dir),
            )


def _run_optimization(args: argparse.Namespace) -> None:
    """Entry point for running optimization."""
    # Load configuration
    _validate_experimental_data(args.experimental_npz)
    work_dir = _prepare_output_dir(args)
    sim_params, cma_cfg = load_config(args.config)
    if cma_cfg is None:
        raise ValueError(
            f"Errant or missing CMA-ES optimizer configuration in {args.config}."
        )
    experimental_data = _load_trajectories(Path(args.experimental_npz))

    # Initialize and run optimizer
    optimizer = CMAOptimizer(
        work_dir=work_dir,
        base_params=sim_params,
        x0=cma_cfg.x0,
        sigma=cma_cfg.sigma0,
        bounds=cma_cfg.bounds,
        max_evals=cma_cfg.max_evals,
        popsize=cma_cfg.popsize,
        n_restarts=cma_cfg.n_restarts,
        experimental_data=experimental_data,
        optimization_target=args.optimization_target,
    )
    LOGGER.info("Running CMA-ES optimization.")
    best_params = optimizer.optimize()

    out_txt = work_dir / "best_parameters.txt"
    with open(out_txt, "w") as output:
        output.write("# Optimized pescoid parameters\n")
        output.write("# ----------------------------\n")
        for name in _ORDER:
            output.write(f"{name}: {getattr(best_params, name)}\n")
    LOGGER.info(f"Best parameters written --> {out_txt}")


def _post_simulation_visualization(args: argparse.Namespace) -> None:
    """Plot saved trajectories without running simulation."""
    visualize_simulation_results(
        data_path=args.simulation_npz,
        experimental_npz=args.experimental_npz if args.experimental_npz else None,
        output_dir=args.output_dir,
    )
    LOGGER.info("Generated post-simulation trajectories.")


def main() -> None:
    """Program entry point."""
    parser = build_parser(
        simulate=_run_simulation,
        optimize=_run_optimization,
        plot=_post_simulation_visualization,
    )
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        LOGGER.exception(f"Unhandled exception: {exc}")
        LOGGER.error("Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
