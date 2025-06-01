"""Command-line interface for Pescoid simulation."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

from pescoid_modelling.objective import ExperimentalTrajectories
from pescoid_modelling.optimizer import CMAOptimizer
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.config import load_config
from pescoid_modelling.visualization.plot_simulation import visualize_simulation_results

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | " "%(name)s:%(lineno)d - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("pescoid.cli")


def _load_trajectories(path: Path) -> ExperimentalTrajectories:
    """Load data from a .npz file and return ExperimentalTrajectories object."""
    data = np.load(path)
    try:
        return ExperimentalTrajectories(
            time=data["time"],
            tissue_size=data["tissue_size"],
            mesoderm_signal=data["mesoderm_signal"],
        )
    except KeyError as e:
        raise ValueError(
            f"Missing required keys in {path}. Expected keys: "
            "'time', 'tissue_size', 'mesoderm_signal'."
        ) from e


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared arguments to the given parser."""
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory in which to write results (will be created).",
    )


def _prepare_output_dir(output_dir: str, config_file_dir: str) -> Path:
    """Prepare the output directory for results. Creates a subdirectory based on
    the prefix of the yaml.
    """
    # Get the prefix from the config file name
    output_path = Path(output_dir)
    config_file = Path(config_file_dir)

    # Create the output directory with the prefix
    work_dir = output_path / config_file.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def _cmd_simulate(args: argparse.Namespace) -> None:
    """Entry point for running simulations."""
    # Load configuration
    sim_params, _ = load_config(args.config)

    # Set up output directory
    work_dir = _prepare_output_dir(
        output_dir=args.output_dir,
        config_file_dir=args.config,
    )

    # Initialize and run simulator
    LOGGER.info(f"Running pescoid simulation --> {work_dir}")
    simulator = PescoidSimulator(parameters=sim_params, work_dir=work_dir)
    simulator.run()

    # Save results
    out_npz = work_dir / "simulation_results.npz"
    simulator.save(out_npz)
    LOGGER.info(f"Simulation results saved --> {out_npz}")

    results = simulator.results
    if results.get("aborted", [False])[0]:
        LOGGER.warning("Simulation aborted before completion.")
    else:
        LOGGER.info("Simulation completed successfully.")
        LOGGER.info("Visualizing simulation results.")
        # visualize_simulation_results(
        #     data_path=str(out_npz),
        #     output_dir=str(work_dir),
        # )


def _cmd_optimize(args: argparse.Namespace) -> None:
    """Entry point for running optimization."""
    # Load configuration
    sim_params, cma_cfg = load_config(args.config)
    experimental_data = _load_trajectories(Path(args.experimental_npz))

    # Set up output directory
    work_dir = _prepare_output_dir(
        output_dir=args.output_dir,
        config_file_dir=args.config,
    )

    # Initialize and run optimizer
    optimizer = CMAOptimizer(
        work_dir=work_dir,
        base_params=sim_params,
        init_guess=cma_cfg.x0,
        sigma=cma_cfg.sigma0,
        bounds=cma_cfg.bounds,
        max_evals=cma_cfg.max_evals,
        popsize=cma_cfg.popsize,
        n_restarts=cma_cfg.n_restarts,
        experimental_data=experimental_data,
    )
    LOGGER.info("Running CMA-ES optimisation.")
    best_params = optimizer.optimize()

    out_txt = work_dir / "best_parameters.txt"
    with open(out_txt, "w") as output:
        output.write("# Optimised pescoid parameters\n")
        output.write("# ----------------------------\n")
        for name in _ORDER:
            output.write(f"{name}: {getattr(best_params, name)}\n")
    LOGGER.info(f"Best parameters written --> {out_txt}")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the root parser with sub-commands."""
    parser = argparse.ArgumentParser(
        prog="pescoid",
        description="CLI for pescoid simulations and optimization.",
    )
    sub = parser.add_subparsers(
        title="sub-commands",
        dest="command",
        metavar="{simulate, optimize}",
        required=True,
    )

    sim_parser = sub.add_parser("simulate", help="Run a single pescoid simulation.")
    _add_common_args(sim_parser)
    sim_parser.set_defaults(func=_cmd_simulate)

    opt_parser = sub.add_parser(
        "optimize", help="Run CMA-ES to optimise pescoid parameters."
    )
    _add_common_args(opt_parser)
    opt_parser.add_argument(
        "--experimental_npz",
        type=str,
        default="data/experimental_timeseries.npz",
        help="Experimental trajectories saved as np.savez(...). "
        "Should include arrays for time scale, normalized area, and mesoderm signal.",
    )
    opt_parser.set_defaults(func=_cmd_optimize)

    return parser


def main() -> None:
    """Program entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        LOGGER.exception(f"Unhandled exception: {exc}")
        LOGGER.error("Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
