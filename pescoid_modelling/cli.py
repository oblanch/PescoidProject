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
from pescoid_modelling.utils.config import load_config
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


def _run_simulation(args: argparse.Namespace) -> None:
    """Entry point for running simulations."""
    # Load configuration
    if args.generate_figures:
        _validate_experimental_data(args.experimental_npz)

    work_dir = _prepare_output_dir(
        output_dir=args.output_dir,
        config_file_dir=args.config,
    )

    sim_params, _ = load_config(args.config, require_cma=False)
    out_npz = work_dir / "simulation_results.npz"

    # Initialize and run simulator
    LOGGER.info(f"Running pescoid simulation --> {work_dir}")
    simulator = PescoidSimulator(parameters=sim_params, work_dir=work_dir)
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
                experimental_npz=args.experimental_npz,
                output_dir=str(work_dir),
            )


def _run_optimization(args: argparse.Namespace) -> None:
    """Entry point for running optimization."""
    # Load configuration
    _validate_experimental_data(args.experimental_npz)
    work_dir = _prepare_output_dir(
        output_dir=args.output_dir,
        config_file_dir=args.config,
    )

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

    # TODO: Implement visualization for optimization results
    if args.generate_figures:
        LOGGER.info("Figure generation for optimization not yet implemented.")


def _post_simulation_visualization(args: argparse.Namespace) -> None:
    """Plot saved trajectories without running simulation."""
    _validate_experimental_data(args.experimental_npz)

    visualize_simulation_results(
        data_path=args.simulation_npz,
        experimental_npz=args.experimental_npz,
        output_dir=args.output_dir,
    )
    LOGGER.info("Generated post-simulation trajectories.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the root parser with sub-commands."""
    parser = argparse.ArgumentParser(
        prog="pescoid",
        description="CLI for pescoid simulations and optimization.",
    )
    sub = parser.add_subparsers(
        title="sub-commands",
        dest="command",
        metavar="{simulate, optimize, plot}",
        required=True,
    )
    _build_standard_parser(
        sub, "simulate", "Run a single pescoid simulation.", _run_simulation
    )
    _build_standard_parser(
        sub, "optimize", "Run CMA-ES to optimize PDE parameters.", _run_optimization
    )
    _build_visualization_parser(sub)
    return parser


def _build_standard_parser(
    subparsers: argparse._SubParsersAction, name: str, help_text: str, func
) -> None:
    """Add a standard sub-command that only needs common args."""
    parser = subparsers.add_parser(name, help=help_text)
    _add_common_args(parser)
    parser.add_argument(
        "--experimental_npz",
        type=str,
        default="data/reference_timeseries.npz",
        help="Experimental trajectories saved via np.savez(...). "
        "Should include arrays for time scale, normalized area, and mesoderm fraction.",
    )
    parser.add_argument(
        "--generate_figures",
        action="store_true",
        help="Generate visualization plots after run completes.",
    )
    parser.set_defaults(func=func)


def _build_visualization_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the visualization sub-command to the parser."""
    plot_parser = subparsers.add_parser(
        "plot", help="Plot trajectories from existing .npz files."
    )
    plot_parser.add_argument(
        "--simulation_npz",
        required=True,
        type=str,
        help="Path to simulation_results.npz",
    )
    plot_parser.add_argument(
        "--experimental_npz",
        type=str,
        default="data/reference_timeseries.npz",
        help="Experimental trajectories saved via np.savez(...). "
        "Should include arrays for time scale, normalized area, and mesoderm fraction.",
    )
    plot_parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory in which to write results (will be created).",
    )
    plot_parser.set_defaults(func=_post_simulation_visualization)


def main() -> None:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        LOGGER.exception(f"Unhandled exception: {exc}")
        LOGGER.error("Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
