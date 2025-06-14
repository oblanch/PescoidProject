"""Command-line interface for Pescoid simulation."""

import argparse
from typing import Callable


def build_parser(
    simulate: Callable, optimize: Callable, plot: Callable
) -> argparse.ArgumentParser:
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

    _build_simulation_parser(sub, simulate)
    _build_optimization_parser(sub, optimize)
    _build_visualization_parser(sub, plot)
    return parser


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
    parser.add_argument(
        "--name",
        type=str,
        help="Optionally override the default output folder name (based on the config file).",
    )


def _build_simulation_parser(
    subparsers: argparse._SubParsersAction, func: Callable
) -> None:
    """Adds `simulate` sub-command."""
    _build_run_parser(
        subparsers,
        name="simulate",
        help_msg="Run a single pescoid simulation.",
        func=func,
        corrected_pressure=True,
    )


def _build_optimization_parser(
    subparsers: argparse._SubParsersAction, func: Callable
) -> None:
    """Adds `optimize` sub-command (same args minus pressure toggle)."""
    _build_run_parser(
        subparsers,
        name="optimize",
        help_msg="Run CMA-ES to optimize PDE parameters.",
        func=func,
    )


def _build_run_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_msg: str,
    func: Callable,
    *,
    corrected_pressure: bool = False,
    optimize_args: bool = False,
) -> None:
    """Helper used to build *simulate* and *optimize* sub-commands."""
    parser = subparsers.add_parser(name, help=help_msg)
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
    if corrected_pressure:
        parser.add_argument(
            "--corrected_pressure",
            action="store_true",
            help="Use corrected pressure velocity for simulation.",
            default=False,
        )

    if optimize_args:
        parser.add_argument(
            "--optimization_target",
            type=str,
            choices=["tissue", "mesoderm", "tissue_and_mesoderm"],
            default="tissue_and_mesoderm",
            help="What to optimize over: tissue, mesoderm, or tissue_and_mesoderm",
        )

    parser.set_defaults(func=func)


def _build_visualization_parser(
    subparsers: argparse._SubParsersAction, func: Callable
) -> None:
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
    plot_parser.set_defaults(func=func)
