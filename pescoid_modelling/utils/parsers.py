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

    if name == "simulate":
        _add_simulation_parameter_overrides(parser)

    parser.add_argument(
        "--experimental_npz",
        type=str,
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


def _add_simulation_parameter_overrides(parser: argparse.ArgumentParser) -> None:
    """Add optional arguments for all simulation parameters."""
    param_group = parser.add_argument_group(
        "simulation parameter overrides", "Override any parameter from the config file"
    )

    # Time and domain parameters
    param_group.add_argument("--delta_t", type=float, help="Time step (t_g units)")
    param_group.add_argument(
        "--total_hours", type=float, help="Total biological time to simulate [h]"
    )
    param_group.add_argument(
        "--domain_length", type=float, help="Physical domain length in nondim units"
    )
    param_group.add_argument("--dx_interval", type=float, help="IntervalMesh spacing")

    # Physical parameters
    param_group.add_argument("--diffusivity", type=float, help="δ - nondim diffusivity")
    param_group.add_argument(
        "--m_diffusivity", type=float, help="Artificial diffusivity on mesoderm"
    )
    param_group.add_argument("--tau_m", type=float, help="Mesoderm growth time-scale")
    param_group.add_argument(
        "--flow", type=float, help="F - nondimensional advection or flow"
    )
    param_group.add_argument("--activity", type=float, help="A - activity parameter")
    param_group.add_argument(
        "--beta", type=float, help="β - contribution of mesoderm fate to contractility"
    )
    param_group.add_argument("--gamma", type=float, help="Γ non friction coefficient")
    param_group.add_argument(
        "--sigma_c", type=float, help="σ_c - critical amount of mechanical feedback"
    )
    param_group.add_argument(
        "--r", type=float, help="Sensitivity of cells to mechanical feedback"
    )
    param_group.add_argument(
        "--rho_sensitivity",
        type=float,
        help="Saturation of active stress at high density",
    )
    param_group.add_argument(
        "--m_sensitivity",
        type=float,
        help="Sensitivity of contractility increase when cells become mesoderm",
    )
    param_group.add_argument(
        "--morphogen_feedback",
        type=float,
        help="R - morphogen sensitivity for chemical feedback",
    )
    param_group.add_argument(
        "--proliferation_factor",
        type=float,
        default=1.0,
        help="Multiplies the proliferation term ρ(1-ρ) to simulate "
        "increased or decreased proliferation.",
    )

    # Mode parameter
    param_group.add_argument(
        "--feedback_mode",
        type=str,
        choices=["active_stress", "strain_rate"],
        help="Mode of mechanical feedback",
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
        optimize_args=True,
    )


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
