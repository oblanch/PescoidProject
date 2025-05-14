"""Run the CMA-ES optimization algorithm on PescoidSimulator."""

import argparse
from pathlib import Path

from pescoid_modelling.config import load_config
from pescoid_modelling.optimizer import CMAOptimizer


def main() -> None:
    """Main function to run the CMA-ES optimization."""
    parser = argparse.ArgumentParser(description="Run CMA-ES optimization.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="optimization_results",
        help="Directory to store optimization results.",
    )

    args = parser.parse_args()

    # Load configuration
    sim_params_template, cma_cfg = load_config(args.config)

    # Create work directory
    work_root = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    # Initialize the optimizer
    optimizer = CMAOptimizer(
        work_root=work_root,
        base_params=sim_params_template,
        init_guess=cma_cfg.x0,
        sigma=cma_cfg.sigma0,
        bounds=cma_cfg.bounds,
        max_evals=cma_cfg.max_evals,
    )

    # Run optimization
    print("Starting CMA-ES optimization...")
    best_params = optimizer.optimize()

    # Save best parameters
    output_file = work_root / "best_parameters.txt"
    with open(output_file, "w") as f:
        f.write("# Optimized Parameters\n")
        f.write("# ------------------\n")
        for param_name in optimizer._ORDER:
            value = getattr(best_params, param_name)
            f.write(f"{param_name}: {value}\n")

    print(f"Optimization complete. Best parameters saved to {output_file}")


if __name__ == "__main__":
    main()
