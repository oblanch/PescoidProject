"""Run the Pescoid simulation."""

import argparse
from pathlib import Path

from pescoid_modelling.config import load_config
from pescoid_modelling.simulation import PescoidSimulator


def main() -> None:
    """Main function to run a single Pescoid simulation."""
    parser = argparse.ArgumentParser(description="Run Pescoid simulation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="simulation_results",
        help="Directory to store simulation results.",
    )

    args = parser.parse_args()

    # Load configuration
    sim_params, _ = load_config(args.config)

    # Create work directory
    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run simulator
    print("Starting Pescoid simulation...")
    simulator = PescoidSimulator(parameters=sim_params, work_dir=work_dir)
    simulator.run()

    # Save results
    output_file = work_dir / "simulation_results.npz"
    simulator.save(output_file)

    print(f"Simulation complete. Results saved to {output_file}")

    # Print summary statistics
    results = simulator.results
    if "boundary_positions" in results and len(results["boundary_positions"]) > 0:
        final_position = results["boundary_positions"][-1]
        print(f"Final boundary position: {final_position:.4f}")

    if "aborted" in results and results["aborted"][0]:
        print("Note: Simulation was aborted before completion.")


if __name__ == "__main__":
    main()
