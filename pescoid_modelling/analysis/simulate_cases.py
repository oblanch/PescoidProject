"""Run simulations against experimental cases."""

from pathlib import Path
import subprocess

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from pescoid_modelling.visualization import _set_matplotlib_publication_parameters
from pescoid_modelling.visualization.plot_trajectories import _load_trajectory_data
from pescoid_modelling.visualization.plot_trajectories import (
    _set_matplotlib_publication_parameters,
)


def get_cases():
    """Return param adjusted test cases for simulation.

    NOTE* we skipped typehinting this function because mypy wasn't playing well
    without unecessary workarounds. Apologies!
    """
    base_params = {
        "delta_t": 0.01,
        "total_hours": 12.0,
        "domain_length": 10.0,
        "dx_interval": 0.01,
        "diffusivity": 0.00026361376409494525,
        "m_diffusivity": 2e-3,
        "tau_m": 6.242153148113385,
        "flow": 0.19627128512102468,
        "activity": 0.8749997229064099,
        "beta": 0.9969669403539715,
        "gamma": 0.538795096984612,
        "sigma_c": 0.1,
        "r": 0.968870632027848,
        "rho_sensitivity": 0.0,
        "m_sensitivity": 0.05070302382000081,
        "morphogen_feedback": 1.2337528178981996,
        "proliferation_factor": 1.0,
        "feedback_mode": "active_stress",
    }

    cases = {
        "control": {
            **base_params,
        },
        "blebbistatin": {
            **base_params,
            "activity": base_params["activity"] * 0.1,
            "beta": base_params["beta"] * 0.05,
            "proliferation_factor": base_params["proliferation_factor"] * 0.05,
        },
        "rock_inhibitor": {
            **base_params,
            "r": base_params["r"] * 0.95,
            "activity": base_params["activity"] * 0.98,
            # "sigma_c": base_params["sigma_c"] * 1.1,
        },
        # "increased_proliferation": { **base_params, "proliferation_factor":
        #     base_params["proliferation_factor"] * 1.75, "r": base_params["r"]
        #     * 0.75, },
        # "activin": {
        #     **base_params,
        #     "proliferation_factor": base_params["proliferation_factor"] * 1.75,
        #     "sigma_c": 0.2,
        #     "beta": base_params["beta"] * 2.5,
        #     "activity": base_params["activity"] * 1.2,
        #     "gamma": base_params["gamma"] * 1.2,
        # },
    }
    return cases


def simulate_case(
    case_name: str,
    config_path: str = "../PescoidProject/configs/current_best.yaml",
    output_dir: str = "",
) -> bool:
    """Run a single experimental case via subprocess."""
    cases = get_cases()

    if case_name not in cases:
        available = list(cases.keys())
        raise ValueError(f"Case '{case_name}' not found. Available: {available}")

    case_params = cases[case_name]
    base_params = cases["control"]

    cmd = [
        "pescoid",
        "simulate",
        "--config",
        config_path,
        "--output_dir",
        output_dir,
        "--generate_figures",
        "--name",
        case_name,
    ]
    # cmd.extend([f"--dx_interval", str(0.01)])

    for param, value in case_params.items():
        if param != "feedback_mode" and value != base_params.get(param, None):
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{param}")
            else:
                cmd.extend([f"--{param}", str(value)])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd)
    print("-" * 50)
    if result.returncode == 0:
        print(f"SUCCESS: {case_name}")
        return True
    else:
        print(f"ERROR: {case_name} (exit code: {result.returncode})")
        return False


def plot_comparison_grid(
    control_data: str,
    blebbistatin_data: str,
    rock_data: str,
    save_path: str = "comparison_grid.svg",
) -> None:
    """Create a 2x3 comparison plot of simulations vs reference data."""
    _set_matplotlib_publication_parameters()

    cases = ["control", "rock_inhibitor", "blebbistatin"]
    case_labels = ["Control", "Rock inhibitor", "Blebbistatin"]
    ref_data_paths = [control_data, rock_data, blebbistatin_data]

    fig, axes = plt.subplots(2, 3, figsize=(3.05, 1.9))

    for i, (case, label, ref_path) in enumerate(
        zip(cases, case_labels, ref_data_paths)
    ):
        # sim
        ax1 = axes[0, i]
        ax2 = ax1.twinx()

        try:
            sim_data = _load_trajectory_data(
                f"{case}/simulation_results.npz", simulation=True
            )
            ax1.plot(
                sim_data["time"],
                sim_data["tissue_size"],
                color="tab:blue",
                linewidth=0.6,
                linestyle="-",
                label="Tissue size",
            )
            ax2.plot(
                sim_data["time"],
                sim_data["mesoderm_fraction"],
                color="tab:red",
                linewidth=0.6,
                linestyle="-.",
                label="Mesoderm fraction",
            )

            if case in ["control", "rock_inhibitor"]:
                peak_idx = np.argmax(sim_data["tissue_size"])
                transition_time = sim_data["time"][peak_idx]
                ax1.axvline(
                    transition_time,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.7,
                    label="Transition time",
                )

        except FileNotFoundError:
            print(f"Warning: Simulation data not found for {case}")

        if i == 0:
            ax1.set_ylabel("Tissue size", color="tab:blue")
        else:
            ax1.set_ylabel("")

        ax1.set_ylabel("")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.tick_params(axis="x", length=2, width=0.3, pad=1)
        ax1.spines["left"].set_color("tab:blue")
        ax1.set_yticks([])
        ax1.set_title(f"{label}")
        ax1.set_xmargin(0)

        ax2.set_ylabel("")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.spines["right"].set_color("tab:red")
        ax2.set_yticks([])
        ax2.set_xmargin(0)

        # ref
        ax3 = axes[1, i]
        ax4 = ax3.twinx()

        try:
            ref_data = _load_trajectory_data(ref_path, simulation=False)
            ax3.plot(
                ref_data["time"],
                ref_data["tissue_size"],
                color="tab:blue",
                linewidth=0.6,
                linestyle="-",
            )
            ax4.plot(
                ref_data["time"],
                ref_data["mesoderm_fraction"],
                color="tab:red",
                linewidth=0.6,
                linestyle="-.",
            )

            if case in ["control", "rock_inhibitor"]:
                peak_idx = np.argmax(ref_data["tissue_size"])
                transition_time = ref_data["time"][peak_idx]
                ax3.axvline(
                    transition_time,
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.7,
                )

        except FileNotFoundError:
            print(f"Warning: Reference data not found for {case}")

        ax3.set_ylabel("")
        ax3.tick_params(axis="y", labelcolor="tab:blue")
        ax3.tick_params(axis="x", length=2, width=0.3, pad=1)
        ax3.spines["left"].set_color("tab:blue")

        if i == 1:
            ax3.set_xlabel("Time (min)")
        else:
            ax3.set_xlabel("")

        ax3.set_yticks([])
        ax3.set_xmargin(0)
        ax4.set_ylabel("")

        ax4.tick_params(axis="y", labelcolor="tab:red")
        ax4.spines["right"].set_color("tab:red")
        ax4.set_yticks([])
        ax4.set_xmargin(0)

    fig.text(-0.01, 0.725, "Simulation", rotation=90, va="center", ha="center")
    fig.text(-0.01, 0.325, "Experiment", rotation=90, va="center", ha="center")

    handles = [
        Line2D([0], [0], color="tab:blue", linewidth=0.6, linestyle="-"),
        Line2D([0], [0], color="tab:red", linewidth=0.6, linestyle="-."),
        Line2D([0], [0], color="gray", linewidth=0.5, linestyle="--", alpha=0.7),
    ]
    labels = ["Tissue size", "Mesoderm fraction", "Transition time"]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.075),
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved to: {save_path}")


def main() -> None:
    """Run all cases."""
    control_data = "/Users/steveho/PescoidProject/data/reference_timeseries.npz"
    blebbistatin_data = (
        "/Users/steveho/PescoidProject/data/blebbistatin_reference_timeseries.npz"
    )
    rock_data = (
        "/Users/steveho/PescoidProject/data/rock_inhibitor_reference_timeseries.npz"
    )

    output_dir = Path("")
    cases = get_cases()
    for case_name in cases.keys():
        success = simulate_case(case_name, output_dir=str(output_dir))
        if not success:
            print(f"Simulation failed for case: {case_name}")

    plot_comparison_grid(
        control_data=control_data,
        blebbistatin_data=blebbistatin_data,
        rock_data=rock_data,
        save_path=str(output_dir / "comparison_grid.svg"),
    )


if __name__ == "__main__":
    main()
