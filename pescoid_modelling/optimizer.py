"""Parameter optimization using CMA-ES."""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import cma  # type: ignore
import numpy as np
import psutil  # type: ignore
from tqdm import tqdm  # type: ignore

from pescoid_modelling.config import SimulationParams
from pescoid_modelling.simulation import PescoidSimulator


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    cores = psutil.cpu_count(logical=False)
    if cores is None or cores <= 1:
        return 1
    return cores - 1


def optimization_objective(results: Dict[str, np.ndarray]) -> float:
    """Objective function that evaluates simulation results based on transition
    dynamics and mesoderm timing. For now, we use a simple objective based on
    minimizing the difference in the final boundary position.
    """
    boundary_positions = results.get("boundary_positions", np.array([0.0]))
    if len(boundary_positions) == 0:
        return 1e9  # Large penalty for failed simulations

    # Use the last boundary position as the objective
    return float(boundary_positions[-1] ** 2)


class CMAOptimizer:
    """Run CMA-ES to minimize an objective produced by PescoidSimulator."""

    _ORDER = [
        "length_scale",
        "Delta",
        "Flow",
        "tau_m",
        "Gamma",
        "Activity",
        "Beta",
        "Sigma_c",
        "R",
        "delta_t",
    ]

    def __init__(
        self,
        work_root: Path,
        base_params: SimulationParams,
        init_guess: List[float],
        sigma: float,
        bounds: Union[Tuple[List[float], List[float]], None] = None,
        max_evals: int = 256,
        obj_fn: Callable[[Dict[str, np.ndarray]], float] = optimization_objective,
    ) -> None:
        """Initialize the CMA-ES optimizer.

        Args:
            work_root: Directory to store simulation results
            base_params: Base simulation parameters
            init_guess: Initial parameter vector
            sigma: Initial step size
            bounds: Parameter bounds as (lower_bounds, upper_bounds)
            max_evals: Maximum number of function evaluations
            obj_fn: Objective function to minimize
            n_workers: Number of parallel workers (defaults to CPU count - 1)
        """
        self.base_params = base_params
        self.obj_fn = obj_fn
        self.max_evals = max_evals
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)

        self.n_workers = get_physical_cores()

        opts: dict = {
            "verb_disp": 1,
            "maxfevals": max_evals,
            "popsize": 8,
        }
        if bounds is not None:
            opts["bounds"] = list(bounds)

        self.es = cma.CMAEvolutionStrategy(init_guess, sigma, opts)

    def _evaluate_individual(self, x: List[float]) -> float:
        """Evaluate a single parameter vector by running a simulation.

        Args:
            x: Parameter vector to evaluate

        Returns:
            Objective function value (lower is better)
        """
        # Unique hash for caching
        p = self.vector_to_params(x, self.base_params)
        param_hash = hash(tuple(x))
        job_dir = self.work_root / f"sim_{param_hash:x}"

        simulator = PescoidSimulator(p, job_dir)
        try:
            simulator.run()
            results = simulator.results
        except Exception as exc:
            print(f"Simulation failure ({exc}), penalizing...")
            return 1e9

        return self.obj_fn(results)

    def optimize(self) -> SimulationParams:
        """Run the CMA-ES optimization process.

        Returns:
            Optimized simulation parameters
        """
        with mp.Pool(processes=self.n_workers) as pool:
            while not self.es.stop():
                X = self.es.ask()
                fitness: List[float] = list(
                    tqdm(
                        pool.imap(self._evaluate_individual, X),
                        total=len(X),
                        desc="Evaluating population",
                    )
                )
                self.es.tell(X, fitness)
                self.es.disp()

        best_x = self.es.result.xbest
        return self.vector_to_params(best_x, self.base_params)

    @classmethod
    def vector_to_params(
        cls, v: List[float], base: SimulationParams
    ) -> SimulationParams:
        """Convert a vector of parameter values to SimulationParams."""
        kwargs = asdict(base)
        for k, val in zip(cls._ORDER, v):
            kwargs[k] = val
        return SimulationParams(**kwargs)
