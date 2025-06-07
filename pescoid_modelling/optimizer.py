"""Parameter optimization using CMA-ES."""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import cma  # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import psutil  # type: ignore
from tqdm import tqdm  # type: ignore

from pescoid_modelling.objective import optimization_objective
from pescoid_modelling.objective import ReferenceTrajectories
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.config import SimulationParams
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters


def get_physical_cores() -> int:
    """Return physical core count, subtracted by one to account for the main
    process / overhead.
    """
    cores = psutil.cpu_count(logical=False)
    if cores is None or cores <= 1:
        return 1
    return cores - 1


class CMAOptimizer:
    """Run CMA-ES to minimize an objective produced by PescoidSimulator.

    Examples::
        # Instantiate the optimizer
        >>> optimizer = CMAOptimizer(
        ...     work_dir=Path("path/to/work_dir"),
        ...     base_params=SimulationParams(),
        ...     init_guess=[1.0, 2.0, 3.0],
        ...     sigma=0.5,
        ...     experimental_data=ExperimentalTrajectories(),
        ...     bounds=([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        ...     max_evals=100,
        ...     popsize=8,
        ...     objective_function=optimization_objective,
        ... )

        # Run the optimization and return the best parameters
        >>> optimized_params = optimizer.optimize()

    """

    def __init__(
        self,
        work_dir: Path,
        base_params: SimulationParams,
        init_guess: List[float],
        sigma: float,
        experimental_data: ReferenceTrajectories,
        bounds: Union[Tuple[List[float], List[float]], None] = None,
        max_evals: int = 256,
        popsize: int = 8,
        n_restarts: int = 4,
        objective_function: Callable[..., float] = optimization_objective,
    ) -> None:
        """Initialize the CMA-ES optimizer.

        Args:
            work_dir: Directory to store simulation results
            base_params: Base simulation parameters
            init_guess: Initial parameter vector
            sigma: Initial step size
            bounds: Parameter bounds as (lower_bounds, upper_bounds)
            max_evals: Maximum number of function evaluations
            objective_function: Objective function to minimize
            n_workers: Number of parallel workers (defaults to CPU count - 1)
        """
        self.base_params = base_params
        self.objective_function = objective_function
        self.experimental_data = experimental_data
        self.work_dir = work_dir

        self.n_workers = get_physical_cores()

        opts: Dict[str, Any] = {
            "verb_disp": 1,
            "maxfevals": max_evals,
            "popsize": popsize,
        }
        if bounds is not None:
            opts["bounds"] = [list(bounds[0]), list(bounds[1])]

        self._es_opts = opts
        self._sigma0 = sigma
        self._x0 = init_guess
        self._n_restarts = n_restarts

        self.es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, self._es_opts)

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
        job_dir = self.work_dir / f"sim_{param_hash:x}"

        simulator = PescoidSimulator(p, job_dir)
        try:
            simulator.run()
            results = simulator.results

            invalid = (
                results.get("aborted", [False])[0]
                or not np.isfinite(results["tissue_size"]).all()
                or not np.isfinite(results["mesoderm_signal"]).all()
            )

            if invalid:
                raise RuntimeError("Invalid simulation results")

            return self.objective_function(results, self.experimental_data)

        except Exception as exc:
            print(f"Simulation failure ({exc}), penalizing...")
            return 1e9 * (1.0 + 0.01 * np.random.rand())

    def optimize(self) -> SimulationParams:
        """Run the CMA-ES optimization process.

        Returns:
          Optimized simulation parameters
        """
        logger = cma.CMADataLogger(str(self.work_dir / "cma_optimization_data"))
        log_file = self.work_dir / "optimization_progress.csv"
        with open(log_file, "w") as f:
            f.write(
                "iteration,evaluations,best_fitness,mean_fitness,sigma,axis_ratio\n"
            )

        restarts_left = self._n_restarts
        iteration = 0

        with mp.Pool(processes=self.n_workers) as pool:
            while True:
                logger = cma.CMADataLogger(str(self.work_dir / f"cma_restart.log"))

                while not self.es.stop():
                    iteration += 1
                    X = self.es.ask()
                    fitness = list(
                        tqdm(
                            pool.imap(self._evaluate_individual, X),
                            total=len(X),
                            desc=f"Gen {iteration} (rst {restarts_left})",
                        )
                    )
                    self.es.tell(X, fitness)
                    self.es.disp()
                    logger.add(self.es)
                    with open(log_file, "a") as f:
                        f.write(
                            f"{iteration},{self.es.countevals},"
                            f"{self.es.result.fbest},{np.mean(fitness)},"
                            f"{self.es.sigma},{self.es.D.max()/self.es.D.min()}\n"  # type: ignore
                        )

                if restarts_left == 0:
                    break

                restarts_left -= 1
                self._x0 = self.es.result.xbest
                self._sigma0 = self.es.sigma
                new_opts = dict(self._es_opts)
                new_opts["popsize"] = int(self.es.popsize * 2)

                self.es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, new_opts)
                logger = cma.CMADataLogger(str(self.work_dir / "cma_optimization_data"))

        self._generate_optimization_plots(logger)
        best_x = self.es.result.xbest
        return self.vector_to_params(best_x, self.base_params)

    def _generate_optimization_plots(self, logger: cma.CMADataLogger) -> None:
        """Generate and save optimization plots."""
        _set_matplotlib_publication_parameters()
        logger.plot()
        figures = plt.get_fignums()
        for fig_num in figures:
            fig = plt.figure(fig_num)
            fig.savefig(self.work_dir / f"cma_plot_{fig_num}.png", dpi=450)

        plt.close("all")

        print(f"Optimization plots saved to {self.work_dir}")

    @staticmethod
    def vector_to_params(v: List[float], base: SimulationParams) -> SimulationParams:
        """Convert a vector of parameter values to SimulationParams."""
        kwargs = asdict(base)
        for k, val in zip(_ORDER, v):
            kwargs[k] = val
        return SimulationParams(**kwargs)
