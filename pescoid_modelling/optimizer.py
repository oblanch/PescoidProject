"""Parameter optimization using CMA-ES."""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Sequence, Tuple, Union

import cma  # type: ignore
from matplotlib import pyplot as plt
import numpy as np
import psutil  # type: ignore
from tqdm import tqdm  # type: ignore

from pescoid_modelling.objective import _invalid_fitness
from pescoid_modelling.objective import optimization_objective
from pescoid_modelling.objective import ReferenceTrajectories
from pescoid_modelling.simulation import PescoidSimulator
from pescoid_modelling.utils.config import _ORDER
from pescoid_modelling.utils.config import SimulationParams
from pescoid_modelling.utils.helpers import get_physical_cores
from pescoid_modelling.utils.parameter_scaler import ParamScaler
from pescoid_modelling.visualization import _set_matplotlib_publication_parameters

GLOBAL_EMA: MutableMapping[str, float] | None = None


def _init_worker(shared_ema_proxy: MutableMapping[str, float]) -> None:
    """Exectued once in every work process to make the manager-dict available as the
    module-level global_ema. This allows the objective function to access the shared
    exponential moving average without passing it explicitly.
    """
    global GLOBAL_EMA
    GLOBAL_EMA = shared_ema_proxy


def vector_to_params(
    v_norm: List[float], base: SimulationParams, scaler: ParamScaler
) -> SimulationParams:
    """Convert normalised vector to physical SimulationParams."""
    v_phys = scaler.to_physical(v_norm)
    kwargs = asdict(base)
    for k, val in zip(_ORDER, v_phys):
        kwargs[k] = float(val)
    return SimulationParams(**kwargs)


def _worker_eval(
    x: List[float],
    base_params: SimulationParams,
    work_dir: Path,
    experimental: ReferenceTrajectories,
    scaler: ParamScaler,
    obj_fn: Callable[..., float],
) -> float:
    """Executed in the pool; has no reference to the CMAOptimizer
    instance to avoid pickling issues.
    """
    p = vector_to_params(x, base_params, scaler)
    job_dir = work_dir / f"sim_{hash(tuple(x)):x}"
    sim = PescoidSimulator(p, job_dir)
    try:
        sim.run()
        res = sim.results
        if res.get("aborted", [False])[0]:
            raise RuntimeError("Invalid simulation results")

        return obj_fn(res, experimental, ema_dict=GLOBAL_EMA)
    except Exception as exc:
        print(f"Simulation failure ({exc}), penalizing...")
        return _invalid_fitness()


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
        x0: List[float],
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
        self._sigma0 = sigma
        self._x0 = x0
        self._n_restarts = n_restarts

        self.n_workers = get_physical_cores()
        self._mp_manager = mp.Manager()
        self.shared_ema: MutableMapping[str, float] = self._mp_manager.dict()  # type: ignore

        opts: Dict[str, Any] = {
            "verb_disp": 1,
            "maxfevals": max_evals,
            "popsize": popsize,
        }
        if bounds is None:
            raise ValueError("Bounds must be provided for normalization")

        log_axes = {"diffusivity", "gamma", "m_sensitivity"}
        self.scaler = ParamScaler(
            lower=bounds[0],
            upper=bounds[1],
            log_mask=[name in log_axes for name in _ORDER],
        )

        x0_norm = self.scaler.to_normalized(x0)
        opts["bounds"] = [np.zeros_like(x0_norm), np.ones_like(x0_norm)]

        self._x0 = x0_norm.tolist()
        self._es_opts = opts
        self.es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, self._es_opts)

    def _evaluate_individual(self, x: List[float]) -> float:
        """Evaluate a single parameter vector by running a simulation.

        Args:
          x: Parameter vector to evaluate

        Returns:
          Objective function value (lower is better)
        """
        # Unique hash for caching
        p = vector_to_params(x, self.base_params, self.scaler)
        param_hash = hash(tuple(x))
        job_dir = self.work_dir / f"sim_{param_hash:x}"

        simulator = PescoidSimulator(p, job_dir)
        try:
            simulator.run()
            results = simulator.results
            if results.get("aborted", [False])[0]:
                raise RuntimeError("Invalid simulation results")

            return self.objective_function(
                results, self.experimental_data, ema_dict=GLOBAL_EMA
            )

        except Exception as exc:
            print(f"Simulation failure ({exc}), penalizing...")
            return _invalid_fitness()

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

        with mp.Pool(
            processes=self.n_workers,
            initializer=_init_worker,
            initargs=(self.shared_ema,),
        ) as pool:
            while True:
                logger = cma.CMADataLogger(str(self.work_dir / f"cma_restart.log"))

                while not self.es.stop():
                    iteration += 1
                    X = self.es.ask()
                    fitness = list(
                        tqdm(
                            pool.starmap(
                                _worker_eval,
                                [
                                    (
                                        x,
                                        self.base_params,
                                        self.work_dir,
                                        self.experimental_data,
                                        self.scaler,
                                        self.objective_function,
                                    )
                                    for x in X
                                ],
                            ),
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

                self.es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, new_opts)
                logger = cma.CMADataLogger(str(self.work_dir / "cma_optimization_data"))

        self._generate_optimization_plots(logger)
        best_x_norm = self.es.result.xbest
        return vector_to_params(best_x_norm, self.base_params, self.scaler)

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
