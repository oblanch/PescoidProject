"""Parameter optimization using CMA-ES."""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from typing import Callable, List

import cma  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm

from pescoid_modelling.config import SimulationParams
from pescoid_modelling.simulation import PescoidSimulator

# ---------------------------------------------------------------------------
# Parameter <‑‑> vector helpers
# ---------------------------------------------------------------------------
_ORDER = [
    "diffusivity",
    "tau_m",
    "length_scale",
    "flow",
    "activity",
    "beta",
    "r",
]


def params_to_vector(p: SimulationParams) -> List[float]:
    return [getattr(p, k) for k in _ORDER]


def vector_to_params(v: List[float], base: SimulationParams) -> SimulationParams:
    kwargs = asdict(base)
    for k, val in zip(_ORDER, v):
        kwargs[k] = val
    return SimulationParams(**kwargs)


# ---------------------------------------------------------------------------
# Objective function stub
# ---------------------------------------------------------------------------


def default_objective(res: PescoidSimulator) -> float:
    """Example objective: minimise final boundary position drift (dummy)."""
    return float(res.boundary_positions[-1] ** 2)


# ---------------------------------------------------------------------------
# Optimiser façade
# ---------------------------------------------------------------------------


class CMAOptimizer:
    """Run CMA‑ES to minimise an objective produced by *pesc* simulations."""

    def __init__(
        self,
        work_root: Path,
        base_params: SimulationParams,
        init_guess: List[float],
        sigma: float,
        bounds: tuple[List[float], List[float]] | None = None,
        max_evals: int = 256,
        obj_fn: Callable[[PescoidSimulator], float] = default_objective,
        n_workers: int | None = None,
    ) -> None:
        self.work_root = work_root.expanduser().resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.base_params = base_params
        self.obj_fn = obj_fn
        self.max_evals = max_evals
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

        opts: dict = {
            "verb_disp": 1,
            "maxfevals": max_evals,
            "popsize": 8,
        }
        if bounds is not None:
            opts["bounds"] = list(bounds)

        self.es = cma.CMAEvolutionStrategy(init_guess, sigma, opts)

    # .................................................................
    def _evaluate_individual(self, x: List[float]) -> float:
        # Derive unique hash for reproducibility / caching
        p = vector_to_params(x, self.base_params)
        job_dir = self.work_root / f"sim_{hash(tuple(x))}"
        sim = PescoidSimulator(p, job_dir)
        try:
            res = sim.run()
        except Exception as exc:  # simulation crashed ⇒ large penalty
            print(f"Simulation failure ({exc}), penalising…")
            return 1e9
        return self.obj_fn(res)

    # .................................................................
    def optimise(self) -> SimulationParams:
        with mp.Pool(processes=self.n_workers) as pool:
            while not self.es.stop():
                # Ask for a batch of candidate points
                X = self.es.ask()
                # Parallel evaluation
                fitness: List[float] = list(
                    tqdm(
                        pool.imap(self._evaluate_individual, X),
                        total=len(X),
                        desc="Evaluating pop",
                    )
                )
                self.es.tell(X, fitness)
                self.es.disp()

        best_x = self.es.result.xbest
        return vector_to_params(best_x, self.base_params)

    def __call__(self, x: List[float]) -> float:
        return self._evaluate_individual(x)
