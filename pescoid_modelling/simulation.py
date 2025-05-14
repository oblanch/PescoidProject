"""Pescoid fluid-dynamics simulation."""

from pathlib import Path
from typing import Dict, Tuple

from dolfin import assign  # type: ignore
from dolfin import conditional  # type: ignore
from dolfin import Constant  # type: ignore
from dolfin import dx  # type: ignore
from dolfin import Expression  # type: ignore
from dolfin import Form  # type: ignore
from dolfin import Function  # type: ignore
from dolfin import FunctionSpace  # type: ignore
from dolfin import IntervalMesh  # type: ignore
from dolfin import lhs  # type: ignore
from dolfin import LogLevel  # type: ignore
from dolfin import lt  # type: ignore
from dolfin import MixedFunctionSpace  # type: ignore
from dolfin import project  # type: ignore
from dolfin import rhs  # type: ignore
from dolfin import set_log_level  # type: ignore
from dolfin import solve  # type: ignore
from dolfin import split  # type: ignore
from dolfin import tanh  # type: ignore
from dolfin import TestFunctions  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore

from pescoid_modelling.config import SimulationParams
from pescoid_modelling.constants import LEADING_EDGE_THRESHOLD
from pescoid_modelling.constants import M_SENSITIVITY
from pescoid_modelling.constants import RHO_GATE_CENTER
from pescoid_modelling.constants import RHO_GATE_WIDTH
from pescoid_modelling.constants import RHO_SENSITIVITY
from pescoid_modelling.constants import SNAPSHOT_EVERY_N_STEPS
from pescoid_modelling.simulation_logger import SimulationLogger


class PescoidSimulator:
    """Finite-element integrator for the non-dimensional PESC model.

    Attributes:
      params: Simulation parameters.
      work_dir: Directory for simulation output.
      aborted: Flag indicating if the simulation was aborted.
      mesh: Mesh for the simulation domain.
      final_time: Total simulation time.
      time_step: Time step for the simulation.
      num_steps: Number of time steps in the simulation.
      mixed_function_space: Mixed function space for the simulation variables.
      previous_state: Function representing the previous state of the system.
      current_state: Function representing the current state of the system.
      forms: Variational forms for the PDEs.
      logger: Logger for recording simulation results.

    Examples::
      # Initialize the simulator with parameters and work directory
      >>> params = SimulationParams(...)
      >>> simulator = PescoidSimulator(params, work_dir="output")

      # Run the simulation
      >>> simulator.run()

      # Save the results
      >>> simulator.save("simulation_results.npz")

      # Access the results
      >>> results = simulator.results
    """

    def __init__(self, parameters: SimulationParams, work_dir: str | Path) -> None:
        """Initialize the simulator class."""
        self.params = parameters
        self.aborted: bool = False
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        set_log_level(LogLevel.ERROR)
        self._set_simulation_time()

    def _set_simulation_time(
        self, total_hours: float = 12.0, minutes_per_generation: float = 30.0
    ) -> None:
        """Set the simulation time based on the number of generations."""
        self.final_time = (total_hours * 60.0) / minutes_per_generation
        self.time_step = self.params.delta_t
        self.num_steps = int(round(self.final_time / self.time_step))

    def run(self) -> "PescoidSimulator":
        """Run the simulation."""
        self._initialize_simulation_and_solver()

        for step_idx in tqdm(range(self.num_steps), desc="Simulation time steps"):
            if not self._advance(step_idx):
                self.aborted = True
                break

        self.logger.finalize()
        return self

    def _initialize_simulation_and_solver(self) -> None:
        """Set up the simulation environment."""
        self._setup_mesh()
        self._initialize_logger()
        self._setup_function_spaces()
        self._set_initial_conditions()
        self._build_variational_forms()

    def _setup_mesh(self) -> None:
        """Create the simulation mesh with appropriate resolution."""
        domain_length = getattr(self.params, "domain_length", 10.0)
        mesh_spacing = getattr(self.params, "dx_interval", 5e-3)
        num_mesh_points = int(round(domain_length / mesh_spacing))
        half_domain_length = domain_length / 2.0

        self._mesh = IntervalMesh(
            num_mesh_points, -half_domain_length, half_domain_length
        )

    def _initialize_logger(self) -> None:
        """Initialize the simulation logger to record results."""
        if self._mesh is None:
            raise RuntimeError("Mesh must be set up before initializing logger.")

        self.logger = SimulationLogger(
            num_steps=self.num_steps,
            mesh_size=len(self._mesh.coordinates().flatten()),
            snapshot_interval=SNAPSHOT_EVERY_N_STEPS,
        )
        self.logger.x_coordinates = self._mesh.coordinates().flatten().copy()

    def _setup_function_spaces(self) -> None:
        """Create function spaces for the simulation variables."""
        if self._mesh is None:
            raise RuntimeError("Mesh must be set up before creating function spaces.")

        self._mixed_function_space = MixedFunctionSpace(
            [FunctionSpace(self._mesh, "CG", 1) for _ in range(3)]
        )
        self._previous_state = Function(self._mixed_function_space)
        self._current_state = Function(self._mixed_function_space)

    def _set_initial_conditions(self) -> None:
        """Set initial conditions for density, mesoderm, and velocity."""
        if self._previous_state is None:
            raise RuntimeError(
                "Function spaces must be set up before setting initial conditions."
            )

        initial_amplitude = 1.0
        transition_width = 1e-2
        length_scale = self.params.length_scale

        # Define initial condition expressions
        density_expression = (
            f"({initial_amplitude}/2) * ((-tanh((pow(x[0], 2) - {length_scale})/{transition_width})) + 1)"
            "* pow(0.5, x[0]*x[0])"
        )
        density_initial_condition = Expression(density_expression, degree=2)
        mesoderm_initial_condition = Expression("-1.0", degree=1)
        velocity_initial_condition = Expression("0.0", degree=1)

        # Assign initial conditions to function
        assign(self._previous_state.sub(0), density_initial_condition)
        assign(self._previous_state.sub(1), mesoderm_initial_condition)
        assign(self._previous_state.sub(2), velocity_initial_condition)

    def _build_variational_forms(self) -> None:
        """Build the variational forms for the PDEs."""
        if self._previous_state is None or self._current_state is None:
            raise RuntimeError("Function states must be set up before building forms.")

        self._forms = self._formulate_variational_problem()

    def _formulate_variational_problem(self) -> Tuple[Expression, Expression]:
        """Build the variational forms for the PDEs."""
        # split unknowns / tests
        rho_prev, m_prev, u_prev = split(self._previous_state)  # type: ignore
        rho, m, u = split(self._current_state)  # type: ignore
        t_rho, t_m, t_u = TestFunctions(self._mixed_function_space)  # type: ignore

        # Formulate residuals
        res_rho = self._formulate_density_equation(
            rho=rho,  # type: ignore
            rho_prev=rho_prev,  # type: ignore
            u_prev=u_prev,
            t_rho=t_rho,  # type: ignore
        )
        res_m = self._formulate_mesoderm_equation(
            m=m,
            m_prev=m_prev,
            rho_prev=rho_prev,  # type: ignore
            u_prev=u_prev,
            t_m=t_m,
        )
        res_u = self._formulate_velocity_equation(
            u=u,
            rho_prev=rho_prev,  # type: ignore
            m_prev=m_prev,
            t_u=t_u,
        )

        # Combine residuals
        total_residual = res_rho + res_m + res_u
        return lhs(total_residual), rhs(total_residual)  # type: ignore

    def _formulate_density_equation(
        self,
        rho: Function,
        rho_prev: Function,
        u_prev: Function,
        t_rho: Function,
    ) -> Form:
        """Formulate the variational form for the density equation (tissue
        growth). Equation is of the following form:

            d rho / dt = Delta * d^2 rho / dx^2 - Flow * u_prev * d rho_prev /
            dx + rho_prev * (1 - rho_prev)
        """
        dt = self.time_step

        # Logistic growth + diffusion + advection
        return (
            (rho - rho_prev) * t_rho * dx  # type: ignore
            + dt * self.params.diffusivity * rho.dx(0) * t_rho.dx(0) * dx  # type: ignore
            - dt * self.params.flow * u_prev * rho_prev * t_rho.dx(0) * dx  # type: ignore
            - dt * rho_prev * (1 - rho_prev) * t_rho * dx  # type: ignore
        )

    def _formulate_mesoderm_equation(
        self,
        m: Function,
        m_prev: Function,
        rho_prev: Function,
        u_prev: Function,
        t_m: Function,
    ) -> Form:
        """Formulate the variational form for mesoderm differentiation. The
        complete equation is:

        d m / dt = (1/tau_m) * m_prev * (m_prev + 1) * (1 - m_prev)
        + (1/tau_m) * R * [feedback term]
        + Delta * d^2 m / dx^2
        - Flow * u_prev * d m_prev / dx
        """
        dt = self.time_step
        feedback_mode = getattr(self.params, "feedback_mode", "strain_rate")

        # Common decay term
        common_decay = (
            dt  # type: ignore
            * (1 / self.params.tau_m)  # type: ignore
            * m_prev
            * (m_prev + 1)  # type: ignore
            * (1 - m_prev)  # type: ignore
            * t_m
            * dx
        )

        if feedback_mode == "strain_rate":
            feedback = self._formulate_strain_rate_feedback(rho_prev, m_prev, t_m)
        else:  # active_stress
            feedback = self._formulate_active_stress_feedback(u_prev, t_m)

        # Complete residual
        return (
            (m - m_prev) * t_m * dx  # type: ignore
            - common_decay
            - feedback
            + dt * self.params.diffusivity * m.dx(0) * t_m.dx(0) * dx  # type: ignore
            + dt * self.params.flow * u_prev * m_prev.dx(0) * t_m * dx  # type: ignore
        )

    def _formulate_strain_rate_feedback(
        self,
        rho_prev: Function,
        m_prev: Function,
        t_m: Function,
    ) -> Form:
        """Formulate the strain rate feedback term for mesoderm differentiation:

        (1/tau_m) * R * (stress_term - Sigma_c)

        where

        stress_term = density_prev * Activity
        * (density_prev/(1 + RHO_SENSITIVITY * density_prev^2))
        * (1 + Beta * (
            (tanh((mesoderm_prev - M_SENSITIVITY)/M_SENSITIVITY) + 1)/2)
        )
        """
        dt = self.time_step

        stress_term = (
            rho_prev
            * self.params.activity  # type: ignore
            * (rho_prev / (1 + RHO_SENSITIVITY * rho_prev * rho_prev))  # type: ignore
            * (
                1
                + self.params.beta
                * ((tanh((m_prev - M_SENSITIVITY) / M_SENSITIVITY) + 1) / 2)  # type: ignore
            )
        )
        return (
            dt
            * (1 / self.params.tau_m)
            * self.params.r
            * (stress_term - self.params.sigma_c)
            * t_m
            * dx
        )

    def _formulate_active_stress_feedback(
        self,
        u_prev: Function,
        t_m: Function,
    ) -> Form:
        """Formulate the active stress feedback term for mesoderm
        differentiation:

        (1/tau_m) * R * (Sigma_c - velocity_prev.dx(0))
        """
        dt = self.time_step

        return (
            dt
            * (1 / self.params.tau_m)
            * self.params.r
            * (self.params.sigma_c - u_prev.dx(0))  # type: ignore
            * t_m
            * dx
        )

    def _formulate_velocity_equation(
        self,
        u: Function,
        rho_prev: Function,
        m_prev: Function,
        t_u: Function,
    ) -> Form:
        """Formulate the variational form for tissue velocity (force balance
        equation):

        rho_gate * Gamma * u + rho_gate * d u / dx = div(stress)
        """
        # Density gating function
        rho_gate = 0.5 * (tanh((rho_prev - RHO_GATE_CENTER) / RHO_GATE_WIDTH) + 1)  # type: ignore

        # Stress divergence term
        stress_div = self._calculate_stress_divergence(rho_prev, m_prev)

        return (
            rho_gate * self.params.gamma * u * t_u * dx
            + rho_gate * u.dx(0) * t_u.dx(0) * dx  # type: ignore
            - stress_div * t_u * dx  # type: ignore
        )

    def _calculate_stress_divergence(
        self, rho_prev: Function, m_prev: Function
    ) -> Function:
        """Calculate the divergence of the active stress tensor:

        d/dx [density_prev
            * Activity
            * (density_prev/(1 + RHO_SENSITIVITY * density_prev^2))
            * (
                1
                + Beta
                * ((tanh((mesoderm_prev - M_SENSITIVITY)/M_SENSITIVITY) + 1)/2)
            ) - 1]
        """
        return (
            rho_prev
            * self.params.activity  # type: ignore
            * (rho_prev / (1 + RHO_SENSITIVITY * rho_prev * rho_prev))  # type: ignore
            * (
                1
                + self.params.beta
                * ((tanh((m_prev - M_SENSITIVITY) / M_SENSITIVITY) + 1) / 2)  # type: ignore
            )
            - 1
        ).dx(0)

    def _advance(self, step_idx: int) -> bool:
        """Advance the simulation by one time step."""
        lhs_form, rhs_form = self._forms
        solve(lhs_form == rhs_form, self._current_state)
        self._previous_state.assign(self._current_state)

        rho_fn, m_fn, u_fn = self._current_state.split(deepcopy=True)
        if self._check_for_errant_values(rho_fn) or self._check_for_errant_values(m_fn):
            return False

        rho_fn.assign(self._ensure_non_negative(rho_fn))
        assign(self._current_state.sub(0), rho_fn)

        self._log_simulation_state(step_idx)
        return True

    def _check_for_errant_values(self, var_fn: Function) -> bool:
        """Check for errant values in the variable function."""
        if np.any(np.isnan(var_fn.vector().get_local())):
            print("Aborting: NaN detected in solution.")
            return True
        if np.any(np.isinf(var_fn.vector().get_local())):
            print("Aborting: Inf detected in solution.")
            return True
        return False

    def _log_simulation_state(self, step_idx: int) -> None:
        """Take a snapshot of the simulation state."""
        current_time = (step_idx + 1) * self.time_step
        if self.logger.should_log(step_idx):
            self._log_state(step_idx, current_time)

    def _compute_stress(self, rho_fn: Function, m_fn: Function) -> np.ndarray:
        """Compute stress based on the current state."""
        scalar_space = FunctionSpace(self._mesh, "CG", 1)
        stress_expression = (
            rho_fn
            * self.params.activity  # type: ignore
            * (rho_fn / (1 + RHO_SENSITIVITY * rho_fn * rho_fn))  # type: ignore
            * (
                1
                + self.params.beta
                * ((tanh((m_fn - M_SENSITIVITY) / M_SENSITIVITY) + 1) / 2)  # type: ignore
            )
        )
        stress_fn = project(stress_expression, scalar_space)
        return stress_fn.vector().get_local()

    def _compute_leading_edge(
        self, rho_vals: np.ndarray, x_coords: np.ndarray
    ) -> Tuple[float, int]:
        """Compute the leading edge position based on density values."""
        mask = np.abs(rho_vals - LEADING_EDGE_THRESHOLD) < 1e-1
        if np.any(mask):
            edge_x = x_coords[mask][-1]
            edge_idx = mask.nonzero()[0][-1]
        else:
            edge_x = x_coords[-1]
            edge_idx = len(x_coords) - 1
        return edge_x, edge_idx

    def _log_state(self, step_idx: int, current_time: float) -> None:
        """Calculate and log the simulation state."""
        rho_fn, m_fn, u_fn = self._current_state.split()
        rho_vals = rho_fn.compute_vertex_values(self._mesh)
        m_vals = m_fn.compute_vertex_values(self._mesh)
        u_vals = u_fn.compute_vertex_values(self._mesh)
        x_coords = self._mesh.coordinates().flatten()
        max_m = m_vals.max()
        meso_frac = (m_vals > 0).mean()

        edge_x, edge_idx = self._compute_leading_edge(rho_vals, x_coords)
        stress_vals = self._compute_stress(rho_fn, m_fn)

        self.logger.log_snapshot(
            step_idx=step_idx,
            current_time=current_time,
            rho_vals=rho_vals,
            m_vals=m_vals,
            u_vals=u_vals,
            stress_vals=stress_vals,
            edge_x=edge_x,
            edge_idx=edge_idx,
            meso_frac=meso_frac,
            max_m=max_m,
        )

    def save(self, file: str | Path) -> None:
        out = Path(file).with_suffix(".npz")
        np.savez_compressed(out, allow_pickle=False, **self.results)

    @property
    def results(self) -> Dict[str, np.ndarray]:
        """Return simulation results as a dictionary."""
        result_dict = self.logger.to_dict()
        result_dict["aborted"] = np.asarray([self.aborted])
        result_dict["dt"] = np.asarray([self.time_step])
        return result_dict

    @staticmethod
    def _ensure_non_negative(var_fn: Function) -> Function:
        """Ensure the variable function is non-negative."""
        V = var_fn.function_space()
        corrected = Function(V)
        corrected_expr = conditional(lt(var_fn, 0), Constant(0.0), var_fn)
        corrected.assign(project(corrected_expr, V))
        return corrected
