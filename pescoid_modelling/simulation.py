"""Pescoid fluid-dynamics simulation."""

from pathlib import Path
from typing import Dict, Tuple

from dolfin import assign  # type: ignore
from dolfin import conditional  # type: ignore
from dolfin import Constant  # type: ignore
from dolfin import dx  # type: ignore
from dolfin import Expression  # type: ignore
from dolfin import FiniteElement  # type: ignore
from dolfin import Form  # type: ignore
from dolfin import Function  # type: ignore
from dolfin import FunctionSpace  # type: ignore
from dolfin import IntervalMesh  # type: ignore
from dolfin import lhs  # type: ignore
from dolfin import LogLevel  # type: ignore
from dolfin import lt  # type: ignore
from dolfin import MixedElement  # type: ignore
from dolfin import project  # type: ignore
from dolfin import rhs  # type: ignore
from dolfin import set_log_level  # type: ignore
from dolfin import solve  # type: ignore
from dolfin import split  # type: ignore
from dolfin import TestFunctions  # type: ignore
from dolfin import TrialFunctions  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore
from ufl import tanh  # type: ignore

from pescoid_modelling.utils.config import SimulationParams
from pescoid_modelling.utils.constants import ETA
from pescoid_modelling.utils.constants import INITIAL_AMPLITUDE
from pescoid_modelling.utils.constants import LEADING_EDGE_THRESHOLD
from pescoid_modelling.utils.constants import LENGTH_SCALE
from pescoid_modelling.utils.constants import RHO_GATE_CENTER
from pescoid_modelling.utils.constants import RHO_GATE_WIDTH
from pescoid_modelling.utils.constants import SNAPSHOT_EVERY_N_STEPS
from pescoid_modelling.utils.constants import TRANSITION_WIDTH
from pescoid_modelling.utils.simulation_logger import SimulationLogger


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

    def __init__(
        self,
        parameters: SimulationParams,
        work_dir: str | Path,
    ) -> None:
        """Initialize the simulator class."""
        self.params = parameters
        self.aborted: bool = False
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        set_log_level(LogLevel.ERROR)
        self._set_simulation_time()
        self._initialize_constants()

        self._initial_radius: float | None = None

    @property
    def initial_radius(self) -> float:
        """Leading-edge radius of the *initial* density profile (cached)."""
        if self._initial_radius is None:
            rho_fn, _, _ = self._previous_state.split()
            rho_vals = rho_fn.compute_vertex_values(self._mesh)
            x_coords = self._mesh.coordinates().flatten()

            edge_x, _ = self._compute_leading_edge(rho_vals, x_coords)
            self._initial_radius = edge_x
        return self._initial_radius

    def _set_simulation_time(
        self, total_hours: float = 12.0, minutes_per_generation: float = 30.0
    ) -> None:
        """Set the simulation time based on the number of generations."""
        self.final_time = (total_hours * 60.0) / minutes_per_generation
        self.time_step = self.params.delta_t
        self.num_steps = int(round(self.final_time / self.time_step))

    def _initialize_constants(self) -> None:
        """Initialize FEniCS Constant objects for all scalar parameters."""
        # Simulation parameters
        self._dt_const = Constant(self.time_step)
        self._diffusivity_const = Constant(self.params.diffusivity)
        self._m_diffusivity_const = Constant(self.params.m_diffusivity)
        self._flow_const = Constant(self.params.flow)
        self._tau_m_const = Constant(self.params.tau_m)
        self._activity_const = Constant(self.params.activity)
        self._beta_const = Constant(self.params.beta)
        self._r_const = Constant(self.params.r)
        self._sigma_c_const = Constant(self.params.sigma_c)
        self._gamma_const = Constant(self.params.gamma)
        self._rho_sensitivity_const = Constant(self.params.rho_sensitivity)
        self._m_sensitivity_const = Constant(self.params.m_sensitivity)

        # Imported constants
        self._rho_gate_center_const = Constant(RHO_GATE_CENTER)
        self._rho_gate_width_const = Constant(RHO_GATE_WIDTH)
        self._eta_const = Constant(ETA)

        # Utility constants
        self._one_const = Constant(1.0)
        self._half_const = Constant(0.5)

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
        self._get_half_domain_idx()
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
    def _get_half_domain_idx(self) -> int:
        """Get the index at the center of the mesh"""
        domain_length = getattr(self.params, "domain_length", 10.0)
        mesh_spacing = getattr(self.params, "dx_interval", 5e-3)
        self.half_domain_idx = int(np.round((domain_length/2)/mesh_spacing))

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

        element1 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        element2 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        element3 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        mixed_element = MixedElement([element1, element2, element3])

        self._mixed_function_space = FunctionSpace(self._mesh, mixed_element)
        self._previous_state = Function(self._mixed_function_space)
        self._current_state = Function(self._mixed_function_space)

    def _set_initial_conditions(self) -> None:
        """Set initial conditions for density, mesoderm, and velocity."""
        if self._previous_state is None:
            raise RuntimeError(
                "Function spaces must be set up before setting initial conditions."
            )

        # Define initial condition expressions
        density_expression = (
            f"({INITIAL_AMPLITUDE}/2) * ((-tanh((pow(x[0], 2) - {LENGTH_SCALE})/{TRANSITION_WIDTH})) + 1)"
            "* pow(0.5, x[0]*x[0])"
        )
        density_initial_condition = Expression(density_expression, degree=2)
        mesoderm_initial_condition = Expression("-1.0", degree=1)
        velocity_initial_condition = Expression("0.0", degree=1)

        # Create individual function spaces
        scalar_space = FunctionSpace(self._mesh, "CG", 1)

        # Project expressions onto individual spaces
        density_func = project(density_initial_condition, scalar_space)
        mesoderm_func = project(mesoderm_initial_condition, scalar_space)
        velocity_func = project(velocity_initial_condition, scalar_space)

        # Assign to mixed space
        assign(self._previous_state.sub(0), density_func)
        assign(self._previous_state.sub(1), mesoderm_func)
        assign(self._previous_state.sub(2), velocity_func)

    def _build_variational_forms(self) -> None:
        """Build the variational forms for the PDEs."""
        if self._previous_state is None or self._current_state is None:
            raise RuntimeError("Function states must be set up before building forms.")

        self._forms = self._formulate_variational_problem()

    def _formulate_variational_problem(self) -> Tuple[Expression, Expression]:
        """Build the variational forms for the PDEs."""
        rho_prev, m_prev, u_prev = split(self._previous_state)  # type: ignore
        rho, m, u = TrialFunctions(self._mixed_function_space)  # type: ignore
        test_rho, test_m, test_u = TestFunctions(self._mixed_function_space)  # type: ignore

        # Formulate residuals
        F_rho = self._formulate_density_equation(
            rho=rho,  # type: ignore
            rho_prev=rho_prev,  # type: ignore
            u_prev=u_prev,
            test_rho=test_rho,  # type: ignore
        )
        F_m = self._formulate_mesoderm_equation(
            m=m,
            m_prev=m_prev,
            rho_prev=rho_prev,  # type: ignore
            u_prev=u_prev,
            test_m=test_m,
        )
        F_u = self._formulate_velocity_equation(
            u=u,
            rho_prev=rho_prev,  # type: ignore
            m_prev=m_prev,
            test_u=test_u,
        )

        total_form = F_rho + F_m + F_u
        return lhs(total_form), rhs(total_form)  # type: ignore

    def _formulate_density_equation(
        self,
        rho: Function,
        rho_prev: Function,
        u_prev: Function,
        test_rho: Function,
    ) -> Form:
        """Formulate the variational form for the density equation (tissue
        growth). Equation is of the following form:

            d rho / dt = Delta * d^2 rho / dx^2 - Flow * u_prev * d rho_prev /
            dx + rho_prev * (1 - rho_prev)
        """
        temporal = (rho - rho_prev) * test_rho * dx  # type: ignore
        diffusion = (
            self._dt_const * self._diffusivity_const * rho.dx(0) * test_rho.dx(0) * dx  # type: ignore
        )
        advection = (
            -self._dt_const * self._flow_const * u_prev * rho_prev * test_rho.dx(0) * dx  # type: ignore
        )
        reaction = (
            -self._dt_const * rho_prev * (self._one_const - rho_prev) * test_rho * dx  # type: ignore
        )

        # Complete form
        return temporal + diffusion + advection + reaction

    def _formulate_mesoderm_equation(
        self,
        m: Function,
        m_prev: Function,
        rho_prev: Function,
        u_prev: Function,
        test_m: Function,
    ) -> Form:
        """Formulate the variational form for mesoderm differentiation. The
        complete equation is:

        d m / dt = (1/tau_m) * m_prev * (m_prev + 1) * (1 - m_prev)
        + (1/tau_m) * R * [feedback term]
        + Delta * d^2 m / dx^2
        - Flow * u_prev * d m_prev / dx
        """
        feedback_mode = getattr(self.params, "feedback_mode", "strain_rate")

        # Build term by term
        temporal = (m - m_prev) * test_m * dx  # type: ignore
        common_decay = (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * m_prev
            * (m_prev + self._one_const)  # type: ignore
            * (self._one_const - m_prev)  # type: ignore
            * test_m
            * dx
        )

        if feedback_mode == "strain_rate":
            feedback = self._formulate_strain_rate_feedback(u_prev, rho_prev, test_m)
        else:  # active_stress
            feedback = self._formulate_active_stress_feedback(rho_prev, m_prev, test_m)

        diffusion = self._dt_const * self._m_diffusivity_const * m.dx(0) * test_m.dx(0) * dx  # type: ignore
        advection = self._dt_const * self._flow_const * u_prev * m_prev.dx(0) * test_m * dx  # type: ignore

        # Complete residual
        return temporal - common_decay - feedback + diffusion + advection

    def _formulate_active_stress_feedback(
        self,
        rho_prev: Function,
        m_prev: Function,
        test_m: Function,
    ) -> Form:
        """Formulate the active-stress feedback term for mesoderm differentiation:

        (1/tau_m) * R * (sigma_a - Sigma_c)

        where

        sigma_a = density_prev * Activity  # stress_term
        * (density_prev/(1 + self._rho_sensitivity_const * density_prev^2))
        * (1 + Beta * (
            (tanh((mesoderm_prev - self._m_sensitivity_const)/self._m_sensitivity_const) + 1)/2)
        )
        """
        # active-stress cue, sigma_a
        cue = (
            rho_prev  # type: ignore
            * self._activity_const
            * (
                rho_prev
                / (self._one_const + self._rho_sensitivity_const * rho_prev * rho_prev)  # type: ignore
            )
            * (
                self._one_const
                + self._beta_const
                * (
                    (
                        tanh(
                            (m_prev - self._m_sensitivity_const)  # type: ignore
                            / self._m_sensitivity_const
                        )
                        + self._one_const
                    )
                    / Constant(2.0)
                )
            )
        )

        # full feedback term
        return (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * self._r_const
            * (cue - self._sigma_c_const)
            * test_m
            * dx
        )

    def _formulate_strain_rate_feedback(
        self,
        u_prev: Function,
        rho_prev: Function,
        test_m: Function,
    ) -> Form:
        """Formulate the active stress feedback term for mesoderm
        differentiation:

        (1/τₘ) · R · [ (-η ∂x v_prev) - sigma_c ]

        R remains positive and the sign flip is built into the cue.
        """
        rho_gate = self._half_const * (
            tanh((rho_prev - self._rho_gate_center_const) / self._rho_gate_width_const)  # type: ignore
            + self._one_const
        )

        cue = -rho_gate * u_prev.dx(0)  # type: ignore

        return (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * self._r_const
            * (cue - self._sigma_c_const)  # type: ignore
            * test_m
            * dx
        )

    def _formulate_velocity_equation(
        self,
        u: Function,
        rho_prev: Function,
        m_prev: Function,
        test_u: Function,
    ) -> Form:
        """Formulate the variational form for tissue velocity (force balance
        equation):

        rho_gate * Gamma * u - rho_gate * d^2 u / dx^2 = div(active_stress)
        """
        rho_gate = self._half_const * (
            tanh((rho_prev - self._rho_gate_center_const) / self._rho_gate_width_const)  # type: ignore
            + self._one_const
        )

        # Stress divergence term
        active_stress_div = self._calculate_stress_divergence(rho_prev, m_prev)

        # Build equation terms
        friction = rho_gate * self._gamma_const * u * test_u * dx
        viscosity = rho_gate * u.dx(0) * test_u.dx(0) * dx  # type: ignore
        force = active_stress_div * test_u * dx  # type: ignore

        # complete form
        return friction + viscosity - force

    def _calculate_stress_divergence(
        self, rho_prev: Function, m_prev: Function
    ) -> Function:
        """Calculate the divergence of the active stress tensor:

        d/dx [density_prev
            * (Activity
            * (density_prev/(1 + self._rho_sensitivity_const * density_prev^2))
            * (
                1
                + Beta
                * ((tanh((mesoderm_prev - self._m_sensitivity_const)/self._m_sensitivity_const) + 1)/2)
            ) - 1)]
        """
        active_stress = rho_prev * (
            self._activity_const  # type: ignore
            * (
                rho_prev
                / (self._one_const + self._rho_sensitivity_const * rho_prev * rho_prev)  # type: ignore
            )
            * (
                self._one_const
                + self._beta_const
                * (
                    (
                        tanh(
                            (m_prev - self._m_sensitivity_const)  # type: ignore
                            / self._m_sensitivity_const
                        )
                        + self._one_const
                    )
                    / Constant(2.0)
                )
            )
            - self._one_const
        )

        # divergence
        return active_stress.dx(0)

    def _advance(self, step_idx: int) -> bool:
        """Advance the simulation by one time step."""
        lhs_form, rhs_form = self._forms
        solution = Function(self._mixed_function_space)
        solve(lhs_form == rhs_form, solution)

        # Extract components
        rho_new, m_new, _ = solution.split(deepcopy=True)
        if self._check_for_errant_values(rho_new) or self._check_for_errant_values(
            m_new
        ):
            return False

        # Correct density
        rho_new = self._ensure_non_negative(rho_new)
        assign(solution.sub(0), rho_new)

        self._previous_state.assign(solution)
        self._current_state.assign(self._previous_state)

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
        rho_projected = project(rho_fn, scalar_space)
        m_projected = project(m_fn, scalar_space)

        rho_vals = rho_projected.vector().get_local().astype(float)
        m_vals = m_projected.vector().get_local().astype(float)

        stress_vals = (
            rho_vals
            * float(self.params.activity)
            * (
                rho_vals
                / (1.0 + float(self._rho_sensitivity_const) * rho_vals * rho_vals)
            )
            * (
                1.0
                + float(self.params.beta)
                * (
                    (
                        np.tanh(
                            (m_vals - float(self._m_sensitivity_const))
                            / float(self._m_sensitivity_const)
                        )
                        + 1.0
                    )
                    / 2.0
                )
            )
        )

        return stress_vals

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

    def _compute_mesoderm_fraction(
            self, m_vals: np.array, edge_idx: float
    ) -> float:
        '''Calculate the fraction of tissue that is expressing mesoderm
        '''
        
        m_pos = []
        m_neg = []
        m_frac = []
        i_range = None

        start_index = self.half_domain_idx
        if edge_idx is not None:
            i_range = range(start_index,edge_idx)
            for i in i_range:
                if m_vals[i] > 0:
                    m_pos.append(m_vals[i])  # Use append to add elements to the list
                else:
                    m_neg.append(m_vals[i])
            m_frac = len(m_pos) / (len(m_neg)+len(m_pos))
        return m_frac
        
    def _compute_mesoderm_average(
            self, m_vals: np.array, edge_idx: int
    ) -> float:
        """ Takes average mesoderm values from everywhere within the tissue"""
        meso_all = m_vals[self.half_domain_idx:edge_idx+1]
        meso_avg = meso_all.mean()
        return meso_avg

    def _radius_norm(self, edge_x: float) -> float:
        """Calculate the normalized area based on the leading edge position.

        Returns edge_x/r0.
        """
        return (edge_x / self.initial_radius) if self.initial_radius else float("NaN")

    def _log_state(self, step_idx: int, current_time: float) -> None:
        """Calculate and log the simulation state."""
        rho_fn, m_fn, u_fn = self._current_state.split()
        rho_vals = rho_fn.compute_vertex_values(self._mesh)
        m_vals = m_fn.compute_vertex_values(self._mesh)
        m_center = m_vals[self.half_domain_idx]
        u_vals = u_fn.compute_vertex_values(self._mesh)
        x_coords = self._mesh.coordinates().flatten()
        max_m = m_vals.max()

        edge_x, edge_idx = self._compute_leading_edge(rho_vals, x_coords)
        meso_frac = self._compute_mesoderm_fraction(m_vals, edge_idx)
        m_avg = self._compute_mesoderm_average(m_vals, edge_idx)
        stress_vals = self._compute_stress(rho_fn, m_fn)
        radius_star = self._radius_norm(edge_x)

        self.logger.log_snapshot(
            step_idx=step_idx,
            current_time=current_time,
            rho_vals=rho_vals,
            m_vals=m_vals,
            m_center = m_center,
            m_avg= m_avg,
            u_vals=u_vals,
            stress_vals=stress_vals,
            edge_x=edge_x,
            edge_idx=edge_idx,
            meso_frac=meso_frac,
            max_m=max_m,
            radius_star=radius_star,
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
