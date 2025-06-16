"""Pescoid fluid-dynamics simulation."""

from pathlib import Path
from typing import Dict, Tuple

from dolfin import assign  # type: ignore
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
from dolfin import MixedElement  # type: ignore
from dolfin import project  # type: ignore
from dolfin import rhs  # type: ignore
from dolfin import set_log_level  # type: ignore
from dolfin import solve  # type: ignore
from dolfin import split  # type: ignore
from dolfin import TestFunctions  # type: ignore
from dolfin import TrialFunctions  # type: ignore
import numpy as np
from numpy.linalg import norm
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
        corrected_pressure: bool = False,
    ) -> None:
        """Initialize the simulator class."""
        self.params = parameters
        self.corrected_pressure = corrected_pressure
        self.aborted: bool = False

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        set_log_level(LogLevel.ERROR)
        self._set_simulation_time()
        self._initialize_constants()

        self._initial_radius: float | None = None
        self._half_domain_idx: int | None = None

    @property
    def initial_radius(self) -> float:
        """Leading-edge radius of the *initial* density profile."""
        if self._initial_radius is None:
            rho_fn, _, _, _ = self._previous_state.split()
            rho_vals = rho_fn.compute_vertex_values(self._mesh)
            x_coords = self._mesh.coordinates().flatten()

            edge_x, _ = self._compute_leading_edge(rho_vals, x_coords)
            self._initial_radius = edge_x
        return self._initial_radius

    @property
    def half_domain_idx(self) -> int:
        """Index at the center of the mesh."""
        if self._half_domain_idx is None:
            domain_length = getattr(self.params, "domain_length", 10.0)
            mesh_spacing = getattr(self.params, "dx_interval", 5e-3)
            self._half_domain_idx = int(np.round((domain_length / 2) / mesh_spacing))
        return self._half_domain_idx

    @property
    def results(self) -> Dict[str, np.ndarray]:
        """Return simulation results as a dictionary."""
        result_dict = self.logger.to_dict()
        result_dict["aborted"] = np.asarray([self.aborted])
        result_dict["dt"] = np.asarray([self.time_step])
        return result_dict

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
        self._c_diffusivity_const = Constant(self.params.c_diffusivity)
        self._morphogen_decay_const = Constant(self.params.morphogen_decay)
        self._gaussian_width_const = Constant(self.params.gaussian_width)
        self._morphogen_feedback_const = Constant(self.params.morphogen_feedback)

        # Imported constants
        self._rho_gate_center_const = Constant(RHO_GATE_CENTER)
        self._rho_gate_width_const = Constant(RHO_GATE_WIDTH)
        self._eta_const = Constant(ETA)

        # Utility constants
        self._half_const = Constant(0.5)
        self._one_const = Constant(1.0)
        self._two_const = Constant(2.0)

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

        element1 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        element2 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        element3 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        element4 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
        mixed_element = MixedElement([element1, element2, element3, element4])

        self._mixed_function_space = FunctionSpace(self._mesh, mixed_element)
        self._previous_state = Function(self._mixed_function_space)
        self._current_state = Function(self._mixed_function_space)

    def _set_initial_conditions(self) -> None:
        """Set initial conditions for density, mesoderm, velocity, and
        morphogen.
        """
        if self._previous_state is None:
            raise RuntimeError(
                "Function spaces must be set up before setting initial conditions."
            )

        # Initial conditions
        density_expression = (
            f"({INITIAL_AMPLITUDE}/2) * ((-tanh((pow(x[0], 2) - {LENGTH_SCALE})/{TRANSITION_WIDTH})) + 1)"
            "* pow(0.5, x[0]*x[0])"
        )
        density_initial_condition = Expression(density_expression, degree=2)
        mesoderm_initial_condition = Expression("-1.0", degree=1)
        velocity_initial_condition = Expression("0.0", degree=1)
        morphogen_initial_condition = Expression("0.0", degree=1)

        # Create function spaces
        scalar_space = FunctionSpace(self._mesh, "CG", 1)

        # Project expressions onto function spaces
        density_func = project(density_initial_condition, scalar_space)
        mesoderm_func = project(mesoderm_initial_condition, scalar_space)
        velocity_func = project(velocity_initial_condition, scalar_space)
        morphogen_func = project(morphogen_initial_condition, scalar_space)

        # Assign to mixed space
        assign(self._previous_state.sub(0), density_func)
        assign(self._previous_state.sub(1), mesoderm_func)
        assign(self._previous_state.sub(2), velocity_func)
        assign(self._previous_state.sub(3), morphogen_func)

    def _build_variational_forms(self) -> None:
        """Build the variational forms for the PDEs."""
        if self._previous_state is None or self._current_state is None:
            raise RuntimeError("Function states must be set up before building forms.")

        self._forms = self._formulate_variational_problem()

    def _formulate_variational_problem(self) -> Tuple[Expression, Expression]:
        """Build the variational forms for the PDEs."""
        rho_prev, m_prev, u_prev, c_prev = split(self._previous_state)  # type: ignore
        rho, m, u, c = TrialFunctions(self._mixed_function_space)  # type: ignore
        test_rho, test_m, test_u, test_c = TestFunctions(self._mixed_function_space)  # type: ignore

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
            c_prev=c_prev,
            test_m=test_m,
        )

        if self.corrected_pressure:
            F_u = self._formulate_pressure_corrected_velocity_equation(
                u=u,
                rho_prev=rho_prev,  # type: ignore
                m_prev=m_prev,
                test_u=test_u,
            )
        else:
            F_u = self._formulate_velocity_equation(
                u=u,
                rho_prev=rho_prev,  # type: ignore
                m_prev=m_prev,
                test_u=test_u,
            )

        F_c = self._formulate_morphogen_equation(
            c=c,
            c_prev=c_prev,
            u_prev=u_prev,
            test_c=test_c,
        )

        total_form = F_rho + F_m + F_u + F_c
        return lhs(total_form), rhs(total_form)  # type: ignore

    def _formulate_density_equation(
        self,
        rho: Function,
        rho_prev: Function,
        u_prev: Function,
        test_rho: Function,
    ) -> Form:
        """Formulate the variational form for the density equation (tissue
        growth).

        Strong form:
          ∂ρ/∂t  +  ∂x(F * u * ρ)  =  δ * ∂²ρ/∂x² - ρ * (1 - ρ)

        Weak form:
          (ρ^{n+1} - ρⁿ) * φ * dx
          + Δt * δ * ∂xρ^{n+1} * ∂xφ * dx
          - Δt * F * uⁿ * ρⁿ * ∂xφ * dx
          - Δt * ρⁿ * (1 - ρⁿ) * φ * dx = 0
        """
        # ∂ρ/∂t = (ρ^{n+1} - ρⁿ) * φ * dx
        temporal = (rho - rho_prev) * test_rho * dx  # type: ignore

        # δ * ∂²ρ/∂x² = + Δt * δ * ∂xρ^{n+1} * ∂xφ * dx
        diffusion = (
            self._dt_const * self._diffusivity_const * rho.dx(0) * test_rho.dx(0) * dx  # type: ignore
        )

        # ∂x(F * u * ρ) = - Δt * F * uⁿ * ρⁿ * ∂xφ * dx
        advection = (
            -self._dt_const * self._flow_const * u_prev * rho_prev * test_rho.dx(0) * dx  # type: ignore
        )

        # -ρ * (1 - ρ) = - Δt * ρⁿ * (1 - ρⁿ) * φ * dx
        reaction = (
            -self._dt_const * rho_prev * (self._one_const - rho_prev) * test_rho * dx  # type: ignore
        )

        return temporal + diffusion + advection + reaction

    def _formulate_mesoderm_equation(
        self,
        m: Function,
        m_prev: Function,
        rho_prev: Function,
        u_prev: Function,
        c_prev: Function,
        test_m: Function,
    ) -> Form:
        """Formulate the variational form for mesoderm differentiation.

        Strong form:
          ∂m/∂t = Dₘ * ∂²m/∂x² + F * uⁿ * ∂mⁿ/∂x - (1/τₘ) * mⁿ * (mⁿ+1) * (1-mⁿ)
          - (1/τₘ) * R * [mech_cueⁿ] - (1/τₘ) * R_c * cⁿ

        Weak form:
          (m^{n+1} - mⁿ) * φ * dx
          + Δt * Dₘ * ∂xm^{n+1} * ∂xφ * dx
          + Δt * F * uⁿ * ∂xmⁿ * φ * dx
          - Δt * (1/τₘ) * mⁿ * (mⁿ+1) * (1-mⁿ) * φ * dx
          - Δt * (1/τₘ) * R * [mech_cueⁿ] * φ * dx
          - Δt * (1/τₘ) * R_c * cⁿ * φ * dx = 0
        """
        # ∂m/∂t = (m^{n+1} - mⁿ) * φ * dx
        temporal = (m - m_prev) * test_m * dx  # type: ignore

        # -(1/τₘ) * mⁿ * (mⁿ+1) * (1-mⁿ) = -Δt * (1/τₘ) * mⁿ * (mⁿ+1) * (1-mⁿ) * φ * dx
        common_decay = (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * m_prev
            * (m_prev + self._one_const)  # type: ignore
            * (self._one_const - m_prev)  # type: ignore
            * test_m
            * dx
        )

        # -(1/τₘ) * R * [mech_cueⁿ] = -Δt * (1/τₘ) * R * [mech_cueⁿ] * φ * dx
        mechanical_feedback = self._formulate_mechanical_feedback(
            rho_prev=rho_prev,
            m_prev=m_prev,
            test_m=test_m,
            cue=getattr(self.params, "feedback_mode"),
        )

        # -(1/τₘ) * R_c * cⁿ = -Δt * (1/τₘ) * R_c * cⁿ * φ * dx
        chemical_feedback = (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * self._morphogen_feedback_const
            * c_prev
            * test_m
            * dx
        )

        # Dₘ * ∂²m/∂x² = +Δt * Dₘ * ∂xm^{n+1} * ∂xφ * dx
        diffusion = self._dt_const * self._m_diffusivity_const * m.dx(0) * test_m.dx(0) * dx  # type: ignore

        # F * uⁿ * ∂mⁿ/∂x = +Δt * F * uⁿ * ∂xmⁿ * φ * dx
        advection = self._dt_const * self._flow_const * u_prev * m_prev.dx(0) * test_m * dx  # type: ignore
        # advection = self._dt_const * self._flow_const * u_prev * m.dx(0) * test_m * dx  # type: ignore

        return (
            temporal
            - common_decay
            - mechanical_feedback
            - chemical_feedback
            + diffusion
            + advection
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

        Strong form:
          ρ_gate * Γ * u^{n+1} - ρ_gate * ∂²u/∂x² = ∂σⁿ/∂x

        Weak form:
          ρ_gate * Γ * u^{n+1} * φ * dx
          + ρ_gate * ∂xu^{n+1} * ∂xφ * dx
          - (∂σⁿ/∂x) * φ * dx = 0
        """
        # ρ_gate = (tanh((ρ - ρ₀)/w) + 1)/2
        rho_gate = self._half_const * (
            tanh((rho_prev - self._rho_gate_center_const) / self._rho_gate_width_const)  # type: ignore
            + self._one_const
        )

        # ∂σⁿ/∂x
        active_stress_div = self._calculate_stress_divergence(rho_prev, m_prev)

        # ρ_gate * Γ * u^{n+1} = ρ_gate * Γ * u^{n+1} * φ * dx
        friction = rho_gate * self._gamma_const * u * test_u * dx

        # -ρ_gate * ∂u/∂x * ∂test_u/∂x = +ρ_gate * ∂xu^{n+1} * ∂xφ * dx
        viscosity = rho_gate * u.dx(0) * test_u.dx(0) * dx  # type: ignore

        # (∂σⁿ/∂x) * φ * dx
        force = active_stress_div * test_u * dx  # type: ignore

        return friction + viscosity - force

    def _formulate_pressure_corrected_velocity_equation(
        self,
        u: Function,
        rho_prev: Function,
        m_prev: Function,
        test_u: Function,
    ) -> Form:
        """Formulate the variational form for tissue velocity
        (pressure-corrected active-polar fluid model):

        Strong form:
          η * ∂²u/∂x² + ∂(σᵃ - P)/∂x = 0

        Weak form:
          η * ∂u/∂x * ∂φ/∂x * dx
          - (σᵃ - P) * ∂φ/∂x * dx = 0
        """
        # σᵃ
        active_stress = self._calculate_active_stress_field(rho_prev, m_prev)

        # vᵣ(R)/R
        boundary_term_const = self._calculate_boundary_velocity_term_from_state()

        # P = σᵃ - η * ( vᵣ(R)/R - ∇·v )
        pressure = self._calculate_pressure(u, active_stress, boundary_term_const)

        # σᵃ - P
        corrected_stress = active_stress - pressure  # type: ignore

        # η * ∂²u/∂x² = η * ∂u/∂x * ∂φ/∂x * dx
        viscous_term = self._eta_const * u.dx(0) * test_u.dx(0) * dx  # type: ignore

        # ∂x(σᵃ - P) = - (σᵃ - P) * ∂φ/∂x * dx
        stress_term = -corrected_stress * test_u.dx(0) * dx  # type: ignore

        return viscous_term + stress_term

    def _formulate_morphogen_equation(
        self,
        c: Function,
        c_prev: Function,
        u_prev: Function,
        test_c: Function,
    ) -> Form:
        """Formulate the variational form for morphogen concentration.

        Strong form:
          ∂c/∂t + ∂x(v * c) = s(x) - k * c + D_c * ∂²c/∂x²

        Weak form:
          (c^{n+1} - cⁿ) * φ * dx
          - Δt * vⁿ * cⁿ * ∂φ/∂x * dx
          + Δt * k * cⁿ * φ * dx
          + Δt * D_c * ∂c^{n+1}/∂x * ∂φ/∂x * dx
          - Δt * s(x) * φ * dx = 0
        """
        # s(x) = (1/(σ√(2π))) * exp(-x²/(2σ²))
        sigma = float(self.params.gaussian_width)
        normalization = 1.0 / (sigma * np.sqrt(2 * np.pi))
        gaussian_expr = f"{normalization} * exp(-pow(x[0], 2) / (2 * pow({sigma}, 2)))"
        gaussian_source = Expression(gaussian_expr, degree=2)

        # ∂c/∂t = (c^{n+1} - cⁿ) * φ * dx
        temporal = (c - c_prev) * test_c * dx  # type: ignore

        # ∂x(v * c) = - Δt * vⁿ * cⁿ * ∂φ/∂x * dx
        advection = -self._dt_const * u_prev * c_prev * test_c.dx(0) * dx  # type: ignore

        # D_c * ∂²c/∂x² = + Δt * D_c * ∂c^{n+1}/∂x * ∂φ/∂x * dx
        diffusion = (
            self._dt_const * self._c_diffusivity_const * c.dx(0) * test_c.dx(0) * dx  # type: ignore
        )

        # -k * c = + Δt * k * cⁿ * φ * dx
        decay = self._dt_const * self._morphogen_decay_const * c_prev * test_c * dx  # type: ignore

        # +s(x) = - Δt * s(x) * φ * dx
        source = -self._dt_const * gaussian_source * test_c * dx  # type: ignore

        return temporal + advection + diffusion + decay + source

    def _calculate_active_stress_field(
        self, rho_prev: Function, m_prev: Function
    ) -> Function:
        """Calculate the active stress field:

        σᵃ = ρ * [A * f(ρ, m) - 1]
        """
        # ρ/(1 + α*ρ²)
        density_saturation = rho_prev / (
            self._one_const + self._rho_sensitivity_const * rho_prev * rho_prev  # type: ignore
        )

        # (tanh((m - m₀)/m₀) + 1)/2
        mesoderm_sigmoid = (
            tanh((m_prev - self._m_sensitivity_const) / self._m_sensitivity_const)  # type: ignore
            + self._one_const
        ) / self._two_const

        # 1 + β * sigmoid(m)
        mesoderm_enhancement = self._one_const + self._beta_const * mesoderm_sigmoid

        # A * [ρ/(1 + α*ρ²)] * [1 + β * sigmoid(m)]
        active_stress_factor = (
            self._activity_const * density_saturation * mesoderm_enhancement
        )

        # σᵃ = ρ * (active_stress_factor)
        return rho_prev * (active_stress_factor)

    def _calculate_strain_rate(self, rho_prev: Function, m_prev: Function) -> Function:
        """Calculate the strain rate based on the active stress field:

        -ρ_gate * ∂u/∂x
        """
        # ρ_gate = (tanh((ρ - ρ₀)/w) + 1)/2
        rho_gate = self._half_const * (
            tanh((rho_prev - self._rho_gate_center_const) / self._rho_gate_width_const)  # type: ignore
            + self._one_const
        )

        # -ρ_gate * ∂u/∂x
        return -rho_gate * u_prev.dx(0)  # type: ignore

    def _formulate_mechanical_feedback(
        self,
        rho_prev: Function,
        m_prev: Function,
        test_m: Function,
        cue: "str",
    ) -> Form:
        """Formulate the physical mechanical feedback term.

        Active stress:
            (1/τₘ) * R * (cue - σc)

        Strain rate:
            (1/τₘ) * R * (-ρ_gate * ∂u/∂x - σc)
        """
        if cue == "active_stress":
            mechanical_cue = self._calculate_active_stress_field(rho_prev, m_prev)
        elif cue == "strain_rate":
            mechanical_cue = self._calculate_strain_rate(rho_prev, m_prev)

        return (
            self._dt_const
            * (self._one_const / self._tau_m_const)  # type: ignore
            * self._r_const
            * (mechanical_cue - self._sigma_c_const)  # type: ignore
            * test_m
            * dx
        )

    def _calculate_stress_divergence(
        self, rho_prev: Function, m_prev: Function
    ) -> Function:
        """Calculate the divergence of the active stress tensor:

        ∂σ/∂x where σ = ρ * [A * f(ρ,m) - 1]
        """
        # ρ/(1 + α*ρ²)
        density_saturation = rho_prev / (
            self._one_const + self._rho_sensitivity_const * rho_prev * rho_prev  # type: ignore
        )

        # sigmoid(m) = (tanh((m - m₀)/m₀) + 1)/2
        mesoderm_sigmoid = (
            tanh((m_prev - self._m_sensitivity_const) / self._m_sensitivity_const)  # type: ignore
            + self._one_const
        ) / self._two_const

        # 1 + β * sigmoid(m)
        mesoderm_enhancement = self._one_const + self._beta_const * mesoderm_sigmoid

        # A * [ρ/(1 + α*ρ²)] * [1 + β * sigmoid(m)]
        active_stress_factor = (
            self._activity_const * density_saturation * mesoderm_enhancement
        )

        # σ = ρ * (A * f(ρ,m) - 1)
        active_stress = rho_prev * (active_stress_factor - self._one_const)

        # ∂σ/∂x
        return active_stress.dx(0)

    def _calculate_boundary_velocity_term_from_state(self) -> Constant:
        """Calculate the boundary velocity term using the previous state:

        vᵣ(R)/R
        """
        _, _, u_fn, _ = self._previous_state.split()

        # Get R from mesh coordinates
        mesh_coords = self._mesh.coordinates()
        domain_half_length = (mesh_coords.max() - mesh_coords.min()) / 2.0

        # Mesh vertex values
        u_vals = u_fn.compute_vertex_values(self._mesh)

        # Get the average velocity at the boundary (R)
        boundary_idxs = np.where(np.isclose(mesh_coords, mesh_coords.max(), atol=1e-8))[
            0
        ]
        u_at_boundary = u_vals[boundary_idxs].mean()

        return Constant(u_at_boundary / domain_half_length)

    def _calculate_pressure(
        self, u: Function, active_stress: Function, boundary_term_const: Constant
    ) -> Function:
        """Calculate pressure using the relation:

        P = σᵃ - η * (vᵣ(R)/R - ∇ · v)
        """
        # ∇ · v = ∂u/∂x (in 1D)
        velocity_divergence = u.dx(0)  # type: ignore

        # P = σᵃ - η * (vᵣ(R)/R - ∇ · v)
        pressure = active_stress - self._eta_const * (
            boundary_term_const - velocity_divergence
        )

        return pressure

    def _advance(self, step_idx: int) -> bool:
        """Advance the simulation by one time step."""
        lhs_form, rhs_form = self._forms
        solution = Function(self._mixed_function_space)
        solve(lhs_form == rhs_form, solution)

        # Split old and new states
        rho_old, m_old, u_old, c_old = self._previous_state.split(deepcopy=True)  # type: ignore
        rho_new, m_new, u_new, c_new = solution.split(deepcopy=True)  # type: ignore

        # Get vector differences
        delta_rho = float(
            norm(rho_new.vector().get_local() - rho_old.vector().get_local())
        )
        delta_m = float(norm(m_new.vector().get_local() - m_old.vector().get_local()))
        delta_u = float(norm(u_new.vector().get_local() - u_old.vector().get_local()))
        delta_c = float(norm(c_new.vector().get_local() - c_old.vector().get_local()))

        self.logger.log_norm(
            step_idx=step_idx,
            rho_norm=delta_rho,
            m_norm=delta_m,
            u_norm=delta_u,
            c_norm=delta_c,
        )

        # Extract components
        rho_new, m_new, u_new, c_new = solution.split(deepcopy=True)
        if (
            self._check_for_errant_values(rho_new)
            or self._check_for_errant_values(m_new)
            or self._check_for_errant_values(c_new)
        ):
            return False

        # Ensure non-negative density and morphogen
        rho_new = self._ensure_non_negative(rho_new)
        c_new = self._ensure_non_negative(c_new)
        assign(solution.sub(0), rho_new)
        assign(solution.sub(3), c_new)

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
        if self.logger._should_log(step_idx):
            self._log_state(step_idx, current_time)

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
        self,
        m_vals: np.ndarray,
        edge_idx: float | None,
    ) -> float:
        """Calculate the fraction of tissue that is expressing mesoderm using a
        smooth activation function.
        """
        if edge_idx is None:
            return 0.0

        tissue_mesoderm = m_vals[self.half_domain_idx : int(edge_idx)]
        if tissue_mesoderm.size == 0:
            return 0.0

        return (tissue_mesoderm > 0.0).mean()

    def _compute_mesoderm_average(self, m_vals: np.ndarray, edge_idx: int) -> float:
        """Get all-tissue mesoderm average."""
        meso_all = m_vals[self.half_domain_idx : edge_idx + 1]
        meso_avg = meso_all.mean()
        return meso_avg

    def _compute_stress(self, rho_arr: np.ndarray, m_arr: np.ndarray) -> np.ndarray:
        """Compute stress of the simulation based on the current state."""
        # ρ/(1 + α ρ²)
        sat = rho_arr / (1.0 + self.params.rho_sensitivity * rho_arr**2)

        # sigmoid(m) = (tanh((m - m₀)/m₀) + 1)/2
        sig = 0.5 * (
            1 + np.tanh((m_arr - self.params.m_sensitivity) / self.params.m_sensitivity)
        )
        # σ = ρ * A * sat * (1 + β sig)
        return rho_arr * self.params.activity * sat * (1 + self.params.beta * sig)

    def _radius_norm(self, edge_x: float) -> float:
        """Calculate the normalized area based on the leading edge position.

        Returns edge_x/r0.
        """
        return (edge_x / self.initial_radius) if self.initial_radius else float("NaN")

    def _log_state(self, step_idx: int, current_time: float) -> None:
        """Calculate and log the simulation state."""
        rho_fn, m_fn, u_fn, c_fn = self._current_state.split()
        rho_vals = rho_fn.compute_vertex_values(self._mesh)
        m_vals = m_fn.compute_vertex_values(self._mesh)
        u_vals = u_fn.compute_vertex_values(self._mesh)
        c_vals = c_fn.compute_vertex_values(self._mesh)

        x_coords = self._mesh.coordinates().flatten()
        edge_x, edge_idx = self._compute_leading_edge(rho_vals, x_coords)
        radius_star = self._radius_norm(edge_x)

        mesoderm_fraction = self._compute_mesoderm_fraction(m_vals, edge_idx)
        m_avg = self._compute_mesoderm_average(m_vals, edge_idx)
        stress_vals = self._compute_stress(rho_vals, m_vals)

        if edge_idx < len(c_vals):
            morphogen_edge = c_vals[edge_idx]
        else:
            morphogen_edge = 0.0
        c_gradient = np.gradient(c_vals, x_coords)

        self.logger.log_snapshot(
            step_idx=step_idx,
            current_time=current_time,
            rho_vals=rho_vals,
            m_vals=m_vals,
            m_center=m_vals[self.half_domain_idx],
            m_avg=m_avg,
            u_vals=u_vals,
            stress_vals=stress_vals,
            edge_x=edge_x,
            edge_idx=edge_idx,
            mesoderm_fraction=mesoderm_fraction,
            max_m=m_vals.max(),
            tissue_size=radius_star,
            c_vals=c_vals,
            c_center=c_vals[self.half_domain_idx],
            max_c=c_vals.max(),
            morphogen_edge=morphogen_edge,
            morphogen_gradient_max=np.max(np.abs(c_gradient)),
            morphogen_gradient_center=c_gradient[self.half_domain_idx],
        )

    def save(self, file: str | Path) -> None:
        out = Path(file).with_suffix(".npz")
        np.savez_compressed(out, allow_pickle=False, **self.results)

    @staticmethod
    def _ensure_non_negative(var_fn: Function) -> Function:
        arr = var_fn.vector().get_local()
        np.maximum(arr, 0.0, out=arr)
        var_fn.vector().set_local(arr)
        var_fn.vector().apply("insert")
        return var_fn
