"""Class to handle logging of simulation data."""

import numpy as np

from pescoid_modelling.utils.constants import SNAPSHOT_EVERY_N_STEPS


class SimulationLogger:
    """Logs simulation data at specified intervals. First allocates arrays for
    each logged variable, then populates eacch array with data at each snapshot.
    Finally, the class trims each array to the number of snapshots taken.

    Examples::
        # Instantiate the logger
        >>> logger = SimulationLogger(
        ...     num_steps=num_steps,
        ...     mesh_size=mesh_size,
        ...     snapshot_interval=snapshot_interval,
        ... )

        # Log a snapshot
        >>> logger.log_snapshot(
        ...     step_idx=step_idx,
        ...     current_time=current_time,
        ...     rho_vals=rho_vals,
        ...     m_vals=m_vals,
        ...     m_center=m_center,
        ...     m_avg=m_avg,
        ...     u_vals=u_vals,
        ...     stress_vals=stress_vals,
        ...     edge_x=edge_x,
        ...     edge_idx=edge_idx,
        ...     mesoderm_fraction=mesoderm_fraction,
        ...     max_m=max_m,
        ...     tissue_size=tissue_size,
        ...     c_vals=c_vals,
        ...     c_center=c_center,
        ...     max_c=max_c,
        ... )
    """

    SNAPSHOT_INTERVAL: int = 1

    def __init__(
        self,
        num_steps: int,
        mesh_size: int,
        snapshot_interval: int = SNAPSHOT_EVERY_N_STEPS,
    ) -> None:
        """Initialize the logger."""
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = num_steps // snapshot_interval + 1
        self.mesh_size = mesh_size
        self._snapshot_count = 0
        self._step_counter = 0

        # 2D arrays
        self.density = np.zeros((self.max_snapshots, mesh_size))
        self.mesoderm = np.zeros((self.max_snapshots, mesh_size))
        self.velocity = np.zeros((self.max_snapshots, mesh_size))
        self.stress = np.zeros((self.max_snapshots, mesh_size))
        self.morphogen = np.zeros((self.max_snapshots, mesh_size))

        # 1D arrays
        self.tissue_size = np.zeros(self.max_snapshots)
        self.meso_mean = np.zeros(self.max_snapshots)
        self.times = np.zeros(self.max_snapshots)
        self.boundary_positions = np.zeros(self.max_snapshots)
        self.boundary_velocity = np.zeros(self.max_snapshots)
        self.mesoderm_fraction = np.zeros(self.max_snapshots)
        self.mesoderm_avg = np.zeros(self.max_snapshots)
        self.mesoderm_center = np.zeros(self.max_snapshots)
        self.max_mesoderm = np.zeros(self.max_snapshots)
        self.c_vals = np.zeros(self.max_snapshots)
        self.c_center = np.zeros(self.max_snapshots)
        self.morphogen_center = np.zeros(self.max_snapshots)
        self.max_morphogen = np.zeros(self.max_snapshots)
        self.morphogen_mean = np.zeros(self.max_snapshots)

        # norms for PDEs
        self.rho_norm = np.zeros(num_steps)
        self.m_norm = np.zeros(num_steps)
        self.u_norm = np.zeros(num_steps)
        self.c_norm = np.zeros(num_steps)

    def _should_log(self, step_idx: int) -> bool:
        """Determines if the current step should be logged."""
        return step_idx % self.snapshot_interval == 0

    def log_norm(
        self,
        step_idx: int,
        rho_norm: float,
        m_norm: float,
        u_norm: float,
        c_norm: float,
    ) -> None:
        """Store L2 norms of each weak form for this step."""
        self.rho_norm[step_idx] = rho_norm
        self.m_norm[step_idx] = m_norm
        self.u_norm[step_idx] = u_norm
        self.c_norm[step_idx] = c_norm
        self._step_counter += 1

    def log_snapshot(
        self,
        step_idx,
        current_time,
        rho_vals,
        m_vals,
        m_center,
        m_avg,
        u_vals,
        stress_vals,
        edge_x,
        edge_idx,
        mesoderm_fraction,
        max_m,
        tissue_size,
        c_vals,
        c_center,
        max_c,
        x_coords=None,
    ):
        """Log a snapshot of simulation data."""
        if not self._should_log(step_idx):
            return

        if self.x_coordinates is None and x_coords is not None:
            self.x_coordinates = x_coords.copy()

        # Scalars
        self.tissue_size[self._snapshot_count] = float(tissue_size)
        self.meso_mean[self._snapshot_count] = float(m_vals.mean())
        self.times[self._snapshot_count] = current_time
        self.boundary_positions[self._snapshot_count] = float(edge_x)
        self.boundary_velocity[self._snapshot_count] = float(u_vals[edge_idx])
        self.mesoderm_fraction[self._snapshot_count] = float(mesoderm_fraction)
        self.mesoderm_center[self._snapshot_count] = m_center
        self.mesoderm_avg[self._snapshot_count] = m_avg
        self.max_mesoderm[self._snapshot_count] = float(max_m)
        self.morphogen_center[self._snapshot_count] = float(c_center)
        self.max_morphogen[self._snapshot_count] = float(max_c)
        self.morphogen_mean[self._snapshot_count] = float(c_vals.mean())

        # 2D arrays
        self.density[self._snapshot_count] = rho_vals
        self.mesoderm[self._snapshot_count] = m_vals
        self.velocity[self._snapshot_count] = u_vals
        self.stress[self._snapshot_count] = stress_vals
        self.morphogen[self._snapshot_count] = c_vals

        self._snapshot_count += 1

    def finalize(self) -> None:
        """Trim unused space in pre-allocated arrays."""
        if self._snapshot_count < self.max_snapshots:
            # Scalars
            self.tissue_size = self.tissue_size[: self._snapshot_count]
            self.meso_mean = self.meso_mean[: self._snapshot_count]
            self.times = self.times[: self._snapshot_count]
            self.boundary_positions = self.boundary_positions[: self._snapshot_count]
            self.boundary_velocity = self.boundary_velocity[: self._snapshot_count]
            self.mesoderm_fraction = self.mesoderm_fraction[: self._snapshot_count]
            self.mesoderm_center = self.mesoderm_center[: self._snapshot_count]
            self.mesoderm_avg = self.mesoderm_avg[: self._snapshot_count]
            self.max_mesoderm = self.max_mesoderm[: self._snapshot_count]
            self.morphogen_center = self.morphogen_center[: self._snapshot_count]
            self.max_morphogen = self.max_morphogen[: self._snapshot_count]
            self.morphogen_mean = self.morphogen_mean[: self._snapshot_count]

            # 2D arrays
            self.density = self.density[: self._snapshot_count]
            self.mesoderm = self.mesoderm[: self._snapshot_count]
            self.velocity = self.velocity[: self._snapshot_count]
            self.stress = self.stress[: self._snapshot_count]
            self.morphogen = self.morphogen[: self._snapshot_count]

    def to_dict(self) -> dict:
        """Convert all logs to a dictionary for saving or analysis."""
        base = {
            # Time and tissue metrics
            "time": self.times,
            "tissue_size": self.tissue_size,
            "boundary_positions": self.boundary_positions,
            "boundary_times": self.times,
            "boundary_velocity": self.boundary_velocity,
            # Mesoderm metrics
            "mesoderm_mean": self.meso_mean,
            "mesoderm_center": self.mesoderm_center,
            "mesoderm_average": self._normalize_mesoderm_data(self.mesoderm_avg),
            "mesoderm_fraction": self.mesoderm_fraction,
            "max_mesoderm": self._normalize_mesoderm_data(self.max_mesoderm),
            # Morphogen metrics
            "morphogen_mean": self.morphogen_mean,
            "morphogen_center": self.morphogen_center,
            "max_morphogen": self.max_morphogen,
            # Spatial metrics
            "density": self.density,
            "mesoderm": self.mesoderm,
            "velocity": self.velocity,
            "stress": self.stress,
            "morphogen": self.morphogen,
            # Coordinates
            "x_coords": self.x_coordinates,
        }

        base.update(
            {
                "rho_norm": self.rho_norm[: self._step_counter],
                "m_norm": self.m_norm[: self._step_counter],
                "u_norm": self.u_norm[: self._step_counter],
                "c_norm": self.c_norm[: self._step_counter],
            }
        )
        return base

    def _normalize_mesoderm_data(self, mesoderm_data: np.ndarray) -> np.ndarray:
        """Normalize mesoderm data to a [0,1] range."""
        if len(mesoderm_data) == 0:
            return mesoderm_data

        m_min = np.min(mesoderm_data)
        m_max = np.max(mesoderm_data)

        # Avoid division by zero if all values are the same
        if m_max == m_min:
            return np.zeros_like(mesoderm_data)

        return (mesoderm_data - m_min) / (m_max - m_min)
