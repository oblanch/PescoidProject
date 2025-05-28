"""Constants for pescoid modelling."""

SNAPSHOT_EVERY_N_STEPS = 10
LEADING_EDGE_THRESHOLD = 0.2
RHO_GATE_CENTER = 0.02
RHO_GATE_WIDTH = 0.01
M_SENSITIVITY = 1e-1
RHO_SENSITIVITY = 0.1

_ORDER = [
    "length_scale",
    "diffusivity",
    "flow",
    "tau_m",
    "gamma",
    "activity",
    "beta",
    "sigma_c",
    "r",
    "rho_sensitivity",
    "m_sensitivity",
]
