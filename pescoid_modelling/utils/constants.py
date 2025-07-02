"""Constants for pescoid modelling."""

# simulation constants
SNAPSHOT_EVERY_N_STEPS: int = 10
LEADING_EDGE_THRESHOLD: float = 0.2
RHO_GATE_CENTER: float = 0.02
RHO_GATE_WIDTH: float = 0.01
ETA: float = 1.0
INITIAL_AMPLITUDE: float = 1.0
TRANSITION_WIDTH: float = 1e-2
LENGTH_SCALE: float = 1.0
RHO_MAX: float = 3.0

# optimization penalties
OPTIM_PENALTY: float = 1e4
JITTER: float = 1e-6

# constants for dynamic loss weighting
EPS: float = 1e-9
EMA_ALPHA: float = 0.2

# constants to help stiff equations
MAX_HALVES: int = 3
MIN_DT: float = 1e-5
