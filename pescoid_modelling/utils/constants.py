"""Constants for pescoid modelling."""

# Simulation constants
SNAPSHOT_EVERY_N_STEPS: int = 10
LEADING_EDGE_THRESHOLD: float = 0.2
RHO_GATE_CENTER: float = 0.02
RHO_GATE_WIDTH: float = 0.01
ETA: float = 1.0
INITIAL_AMPLITUDE: float = 1.0
TRANSITION_WIDTH: float = 1e-2
LENGTH_SCALE: float = 1.0
RHO_MAX: float = 3.0

# Optimization penalties
OPTIM_PENALTY: float = 1e9
JITTER: float = 1e-6

# Constants for dynamic loss weighting
EPS: float = 1e-9
EMA_ALPHA: float = 0.2

# Constants to help stiff equations
MAX_HALVES: int = 3
MIN_DT: float = 1e-5

# Acceptance criteria for model behavior
ALIGN_TOL = 60.0
PEAK_DROP_MIN = 0.10
GROWTH_MIN = 1.10
WALL_TOL = 0.98
MIN_PEAK_TIME = 180.0
SLOPE_THRESHOLD = -5e-4

# Parameter search constraints
ONSET_THRESH = 0.05
ONSET_TIME_SCALE = 30.0
