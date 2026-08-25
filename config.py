"""
EIT B-Spline + FEM Sobolev Pipeline — Authoritative Configuration
=================================================================

ALL other modules must import constants from here. Do not redefine
constants anywhere else in the project.

Parameter Ordering (AUTHORITATIVE — matches polygon_conductivity_bspline.m)
---------------------------------------------------------------------------
theta = [X1, X2, ..., X32, Y1, Y2, ..., Y32]   shape: (64,)

When reshaped to control points:
    control_points[:, 0] = theta[0:32]   (X coordinates of 32 control points)
    control_points[:, 1] = theta[32:64]  (Y coordinates of 32 control points)
    control_points shape: (32, 2)

This matches the MATLAB:
    control_points = [theta(1:n_control), theta(n_control+1:n_theta)];

FEM Domain
----------
The computational domain is a 2D CIRCULAR domain, radius 1.0 m.
The internal B-spline inclusion is contained within this circle.
The outer domain is NEVER square.

Execution Environment
---------------------
- FEM dataset generation: local Windows with EIDORS/Octave
- ML / training / inversion: Google Colab (no EIDORS required)
- Only physics/fem_backend.py requires EIDORS.
"""

import os
from typing import List, Tuple

# ---------------------------------------------------------------------------
# B-SPLINE GEOMETRY CONSTANTS (from polygon_conductivity_bspline.m)
# ---------------------------------------------------------------------------

N_CONTROL: int = 32           # number of B-spline control points
N_THETA: int = 64             # total parameters = 2 * N_CONTROL
N_BOUNDARY_SAMP: int = 1024   # boundary integration samples
MOLLIFIER_EPS: float = 0.05   # mollifier parameter epsilon
MOLLIFIER_EPS2: float = MOLLIFIER_EPS ** 2  # precomputed eps^2
SERIES_THRESHOLD: float = 1.0e-10  # branch threshold for Taylor vs exact
ALPHA: float = 80.0           # tanh sharpness parameter

# ---------------------------------------------------------------------------
# CONDUCTIVITY VALUES (from MATLAB reference)
# ---------------------------------------------------------------------------

SIGMA_BG: float = 1.0         # background conductivity [S/m]
SIGMA_INC: float = 1.0e-4     # inclusion conductivity [S/m]

# ---------------------------------------------------------------------------
# FEM / EIDORS CONFIGURATION (from validated MATLAB reference)
# ---------------------------------------------------------------------------

N_ELEC: int = 16              # number of surface electrodes
N_STIM: int = 120             # stimulation patterns = C(16,2) = 120
N_MEAS: int = 1920            # total measurements = N_STIM * N_ELEC
Z_CONTACT: float = 0.01       # electrode contact impedance [Ohm*m]
ELECTRODE_WIDTH: float = 0.20 # electrode width [rad]
ELECTRODE_RFNUM: int = 10     # EIDORS rfnum parameter
DOMAIN_RADIUS: float = 1.0    # circular FEM domain radius [m]
MESH_MAXSZ: float = 0.05      # FEM mesh max element size [m] (production)
CURRENT_AMPLITUDE: float = 1.0  # injection current amplitude [A]
N_BOUNDARY_POINTS: int = 256  # boundary polygon points for Netgen circle

# All stimulation patterns: pairs (i, j) with i < j, i,j in 0..15 (0-indexed)
STIM_PATTERNS: List[Tuple[int, int]] = [
    (i, j) for i in range(N_ELEC) for j in range(i + 1, N_ELEC)
]
assert len(STIM_PATTERNS) == N_STIM, "STIM_PATTERNS must have exactly 120 pairs"

# ---------------------------------------------------------------------------
# EIDORS PATHS (configurable via environment variables)
# ---------------------------------------------------------------------------

EIDORS_PATH: str = os.environ.get(
    "EIDORS_PATH",
    r"C:\path\to\eidors-v3.12-ng\eidors",
)
"""Path to the EIDORS startup.m directory (configurable via EIDORS_PATH env var)."""

EIDORS_ROOT: str = os.environ.get(
    "EIDORS_ROOT",
    r"C:\path\to\eidors-v3.12-ng",
)
"""Path to the EIDORS root directory (configurable via EIDORS_ROOT env var)."""

# ---------------------------------------------------------------------------
# ML / TRAINING CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42
NOISE_FRACTIONS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
DATASET_CHUNK_SIZE: int = 100  # samples per HDF5 write chunk
N_ENSEMBLE_MEMBERS: int = 5    # number of deep ensemble members
N_RESTARTS: int = 5            # LM multi-start restarts
TRAIN_STEPS: int = 6000        # surrogate training steps
BATCH_SIZE: int = 512          # training mini-batch size (< 800 training samples for 1K dataset)
LAMBDA_VOLTAGE: float = 1.0    # Sobolev loss: voltage term weight
LAMBDA_JACOBIAN: float = 0.01  # Sobolev loss: Jacobian term weight (LOCKED production value)

# ---------------------------------------------------------------------------
# SURROGATE ARCHITECTURE DEFAULTS
# ---------------------------------------------------------------------------

SURROGATE_HIDDEN_DIM: int = 512
SURROGATE_N_BLOCKS: int = 6

# ---------------------------------------------------------------------------
# DATASET METADATA SCHEMA
# ---------------------------------------------------------------------------

DATASET_SOURCE_SYNTHETIC: str = "synthetic_fem"
DATASET_SOURCE_REAL: str = "real_eit"
