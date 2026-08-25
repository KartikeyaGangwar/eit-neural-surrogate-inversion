"""
FEM model configuration for the circular EIT domain.

The physical model is a 2D circular domain with:
- 16 surface electrodes
- Complete electrode model (CEM)
- EIDORS ng_mk_2d_model (Netgen-based)
- fwd_solve_1st_order

This module does NOT import EIDORS. It defines the model parameters
that fem_backend.py serializes into Octave scripts.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class FEMModelConfig:
    """
    Static configuration for the EIT Finite Element Model.
    """
    n_elec: int = 16
    domain_radius: float = 1.0
    electrode_width: float = 0.20
    electrode_rfnum: int = 10
    z_contact: float = 0.01
    mesh_maxsz: float = 0.05
    n_boundary_points: int = 256  # for Netgen circle construction
    current_amplitude: float = 1.0
    normalize_measurements: int = 0
    solver: str = 'fwd_solve_1st_order'
    system_mat: str = 'system_mat_1st_order'

    @property
    def stim_patterns(self) -> List[Tuple[int, int]]:
        """
        List of all electrode pairs for stimulation (i < j).
        """
        patterns = []
        for i in range(self.n_elec):
            for j in range(i + 1, self.n_elec):
                patterns.append((i, j))
        return patterns

    @property
    def n_stim(self) -> int:
        """Total number of stimulation patterns (120)."""
        return 120

    @property
    def n_meas(self) -> int:
        """Total number of measurements (120 * 16 = 1920)."""
        return 1920

    def measurement_matrix(self) -> np.ndarray:
        """
        Returns the mean-subtracted measurement matrix.
        Shape: (16, 16)
        """
        return np.eye(self.n_elec) - np.ones((self.n_elec, self.n_elec)) / self.n_elec

    def to_dict(self) -> dict:
        """
        Serializable representation for logging/metadata.
        """
        return asdict(self)
