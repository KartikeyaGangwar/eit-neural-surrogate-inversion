"""
High-level forward voltage interface.

No EIDORS dependency at import time. Runs on Colab if FEMBackend is not used.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physics.fem_backend import FEMBackend


def fem_voltage(theta: np.ndarray, backend: "FEMBackend") -> np.ndarray:
    """
    Compute EIT voltage measurements for a given B-spline geometry.

    Args:
        theta:   B-spline parameters, shape (64,).
                 theta = [X1...X32, Y1...Y32]
        backend: Configured and built FEMBackend instance.

    Returns:
        voltage: EIT measurements, shape (1920,).
                 Ordering: 120 stimulation patterns × 16 electrode measurements,
                 flattened. Measurement matrix is eye(16) - ones(16)/16.

    Notes:
        This is a thin wrapper around backend.solve(theta).
        The FEMBackend handles Octave subprocess management, caching,
        and result parsing.
    """
    return backend.solve(theta)
