"""
Geometry Jacobian computation.

J_theta = J_sigma @ dSigma_dTheta

  J_sigma        shape (N_meas, N_elem)  — EIDORS conductivity Jacobian
  dSigma_dTheta  shape (N_elem, 64)      — analytic B-spline derivative
  J_theta        shape (N_meas, 64)      — geometry Jacobian (always small)

No EIDORS dependency at import time. Runs on Colab if FEMBackend not used.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from physics.fem_backend import FEMBackend


def fem_jacobian(theta: np.ndarray, backend: "FEMBackend") -> np.ndarray:
    """
    Compute the geometry Jacobian J_theta = dV/dtheta.

    J_theta is computed as:
        J_theta = J_sigma @ dSigma_dTheta

    where:
        J_sigma       = EIDORS conductivity Jacobian, shape (1920, N_elem)
        dSigma_dTheta = analytic conductivity derivative, shape (N_elem, 64)

    The composition J_theta has shape (1920, 64) regardless of N_elem.
    This avoids the need to ever store large J_sigma in Python memory;
    the matrix multiply is done immediately after Octave returns J_sigma.

    Memory note (at maxsz=0.05):
        N_elem  ~ 2,000–4,000
        J_sigma ~ 45 MB float64   (temporary; freed after multiply)
        J_theta ~  1 MB float64   (always small)

    Args:
        theta:   B-spline parameters, shape (64,).
        backend: Configured and built FEMBackend instance.

    Returns:
        J_theta: Geometry Jacobian, shape (1920, 64).
    """
    _, J_theta = backend.solve_with_jacobian(theta)
    return J_theta


def fem_solve_and_jacobian(
    theta: np.ndarray,
    backend: "FEMBackend",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute voltage and geometry Jacobian in a single Octave call.

    Prefer this over calling fem_voltage + fem_jacobian separately
    to avoid double FEM solves.

    Args:
        theta:   B-spline parameters, shape (64,).
        backend: Configured and built FEMBackend instance.

    Returns:
        voltage:  shape (1920,)
        J_theta:  shape (1920, 64)
    """
    return backend.solve_with_jacobian(theta)
