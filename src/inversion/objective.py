"""
Objective function computation and gradient diagnostics.
=========================================================

Scalar objective:
    f(theta) = 0.5 * ||r(theta)||^2

Gradient:
    g(theta) = J(theta)^T r(theta)

No artificial polygon regularization penalties. Pure measurement objective.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

from inversion.problem import InversionProblem


def compute_objective_and_gradient(
    problem: InversionProblem, theta: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute objective f, gradient g, residual r, and Jacobian J.

    Args:
        problem: InversionProblem instance.
        theta:   shape (64,)

    Returns:
        (f, g, r, J)
            f: float = 0.5 * ||r||^2
            g: shape (64,) = J^T @ r
            r: shape (1920,)
            J: shape (1920, 64)
    """
    r, J = problem.residual_and_jacobian(theta)
    f = float(0.5 * np.sum(r ** 2))
    g = J.T @ r
    return f, g, r, J
