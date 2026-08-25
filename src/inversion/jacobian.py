"""
Jacobian calculation wrapper.
=============================

Provides direct and finite-difference Jacobian computation tools.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import numpy as np
from typing import Callable

import config as cfg


def finite_difference_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """
    Compute central finite-difference Jacobian of func(theta).

    Args:
        func:  Callable mapping theta (64,) -> output (M,).
        theta: shape (64,).
        h:     FD step size.

    Returns:
        J: shape (M, 64)
    """
    f0 = func(theta)
    M = f0.shape[0]
    J = np.zeros((M, cfg.N_THETA))

    for k in range(cfg.N_THETA):
        tp = theta.copy()
        tp[k] += h
        tm = theta.copy()
        tm[k] -= h
        J[:, k] = (func(tp) - func(tm)) / (2.0 * h)

    return J
