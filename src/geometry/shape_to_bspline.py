"""
Shape-to-B-spline fitting and conversion utilities.
===================================================

Fits a 32-control-point closed cubic B-spline to any set of ordered 2D boundary points
via linear least-squares.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Callable, Dict, Any

from config import N_CONTROL
from geometry.bspline import compute_bspline_boundary


def fit_bspline_to_boundary(
    boundary_points: np.ndarray, n_control: int = N_CONTROL
) -> np.ndarray:
    """
    Fit a closed cubic B-spline to ordered 2D boundary points via linear least-squares.

    Args:
        boundary_points: shape (n_samples, 2) — ordered closed curve points.
        n_control:       number of control points (default 32).

    Returns:
        theta: shape (64,) — B-spline parameter vector [X1..X32, Y1..Y32].
    """
    assert boundary_points.ndim == 2 and boundary_points.shape[1] == 2, \
        f"Expected boundary_points shape (n_samples, 2), got {boundary_points.shape}"

    n_samples = boundary_points.shape[0]
    u = np.linspace(0.0, float(n_control), n_samples, endpoint=False)
    segment = np.floor(u).astype(int)
    t = u - segment

    B0 = (1.0 - t) ** 3 / 6.0
    B1 = (3.0 * t ** 3 - 6.0 * t ** 2 + 4.0) / 6.0
    B2 = (-3.0 * t ** 3 + 3.0 * t ** 2 + 3.0 * t + 1.0) / 6.0
    B3 = t ** 3 / 6.0

    A = np.zeros((n_samples, n_control))
    for i in range(n_samples):
        s = segment[i]
        A[i, s % n_control] += B0[i]
        A[i, (s + 1) % n_control] += B1[i]
        A[i, (s + 2) % n_control] += B2[i]
        A[i, (s + 3) % n_control] += B3[i]

    cp_x = np.linalg.lstsq(A, boundary_points[:, 0], rcond=None)[0]
    cp_y = np.linalg.lstsq(A, boundary_points[:, 1], rcond=None)[0]

    return np.concatenate([cp_x, cp_y])


def shape_to_theta(shape_fn: Callable, **shape_kwargs: Any) -> Tuple[np.ndarray, dict]:
    """
    Call a shape generation function, fit a B-spline, and return theta and metadata.

    Args:
        shape_fn:     Function returning (boundary_points (N,2), metadata dict).
        shape_kwargs: Keyword arguments for shape_fn.

    Returns:
        (theta (64,), metadata dict).
    """
    boundary_points, metadata = shape_fn(**shape_kwargs)
    theta = fit_bspline_to_boundary(boundary_points)
    return theta, metadata


def validate_fit_quality(
    theta: np.ndarray, original_boundary: np.ndarray
) -> Dict[str, float]:
    """
    Compute mean, RMS, and max distance between fitted B-spline boundary and original.

    Args:
        theta:             shape (64,)
        original_boundary: shape (n_samples, 2)

    Returns:
        dict with mean_error, rms_error, max_error
    """
    n_samples = original_boundary.shape[0]
    fitted_boundary, _ = compute_bspline_boundary(theta, n_samples)
    distances = np.linalg.norm(fitted_boundary - original_boundary, axis=1)

    return {
        "mean_error": float(np.mean(distances)),
        "rms_error": float(np.sqrt(np.mean(distances ** 2))),
        "max_error": float(np.max(distances)),
    }
