"""
Cubic B-spline boundary evaluation.
====================================

Exact Python translation of polygon_conductivity_bspline.m lines 33-65.

The B-spline is CLOSED: control points wrap around with index arithmetic
mod N_CONTROL (= 32).

Parameter ordering (authoritative):
    theta = [X1 ... X32, Y1 ... Y32]  shape (64,)
    control_points[:, 0] = theta[0:32]
    control_points[:, 1] = theta[32:64]

Cubic uniform B-spline basis (MATLAB reference):
    B0(t) = (1-t)^3 / 6
    B1(t) = (3t^3 - 6t^2 + 4) / 6
    B2(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
    B3(t) = t^3 / 6

Derivatives:
    dB0(t) = -(1-t)^2 / 2
    dB1(t) = (3t^2 - 4t) / 2
    dB2(t) = (-3t^2 + 2t + 1) / 2
    dB3(t) = t^2 / 2

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
# Allow running as a script or as part of the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple

from config import N_CONTROL, N_BOUNDARY_SAMP


def theta_to_control_points(theta: np.ndarray) -> np.ndarray:
    """
    Unpack the 64-D parameter vector into a (32, 2) control-point array.

    Args:
        theta: shape (64,)  — [X1..X32, Y1..Y32]

    Returns:
        control_points: shape (32, 2)
            Column 0 = X coordinates (theta[0:32])
            Column 1 = Y coordinates (theta[32:64])

    Matches MATLAB:
        control_points = [theta(1:n_control), theta(n_control+1:n_theta)];
    """
    assert theta.shape == (64,), f"Expected shape (64,), got {theta.shape}"
    return np.stack([theta[:32], theta[32:]], axis=1)


def compute_bspline_boundary(
    theta: np.ndarray,
    n_samples: int = N_BOUNDARY_SAMP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the closed cubic B-spline at n_samples uniform parameter values.

    Exact translation of polygon_conductivity_bspline.m lines 33-65.

    Args:
        theta:     shape (64,)  — B-spline control point parameters.
        n_samples: number of boundary samples (default 1024).

    Returns:
        boundary_points:   shape (n_samples, 2) — [x, y] at each sample.
        boundary_tangents: shape (n_samples, 2) — [dx/dt, dy/dt] at each sample.

    The parameter u runs in [0, N_CONTROL) with n_samples uniform steps.
    The curve is CLOSED: the last sample smoothly connects back to the first
    through the wrapped index arithmetic.
    """
    assert theta.shape == (64,), f"Expected shape (64,), got {theta.shape}"

    cp = theta_to_control_points(theta)  # (32, 2)

    # Uniform parameter values in [0, N_CONTROL), exactly as MATLAB:
    #   u = linspace(0, n_control, n_boundary_samples + 1)'; u(end) = [];
    u = np.linspace(0.0, float(N_CONTROL), n_samples + 1)[:-1]

    segment = np.floor(u).astype(int)  # which segment [0..N_CONTROL-1]
    t = u - segment                    # local parameter in [0, 1)

    # Cubic B-spline basis (MATLAB lines 41-44)
    B0 = (1.0 - t) ** 3 / 6.0
    B1 = (3.0 * t ** 3 - 6.0 * t ** 2 + 4.0) / 6.0
    B2 = (-3.0 * t ** 3 + 3.0 * t ** 2 + 3.0 * t + 1.0) / 6.0
    B3 = t ** 3 / 6.0

    # Basis derivatives w.r.t t (MATLAB lines 47-50)
    dB0 = -(1.0 - t) ** 2 / 2.0
    dB1 = (3.0 * t ** 2 - 4.0 * t) / 2.0
    dB2 = (-3.0 * t ** 2 + 2.0 * t + 1.0) / 2.0
    dB3 = t ** 2 / 2.0

    # Wrapped control-point indices (MATLAB lines 53-56)
    i0 = segment % N_CONTROL
    i1 = (segment + 1) % N_CONTROL
    i2 = (segment + 2) % N_CONTROL
    i3 = (segment + 3) % N_CONTROL

    # Boundary positions (MATLAB lines 58-61)
    qx = B0 * cp[i0, 0] + B1 * cp[i1, 0] + B2 * cp[i2, 0] + B3 * cp[i3, 0]
    qy = B0 * cp[i0, 1] + B1 * cp[i1, 1] + B2 * cp[i2, 1] + B3 * cp[i3, 1]

    # Boundary tangents (MATLAB lines 64-65)
    qtx = dB0 * cp[i0, 0] + dB1 * cp[i1, 0] + dB2 * cp[i2, 0] + dB3 * cp[i3, 0]
    qty = dB0 * cp[i0, 1] + dB1 * cp[i1, 1] + dB2 * cp[i2, 1] + dB3 * cp[i3, 1]

    boundary_points = np.stack([qx, qy], axis=1)      # (n_samples, 2)
    boundary_tangents = np.stack([qtx, qty], axis=1)  # (n_samples, 2)

    return boundary_points, boundary_tangents


def evaluate_bspline_point(theta: np.ndarray, u: float) -> np.ndarray:
    """
    Evaluate the B-spline at a single parameter value u.

    Args:
        theta: shape (64,)
        u:     parameter value in [0, N_CONTROL)

    Returns:
        point: shape (2,)  — [x, y]
    """
    assert theta.shape == (64,), f"Expected shape (64,), got {theta.shape}"
    cp = theta_to_control_points(theta)

    seg = int(np.floor(u)) % N_CONTROL
    t = u - np.floor(u)

    B0 = (1.0 - t) ** 3 / 6.0
    B1 = (3.0 * t ** 3 - 6.0 * t ** 2 + 4.0) / 6.0
    B2 = (-3.0 * t ** 3 + 3.0 * t ** 2 + 3.0 * t + 1.0) / 6.0
    B3 = t ** 3 / 6.0

    i0 = seg % N_CONTROL
    i1 = (seg + 1) % N_CONTROL
    i2 = (seg + 2) % N_CONTROL
    i3 = (seg + 3) % N_CONTROL

    x = B0 * cp[i0, 0] + B1 * cp[i1, 0] + B2 * cp[i2, 0] + B3 * cp[i3, 0]
    y = B0 * cp[i0, 1] + B1 * cp[i1, 1] + B2 * cp[i2, 1] + B3 * cp[i3, 1]

    return np.array([x, y])
