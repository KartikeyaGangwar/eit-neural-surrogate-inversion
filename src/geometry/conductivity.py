"""
Conductivity field computation for the B-spline EIT model.
==========================================================

Implements the exact divergence-theorem boundary integral from
polygon_conductivity_bspline.m (lines 70-143).

The conductivity model assigns each FEM element a conductivity value based
on whether its centroid lies inside or outside the B-spline boundary.

The membership function uses a mollified indicator via a boundary integral:

    H(x) = (1/2pi) * integral over boundary of A(|x-q|^2) * N_cross ds

where:
    A(s) = (1 - exp(-s/eps^2)) / s      [for s >= threshold]
    N_cross = (q-x) x dq/dt             [cross product = signed area integrand]

The occupancy probability is then:
    occupancy(x) = 0.5 * (1 + tanh(alpha * (H(x) - 0.5)))

And the conductivity:
    sigma(x) = sigma_bg + (sigma_inc - sigma_bg) * occupancy(x)

All constants match polygon_conductivity_bspline.m exactly.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# Use absolute-style imports relative to the package
# These work whether the project is installed or used as a local package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    N_CONTROL,
    N_BOUNDARY_SAMP,
    MOLLIFIER_EPS,
    MOLLIFIER_EPS2,
    SERIES_THRESHOLD,
    ALPHA,
    SIGMA_BG,
    SIGMA_INC,
)
from geometry.bspline import compute_bspline_boundary


def compute_sigma_and_derivative(
    theta: np.ndarray,
    centres: np.ndarray,
    sigma_bg: float = SIGMA_BG,
    sigma_inc: float = SIGMA_INC,
    alpha: float = ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full single-pass computation of conductivity and its analytic derivative.

    Exact Python translation of polygon_conductivity_bspline.m lines 70-143.

    Args:
        theta:      B-spline parameters, shape (64,).
                    theta = [X1...X32, Y1...Y32]
        centres:    FEM element centroid coordinates, shape (N_elem, 2).
        sigma_bg:   Background conductivity [S/m].
        sigma_inc:  Inclusion conductivity [S/m].
        alpha:      tanh sharpness (default 80, matches MATLAB reference).

    Returns:
        elem_sigma:     Element conductivities, shape (N_elem,).
        dSigma_dTheta:  Analytic derivative, shape (N_elem, 64).
                        Column k is d(sigma)/d(theta_k).
                        Columns 0..31: derivatives w.r.t. X coordinates.
                        Columns 32..63: derivatives w.r.t. Y coordinates.
    """
    assert theta.shape == (64,), f"Expected theta shape (64,), got {theta.shape}"
    assert centres.ndim == 2 and centres.shape[1] == 2, \
        f"Expected centres shape (N_elem, 2), got {centres.shape}"

    N_elem = centres.shape[0]
    n_samples = N_BOUNDARY_SAMP
    eps2 = MOLLIFIER_EPS2  # eps^2, exactly as in MATLAB: eps2 = mollifier_eps^2
    du = N_CONTROL / n_samples  # step size in parameter space

    H = np.zeros(N_elem)
    dH_dTheta = np.zeros((N_elem, 64))

    # Compute boundary points and tangents (both shapes: (n_samples, 2))
    boundary_points, boundary_tangents = compute_bspline_boundary(theta, n_samples)
    qx = boundary_points[:, 0]     # (n_samples,)
    qy = boundary_points[:, 1]     # (n_samples,)
    qtx = boundary_tangents[:, 0]  # (n_samples,)
    qty = boundary_tangents[:, 1]  # (n_samples,)

    # Pre-compute B-spline indices and basis values for all samples
    u_vals = np.linspace(0, N_CONTROL, n_samples + 1)[:-1]
    segment = np.floor(u_vals).astype(int)
    t = u_vals - segment

    B0 = (1.0 - t) ** 3 / 6.0
    B1 = (3.0 * t ** 3 - 6.0 * t ** 2 + 4.0) / 6.0
    B2 = (-3.0 * t ** 3 + 3.0 * t ** 2 + 3.0 * t + 1.0) / 6.0
    B3 = t ** 3 / 6.0

    dB0 = -(1.0 - t) ** 2 / 2.0
    dB1 = (3.0 * t ** 2 - 4.0 * t) / 2.0
    dB2 = (-3.0 * t ** 2 + 2.0 * t + 1.0) / 2.0
    dB3 = t ** 2 / 2.0

    # Control point indices (wrapped, matching MATLAB mod(segment, n_control)+1 in 1-indexed)
    i0 = segment % N_CONTROL
    i1 = (segment + 1) % N_CONTROL
    i2 = (segment + 2) % N_CONTROL
    i3 = (segment + 3) % N_CONTROL

    px = centres[:, 0]  # (N_elem,)
    py = centres[:, 1]  # (N_elem,)
    factor = du / (2.0 * np.pi)

    for m in range(n_samples):
        qxm = qx[m]
        qym = qy[m]
        qtxm = qtx[m]
        qtym = qty[m]

        # Displacement from boundary sample to all element centres
        rx = qxm - px  # (N_elem,)
        ry = qym - py  # (N_elem,)

        s = rx ** 2 + ry ** 2  # (N_elem,)

        # Signed cross product (divergence theorem integrand direction)
        Ncross = rx * qtym - ry * qtxm  # (N_elem,)

        A = np.empty(N_elem)
        Aprime = np.empty(N_elem)

        # --- Branch: s < SERIES_THRESHOLD (Taylor series for numerical stability) ---
        small = s < SERIES_THRESHOLD
        if np.any(small):
            ss = s[small]
            A[small] = (1.0 / eps2
                        - ss / (2.0 * eps2 ** 2)
                        + (ss ** 2) / (6.0 * eps2 ** 3))
            Aprime[small] = (-1.0 / (2.0 * eps2 ** 2)
                             + ss / (3.0 * eps2 ** 3))

        # --- Branch: s >= SERIES_THRESHOLD (exact formula) ---
        normal = ~small
        if np.any(normal):
            sn = s[normal]
            e = np.exp(-sn / eps2)
            A[normal] = (1.0 - e) / sn
            Aprime[normal] = ((1.0 + sn / eps2) * e - 1.0) / (sn ** 2)

        # Accumulate boundary integral
        integrand = A * Ncross
        H += factor * integrand

        # Distribute analytic derivatives to the 4 active control points
        idx = [i0[m], i1[m], i2[m], i3[m]]
        B_vals = [B0[m], B1[m], B2[m], B3[m]]
        dB_vals = [dB0[m], dB1[m], dB2[m], dB3[m]]

        for j in range(4):
            cp_idx = idx[j]
            bj = B_vals[j]
            dbj = dB_vals[j]

            # --- Derivative w.r.t. X coordinate of control point cp_idx ---
            # d(rx)/d(X_cp) = bj,   d(qx)/d(X_cp) = bj
            # d(qtx)/d(X_cp) = dbj
            # ds/d(X_cp) = 2*rx*bj
            # d(Ncross)/d(X_cp) = bj*qtym - ry*dbj
            ds_dx = 2.0 * rx * bj
            dNcross_dx = bj * qtym - ry * dbj
            val_x = Aprime * ds_dx * Ncross + A * dNcross_dx
            dH_dTheta[:, cp_idx] += factor * val_x

            # --- Derivative w.r.t. Y coordinate of control point cp_idx ---
            # d(ry)/d(Y_cp) = bj,   d(qy)/d(Y_cp) = bj
            # d(qty)/d(Y_cp) = dbj
            # ds/d(Y_cp) = 2*ry*bj
            # d(Ncross)/d(Y_cp) = -bj*qtxm + rx*dbj
            ds_dy = 2.0 * ry * bj
            dNcross_dy = -bj * qtxm + rx * dbj
            val_y = Aprime * ds_dy * Ncross + A * dNcross_dy
            dH_dTheta[:, 32 + cp_idx] += factor * val_y

    # --- Occupancy mapping ---
    occ_arg = alpha * (H - 0.5)
    s_map = 0.5 * (1.0 + np.tanh(occ_arg))
    ds_dH = 0.5 * alpha * (1.0 - np.tanh(occ_arg) ** 2)

    ds_dTheta = ds_dH[:, np.newaxis] * dH_dTheta  # (N_elem, 64)

    # --- Final conductivity ---
    delta_sigma = sigma_inc - sigma_bg
    elem_sigma = sigma_bg + delta_sigma * s_map       # (N_elem,)
    dSigma_dTheta = delta_sigma * ds_dTheta           # (N_elem, 64)

    return elem_sigma, dSigma_dTheta


def compute_sigma(
    theta: np.ndarray,
    centres: np.ndarray,
    sigma_bg: float = SIGMA_BG,
    sigma_inc: float = SIGMA_INC,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Compute element conductivities (convenience wrapper).

    Args:
        theta:    B-spline parameters, shape (64,).
        centres:  FEM element centroids, shape (N_elem, 2).

    Returns:
        elem_sigma: shape (N_elem,).
    """
    sigma, _ = compute_sigma_and_derivative(theta, centres, sigma_bg, sigma_inc, alpha)
    return sigma


def compute_dsigma_dtheta(
    theta: np.ndarray,
    centres: np.ndarray,
    sigma_bg: float = SIGMA_BG,
    sigma_inc: float = SIGMA_INC,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Compute analytic conductivity derivative (convenience wrapper).

    Args:
        theta:    B-spline parameters, shape (64,).
        centres:  FEM element centroids, shape (N_elem, 2).

    Returns:
        dSigma_dTheta: shape (N_elem, 64).
    """
    _, dsigma = compute_sigma_and_derivative(theta, centres, sigma_bg, sigma_inc, alpha)
    return dsigma
