"""
Parameter-space utilities for the 64-D B-spline parameterization.
==================================================================

Provides authoritative conversions between theta and control_points,
geometry validity checking, concavity analysis, self-intersection testing,
and the canonical reference geometry.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Dict, Any

from config import N_CONTROL, N_THETA, DOMAIN_RADIUS
from geometry.bspline import theta_to_control_points, compute_bspline_boundary


def control_points_to_theta(cp: np.ndarray) -> np.ndarray:
    """
    Pack a (32, 2) control-point array into the 64-D parameter vector.

    Args:
        cp: shape (32, 2)  — control points [[x0,y0], ..., [x31,y31]]

    Returns:
        theta: shape (64,)  — [x0..x31, y0..y31]
    """
    assert cp.shape == (N_CONTROL, 2), f"Expected shape ({N_CONTROL}, 2), got {cp.shape}"
    return np.concatenate([cp[:, 0], cp[:, 1]])


def analyze_concavity(pts: np.ndarray) -> Dict[str, Any]:
    """
    Analyze convexity/concavity of a closed 2D boundary curve.

    IMPORTANT: Uses discrete cross-product (signed-turn) per vertex.
    This correctly detects sharp-cornered concave shapes (star, banana, crescent)
    but may return is_convex=True for smooth polar-mode concave curves whose
    reflex region is spread over many samples.

    For smooth concavity, use analyze_smooth_concavity() which uses the
    curvature estimator.

    Args:
        pts: shape (N, 2) — ordered 2D boundary points.

    Returns:
        dict with:
            is_convex: bool (False if ANY reflex vertex detected)
            n_reflex: int
            reflex_fraction: float
            min_turning_cross: float
            concavity_depth: float (via smooth curvature estimation)
    """
    n = len(pts)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)

    # Cross product at each vertex
    d1 = next_pts - pts
    d2 = prev_pts - pts
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

    orient = np.sign(np.sum(cross))
    if orient == 0.0:
        orient = 1.0

    signed_curv = cross * orient
    scale = float(np.max(np.linalg.norm(pts, axis=1)) ** 2)
    eps = -1e-9 * max(scale, 1e-6)
    reflex_mask = signed_curv < eps
    n_reflex = int(np.sum(reflex_mask))

    # ALSO compute smooth curvature depth (catches smooth concavity)
    smooth_depth = _smooth_concavity_depth(pts)

    # A shape is concave if EITHER discrete reflex OR smooth curvature is negative
    is_concave = (n_reflex > 0) or (smooth_depth > 0.01)

    return {
        "is_convex": not is_concave,
        "n_reflex": n_reflex,
        "reflex_fraction": float(n_reflex / n),
        "min_turning_cross": float(np.min(signed_curv)),
        "concavity_depth": float(smooth_depth),
    }


def _smooth_concavity_depth(pts: np.ndarray) -> float:
    """
    Estimate smooth concavity depth using finite-difference curvature.

    Computes the signed curvature kappa at each point via:
        kappa ~ (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)

    Returns the maximum concavity depth (max of -min_kappa * scale, 0).
    A positive value indicates a genuinely concave smooth curve.
    """
    # Finite differences for first and second derivatives
    pts_p = np.roll(pts, -1, axis=0)  # p_{i+1}
    pts_m = np.roll(pts, 1, axis=0)   # p_{i-1}

    dx = (pts_p[:, 0] - pts_m[:, 0]) / 2.0   # x'
    dy = (pts_p[:, 1] - pts_m[:, 1]) / 2.0   # y'
    d2x = pts_p[:, 0] - 2*pts[:, 0] + pts_m[:, 0]  # x''
    d2y = pts_p[:, 1] - 2*pts[:, 1] + pts_m[:, 1]  # y''

    speed_sq = dx**2 + dy**2
    kappa_num = dx * d2y - dy * d2x  # signed curvature numerator

    # Determine overall orientation (CCW = positive)
    # For CCW curve, mean kappa_num should be positive
    orient = np.sign(np.mean(kappa_num))
    if orient == 0.0:
        orient = 1.0

    signed_kappa_num = kappa_num * orient
    min_kappa = float(np.min(signed_kappa_num))
    mean_speed = float(np.mean(speed_sq))

    # Normalize by scale
    depth = max(0.0, -min_kappa / (mean_speed + 1e-15))
    return depth


def check_self_intersection(pts: np.ndarray, max_subsample: int = 128) -> bool:
    """
    Check if a closed 2D curve self-intersects using a robust O(N^2) segment test.

    Args:
        pts: shape (N, 2) — ordered boundary points.
        max_subsample: max points for fast intersection test.

    Returns:
        True if self-intersecting, False otherwise.
    """
    if len(pts) > max_subsample:
        indices = np.linspace(0, len(pts) - 1, max_subsample, dtype=int)
        pts = pts[indices]

    n = len(pts)

    def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """
        Test whether three 2D points A, B, C are in counter-clockwise order.
        
        Args:
            A, B, C: 2D coordinate arrays of shape (2,).
            
        Returns:
            True if (B - A) x (C - A) > 0 (counter-clockwise orientation).
        """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
        """
        Test whether line segment AB intersects line segment CD.
        
        Args:
            A, B: Endpoints of segment 1.
            C, D: Endpoints of segment 2.
            
        Returns:
            True if segments AB and CD strictly intersect.
        """
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if intersect(p1, p2, p3, p4):
                return True

    return False


def check_geometry_validity(
    theta: np.ndarray,
    domain_radius: float = DOMAIN_RADIUS,
    margin: float = 0.02,
) -> Tuple[bool, str]:
    """
    Check whether theta represents a valid FEM geometry.

    Validity criteria:
        1. theta has shape (64,)
        2. All values are finite (no NaN, no Inf)
        3. All control points lie within the FEM domain circle (radius < domain_radius - margin)
        4. B-spline boundary points lie within the FEM domain circle
        5. B-spline boundary curve does not self-intersect

    NOT checked (intentionally):
        - Convexity (concave shapes are valid)
        - Winding order (B-spline has no orientation constraint)

    Args:
        theta:         shape (64,)
        domain_radius: outer FEM domain radius (default 1.0)
        margin:        optional inset margin [0, domain_radius)

    Returns:
        (valid, reason): bool and human-readable explanation
    """
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta, dtype=float)

    if theta.shape != (N_THETA,):
        return False, f"Shape must be ({N_THETA},), got {theta.shape}"

    if not np.all(np.isfinite(theta)):
        n_bad = int(np.sum(~np.isfinite(theta)))
        return False, f"{n_bad} non-finite values (NaN or Inf) in theta"

    cp = theta_to_control_points(theta)
    cp_radii = np.sqrt(cp[:, 0] ** 2 + cp[:, 1] ** 2)
    limit = domain_radius - margin
    if np.any(cp_radii > limit):
        worst = float(np.max(cp_radii))
        return False, (
            f"Control point radius {worst:.4f} exceeds limit {limit:.4f}."
        )

    # Evaluate B-spline boundary curve
    bnd_pts, _ = compute_bspline_boundary(theta, n_samples=256)
    bnd_radii = np.sqrt(bnd_pts[:, 0] ** 2 + bnd_pts[:, 1] ** 2)
    if np.any(bnd_radii > limit):
        worst = float(np.max(bnd_radii))
        return False, (
            f"B-spline boundary radius {worst:.4f} exceeds domain limit {limit:.4f}."
        )

    if check_self_intersection(bnd_pts):
        return False, "B-spline boundary curve self-intersects."

    return True, "Valid"


def sample_reference_geometry() -> np.ndarray:
    """
    Return the canonical reference theta (64,) from the MATLAB validation scripts.

    Returns:
        theta: shape (64,) — [X1..X32, Y1..Y32]
    """
    cp = np.array([
        [ 0.45,  0.00], [ 0.43,  0.10], [ 0.36,  0.19], [ 0.25,  0.27],
        [ 0.12,  0.31], [-0.02,  0.30], [-0.15,  0.25], [-0.28,  0.18],
        [-0.38,  0.07], [-0.42, -0.06], [-0.38, -0.18], [-0.29, -0.27],
        [-0.17, -0.31], [-0.04, -0.30], [ 0.06, -0.24], [ 0.03, -0.13],
        [-0.05, -0.06], [-0.10,  0.03], [-0.07,  0.12], [ 0.02,  0.17],
        [ 0.13,  0.16], [ 0.22,  0.10], [ 0.28,  0.00], [ 0.25, -0.10],
        [ 0.18, -0.17], [ 0.27, -0.22], [ 0.38, -0.19], [ 0.46, -0.12],
        [ 0.49, -0.04], [ 0.48,  0.02], [ 0.47,  0.05], [ 0.45,  0.00],
    ], dtype=float)
    assert cp.shape == (N_CONTROL, 2)
    return control_points_to_theta(cp)
