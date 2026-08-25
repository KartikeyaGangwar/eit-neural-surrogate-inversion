"""
Geometry reconstruction evaluation and metrics.
================================================

Computes reconstruction metrics between predicted theta and true theta:
  - parameter_error:      ||theta_pred - theta_true||
  - boundary_mean_dist:   mean distance between B-spline boundary curves
  - boundary_rms_dist:    RMS distance between B-spline boundary curves
  - boundary_max_dist:    max distance between B-spline boundary curves
  - iou:                  Intersection-over-Union via rasterization

No polygon IoU logic. All metrics operate on 64-D B-spline boundaries.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, Tuple

import config as cfg
from geometry.bspline import compute_bspline_boundary


from matplotlib.path import Path


def compute_boundary_distances(
    b_pred: np.ndarray, b_true: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute symmetric point-to-curve boundary distances (Mean, RMS, Max/Hausdorff).
    Invariant to parameterization starting point, speed, and orientation.

    Args:
        b_pred: shape (N, 2)
        b_true: shape (M, 2)

    Returns:
        mean_dist, rms_dist, max_dist in meters [m].
    """
    b_pred = np.asarray(b_pred, dtype=np.float32)
    b_true = np.asarray(b_true, dtype=np.float32)

    d_pred_to_true = np.empty(len(b_pred), dtype=np.float32)
    for i, p in enumerate(b_pred):
        d_pred_to_true[i] = np.sqrt(np.min(np.sum((b_true - p) ** 2, axis=1)))

    d_true_to_pred = np.empty(len(b_true), dtype=np.float32)
    for j, q in enumerate(b_true):
        d_true_to_pred[j] = np.sqrt(np.min(np.sum((b_pred - q) ** 2, axis=1)))

    mean_dist = float(0.5 * (np.mean(d_pred_to_true) + np.mean(d_true_to_pred)))
    rms_dist = float(np.sqrt(0.5 * (np.mean(d_pred_to_true ** 2) + np.mean(d_true_to_pred ** 2))))
    max_dist = float(max(np.max(d_pred_to_true), np.max(d_true_to_pred)))

    return mean_dist, rms_dist, max_dist


def compute_metrics(
    theta_pred: np.ndarray,
    theta_true: np.ndarray,
    n_boundary_samples: int = 300,
) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction metrics between predicted and true theta.

    Args:
        theta_pred: shape (64,)
        theta_true: shape (64,)

    Returns:
        dict containing:
            - parameter_l2_error
            - parameter_rel_error
            - boundary_mean_dist [m]
            - boundary_rms_dist [m]
            - boundary_max_dist [m]
            - iou
    """
    assert theta_pred.shape == (cfg.N_THETA,)
    assert theta_true.shape == (cfg.N_THETA,)

    # Parameter space errors
    param_l2 = float(np.linalg.norm(theta_pred - theta_true))
    param_rel = float(param_l2 / max(np.linalg.norm(theta_true), 1e-14))

    # Boundary space errors
    b_pred, _ = compute_bspline_boundary(theta_pred, n_boundary_samples)
    b_true, _ = compute_bspline_boundary(theta_true, n_boundary_samples)

    mean_dist, rms_dist, max_dist = compute_boundary_distances(b_pred, b_true)

    # Rasterized IoU
    iou_val = rasterize_iou(theta_pred, theta_true, n_boundary_samples=n_boundary_samples)

    return {
        "parameter_l2_error": param_l2,
        "parameter_rel_error": param_rel,
        "boundary_mean_dist": mean_dist,
        "boundary_rms_dist": rms_dist,
        "boundary_max_dist": max_dist,
        "iou": iou_val,
    }


def rasterize_iou(
    theta_pred: np.ndarray,
    theta_true: np.ndarray,
    grid_size: int = 200,
    n_boundary_samples: int = 300,
) -> float:
    """
    Compute Intersection-over-Union (IoU) by rasterizing closed B-spline contours onto a 2D grid.

    Args:
        theta_pred: shape (64,)
        theta_true: shape (64,)
        grid_size:  resolution of grid (default 200x200).
        n_boundary_samples: samples along B-spline polygon boundary.

    Returns:
        IoU score in [0, 1].
    """
    b_pred, _ = compute_bspline_boundary(theta_pred, n_boundary_samples)
    b_true, _ = compute_bspline_boundary(theta_true, n_boundary_samples)

    x = np.linspace(-cfg.DOMAIN_RADIUS, cfg.DOMAIN_RADIUS, grid_size, dtype=np.float32)
    y = np.linspace(-cfg.DOMAIN_RADIUS, cfg.DOMAIN_RADIUS, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    path_pred = Path(b_pred)
    path_true = Path(b_true)

    mask_pred = path_pred.contains_points(grid_pts)
    mask_true = path_true.contains_points(grid_pts)

    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)
