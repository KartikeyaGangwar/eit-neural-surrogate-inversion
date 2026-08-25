"""
Dedicated Multi-Shape Inversion Experiment Module.
==================================================

Framework for evaluating shape-inversion performance across diverse shape families
(circle, ellipse, rectangle, star, banana, crescent, random convex, random concave).

WORKFLOW
--------
  1. Generate shape boundary using geometry.shape_library
  2. Fit shape to 64-D B-spline parameterization (geometry.shape_to_bspline)
  3. Generate ground-truth voltage (via surrogate or FEM)
  4. Perform inversion (LM or MultiStart) starting from a neutral guess (e.g. circle)
  5. Compute reconstruction metrics (boundary distance, parameter error, IoU)
  6. Output structured experiment report

KEY PRINCIPLES
--------------
  - The outer FEM domain REMAINS CIRCULAR for all shape families.
  - All shapes are converted to the unified 64-D B-spline theta BEFORE entering inversion.
  - The inverse problem solver sees only theta in R^64 and is oblivious to the shape family.
  - Concave shapes are supported and evaluated with the same pipeline as convex shapes.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import config as cfg
from geometry import shape_library as sl
from geometry.shape_to_bspline import shape_to_theta
from inversion.residuals import SurrogateResidualProblem
from inversion.multistart import MultiStartSolver
from inversion.reconstruction import compute_metrics
from surrogate.model import VoltageSurrogate


@dataclass
class ShapeExperimentResult:
    """Result container for a single shape inversion experiment."""
    shape_family: str
    theta_true: np.ndarray
    theta_reconstructed: np.ndarray
    metrics: Dict[str, float]
    fit_quality: Dict[str, float]
    optimizer_summary: str


class ShapeInversionExperiment:
    """
    Multi-shape inversion test runner.
    """

    def __init__(
        self,
        surrogate: VoltageSurrogate,
        solver: Optional[MultiStartSolver] = None,
    ) -> None:
        self.surrogate = surrogate
        self.solver = solver or MultiStartSolver(n_restarts=cfg.N_RESTARTS)

    def run_single_experiment(
        self,
        shape_family: str,
        shape_kwargs: Optional[dict] = None,
        noise_level: float = 0.0,
        seed: int = cfg.DEFAULT_SEED,
    ) -> ShapeExperimentResult:
        """
        Run a single shape-inversion test.

        Args:
            shape_family: Name of shape generator ('circle', 'star', 'banana', etc.).
            shape_kwargs: Keyword arguments for shape generator.
            noise_level:  Gaussian noise fraction added to measurements.
            seed:         Random seed.

        Returns:
            ShapeExperimentResult dataclass.
        """
        if shape_kwargs is None:
            shape_kwargs = self._default_kwargs_for_shape(shape_family, seed)

        # 1. Generate shape and convert to 64-D theta
        shape_fn = getattr(sl, f"generate_{shape_family}")
        theta_true, meta = shape_to_theta(shape_fn, **shape_kwargs)

        # 2. Forward solve (synthetic measurement)
        v_true = self.surrogate.predict_numpy(theta_true)
        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, noise_level * np.std(v_true), size=v_true.shape)
            v_measured = v_true + noise
        else:
            v_measured = v_true

        # 3. Setup inversion problem (surrogate-backed)
        problem = SurrogateResidualProblem(self.surrogate, v_measured)

        # 4. Run Multi-start LM optimization
        res = self.solver.solve(problem, verbose=False)
        theta_rec = res.theta_best

        # 5. Compute metrics
        metrics = compute_metrics(theta_rec, theta_true)

        return ShapeExperimentResult(
            shape_family=shape_family,
            theta_true=theta_true,
            theta_reconstructed=theta_rec,
            metrics=metrics,
            fit_quality=meta,
            optimizer_summary=res.summary(),
        )

    def run_full_suite(
        self, noise_level: float = 0.0, seed: int = cfg.DEFAULT_SEED
    ) -> List[ShapeExperimentResult]:
        """
        Run shape inversion tests across all supported shape families.

        Supported shape families:
          - circle (convex)
          - ellipse (convex)
          - rectangle (convex)
          - star (concave)
          - banana (concave)
          - crescent (concave)
          - random_convex (convex)
          - random_concave (concave)

        Returns:
            List of ShapeExperimentResult dataclasses.
        """
        families = [
            "circle", "ellipse", "rectangle",
            "star", "banana", "crescent",
            "random_convex", "random_concave",
        ]
        results = []
        for fam in families:
            res = self.run_single_experiment(fam, noise_level=noise_level, seed=seed)
            results.append(res)
        return results

    @staticmethod
    def _default_kwargs_for_shape(family: str, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        center = (0.0, 0.0)

        if family == "circle":
            return {"radius": 0.35, "center": center}
        elif family == "ellipse":
            return {"a": 0.4, "b": 0.25, "center": center, "angle": 0.3}
        elif family == "rectangle":
            return {"width": 0.5, "height": 0.3, "center": center, "angle": 0.2}
        elif family == "star":
            return {"n_points": 5, "outer_radius": 0.45, "inner_radius": 0.2, "center": center}
        elif family == "banana":
            return {"length": 0.4, "curvature": 0.5, "width": 0.15, "center": center}
        elif family == "crescent":
            return {"outer_radius": 0.4, "inner_radius": 0.25, "offset": 0.15, "center": center}
        elif family in ("random_convex", "random_concave"):
            return {"radius_range": (0.2, 0.45), "center": center, "rng": rng}
        return {}
