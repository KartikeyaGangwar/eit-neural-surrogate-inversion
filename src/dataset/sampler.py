"""
B-spline parameter sampler for dataset generation.
====================================================

Generates diverse theta samples across 8 shape families in the 64-D B-spline
parameter space for use in the FEM dataset generation pipeline:
  - circle
  - ellipse
  - rectangle
  - star
  - banana
  - crescent
  - random_convex
  - random_concave
  - random_bspline

No EIDORS dependency. Runs on Colab (for testing the sampler logic).
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

import config as cfg
from geometry.parameters import (
    control_points_to_theta,
    check_geometry_validity,
    analyze_concavity,
    sample_reference_geometry,
)
from geometry.shape_to_bspline import fit_bspline_to_boundary, validate_fit_quality
from geometry.bspline import compute_bspline_boundary
import geometry.shape_library as sl


ALL_SHAPE_FAMILIES = [
    "circle",
    "ellipse",
    "rectangle",
    "star",
    "banana",
    "crescent",
    "random_convex",
    "random_concave",
]


class BSplineSampler:
    """
    Samples B-spline parameters theta in R^64 across diverse shape families.
    """

    def __init__(
        self,
        domain_radius: float = cfg.DOMAIN_RADIUS,
        domain_margin: float = 0.05,
        seed: int = cfg.DEFAULT_SEED,
        max_rejection_attempts: int = 100,
    ) -> None:
        self.domain_radius = domain_radius
        self.domain_margin = domain_margin
        self.base_rng = np.random.default_rng(seed)
        self.max_attempts = max_rejection_attempts

    def sample_shape(
        self,
        shape_family: str = "balanced",
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Sample a B-spline geometry for a given shape family.

        Args:
            shape_family: One of ALL_SHAPE_FAMILIES, 'balanced', 'random_bspline', or 'reference'.
            rng:          Random generator.
            seed:         Optional seed.

        Returns:
            theta:    shape (64,)
            metadata: dict with shape_family, concavity analysis, B-spline fit quality.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        if rng is None:
            rng = self.base_rng

        if shape_family == "balanced":
            shape_family = str(rng.choice(ALL_SHAPE_FAMILIES))

        if shape_family == "reference":
            theta = sample_reference_geometry()
            bnd, _ = compute_bspline_boundary(theta, 256)
            conc = analyze_concavity(bnd)
            return theta, {
                "shape_family": "reference",
                "is_convex_before": conc["is_convex"],
                "is_convex_after": conc["is_convex"],
                "reflex_fraction_after": conc["reflex_fraction"],
                "fit_rms_error": 0.0,
                "fit_max_error": 0.0,
            }

        if shape_family == "random_bspline":
            return self._sample_random_bspline(rng)

        for attempt in range(self.max_attempts):
            # Center inside domain margin
            center_r = rng.uniform(0.0, 0.25)
            center_th = rng.uniform(0.0, 2.0 * np.pi)
            center = (float(center_r * np.cos(center_th)), float(center_r * np.sin(center_th)))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))

            if shape_family == "circle":
                r = float(rng.uniform(0.18, 0.45))
                pts, s_meta = sl.generate_circle(radius=r, center=center)
            elif shape_family == "ellipse":
                a = float(rng.uniform(0.25, 0.50))
                b = float(rng.uniform(0.15, max(0.16, a * 0.7)))
                pts, s_meta = sl.generate_ellipse(a=a, b=b, center=center, angle=angle)
            elif shape_family == "rectangle":
                w = float(rng.uniform(0.25, 0.55))
                h = float(rng.uniform(0.18, 0.45))
                pts, s_meta = sl.generate_rectangle(width=w, height=h, center=center, angle=angle)
            elif shape_family == "star":
                n_pts = int(rng.choice([5, 6]))
                r_out = float(rng.uniform(0.35, 0.50))
                r_in = float(rng.uniform(0.15, 0.25))
                pts, s_meta = sl.generate_star(n_points=n_pts, outer_radius=r_out, inner_radius=r_in, center=center, angle=angle)
            elif shape_family == "banana":
                length = float(rng.uniform(0.35, 0.52))
                curv = float(rng.uniform(0.35, 0.70))
                width = float(rng.uniform(0.12, 0.20))
                pts, s_meta = sl.generate_banana(length=length, curvature=curv, width=width, center=center, angle=angle)
            elif shape_family == "crescent":
                r_out = float(rng.uniform(0.35, 0.48))
                r_in = float(rng.uniform(0.20, 0.32))
                off = float(rng.uniform(0.12, 0.22))
                pts, s_meta = sl.generate_crescent(outer_radius=r_out, inner_radius=r_in, offset=off, center=center, angle=angle)
            elif shape_family == "random_convex":
                pts, s_meta = sl.generate_random_convex(center=center, rng=rng)
            elif shape_family == "random_concave":
                pts, s_meta = sl.generate_random_concave(center=center, rng=rng)
            else:
                raise ValueError(f"Unknown shape_family: {shape_family}")

            # Concavity BEFORE fitting
            conc_before = analyze_concavity(pts)

            # B-spline fitting
            theta = fit_bspline_to_boundary(pts)

            # Validity check
            valid, reason = check_geometry_validity(theta, domain_radius=self.domain_radius, margin=self.domain_margin)
            if not valid:
                continue

            # Fitting quality & concavity AFTER fitting
            fit_qual = validate_fit_quality(theta, pts)
            bnd_fitted, _ = compute_bspline_boundary(theta, n_samples=len(pts))
            conc_after = analyze_concavity(bnd_fitted)

            meta = {
                "shape_family": shape_family,
                "shape_params": s_meta["shape_params"],
                "is_convex_before": conc_before["is_convex"],
                "is_convex_after": conc_after["is_convex"],
                "n_reflex_after": conc_after["n_reflex"],
                "reflex_fraction_after": conc_after["reflex_fraction"],
                "min_turning_cross_after": conc_after["min_turning_cross"],
                "fit_mean_error": fit_qual["mean_error"],
                "fit_rms_error": fit_qual["rms_error"],
                "fit_max_error": fit_qual["max_error"],
                "attempt": attempt,
            }
            return theta, meta

        raise RuntimeError(
            f"Failed to generate valid {shape_family} after {self.max_attempts} attempts."
        )

    def sample_random_theta(
        self,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample theta uniformly across balanced shape families."""
        return self.sample_shape(shape_family="balanced", rng=rng, seed=seed)

    def _sample_random_bspline(
        self, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback method: perturb nominal circle control points directly."""
        radius = float(rng.uniform(0.20, 0.45))
        angles = np.linspace(0.0, 2.0 * np.pi, cfg.N_CONTROL, endpoint=False)
        cp_nominal = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

        for attempt in range(self.max_attempts):
            cp = cp_nominal + rng.normal(0.0, 0.05, size=cp_nominal.shape)
            theta = control_points_to_theta(cp)
            valid, _ = check_geometry_validity(theta, domain_radius=self.domain_radius, margin=self.domain_margin)
            if valid:
                bnd, _ = compute_bspline_boundary(theta, 256)
                conc = analyze_concavity(bnd)
                return theta, {
                    "shape_family": "random_bspline",
                    "base_radius": radius,
                    "is_convex_before": conc["is_convex"],
                    "is_convex_after": conc["is_convex"],
                    "reflex_fraction_after": conc["reflex_fraction"],
                    "fit_rms_error": 0.0,
                    "fit_max_error": 0.0,
                    "attempt": attempt,
                }
        raise RuntimeError("Failed to generate valid random_bspline.")

    def sample_batch(
        self,
        n: int,
        mode: str = "balanced",
        base_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Sample a batch of n theta vectors.

        Args:
            n:          Number of samples.
            mode:       'balanced', 'random_bspline', or specific shape family name.
            base_seed:  Deterministic base seed.

        Returns:
            thetas:   shape (n, 64)
            metas:    list of metadata dicts
        """
        thetas = []
        metas = []
        for i in range(n):
            seed_i = (base_seed + i) if base_seed is not None else None
            theta, meta = self.sample_shape(shape_family=mode, seed=seed_i)
            meta["sample_index"] = i
            meta["seed"] = seed_i
            thetas.append(theta)
            metas.append(meta)
        return np.stack(thetas, axis=0), metas
