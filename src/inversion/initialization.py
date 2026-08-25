"""
Initial theta generation for multi-start inversion.
===================================================

Generates deterministic initial guesses in 64-D B-spline space using
Latin Hypercube sampling or perturbing base shapes (circle, ellipse, etc.).

No polygon center or convex quadrilateral assumptions.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Tuple, Optional

import config as cfg
from geometry.parameters import check_geometry_validity, sample_reference_geometry
from dataset.sampler import BSplineSampler


class BSplineInitializer:
    """
    Generates initial theta vectors for multi-start inversion.
    """

    def __init__(self, seed: int = cfg.DEFAULT_SEED) -> None:
        self.seed = seed
        self.sampler = BSplineSampler(seed=seed)

    def generate_inits(
        self, n_restarts: int = cfg.N_RESTARTS, mode: str = "mixed"
    ) -> List[np.ndarray]:
        """
        Generate n_restarts initial theta vectors.

        Modes:
            'mixed':     1 circle, 1 reference geometry, (n-2) random perturbations
            'circle':    all near-circular with varying radius/center
            'random':    all random perturbations of nominal circle

        Args:
            n_restarts: Total number of restarts requested.
            mode:       'mixed', 'circle', or 'random'.

        Returns:
            List of valid theta arrays, each shape (64,).
        """
        inits: List[np.ndarray] = []

        if mode == "mixed":
            # Guess 0: Circle centered at origin
            theta_circ, _ = self.sampler.sample_shape(
                shape_family="circle", seed=self.seed
            )
            inits.append(theta_circ)

            # Guess 1: Reference geometry (if restarts >= 2)
            if n_restarts >= 2:
                inits.append(sample_reference_geometry())

            # Remaining guesses: random valid B-splines
            for i in range(len(inits), n_restarts):
                theta_rnd, _ = self.sampler.sample_random_theta(seed=self.seed + i * 10)
                inits.append(theta_rnd)
        else:
            for i in range(n_restarts):
                seed_i = self.seed + i * 10
                if mode == "circle":
                    theta_i, _ = self.sampler.sample_shape(
                        shape_family="circle", seed=seed_i
                    )
                else:
                    theta_i, _ = self.sampler.sample_random_theta(seed=seed_i)
                inits.append(theta_i)

        return inits[:n_restarts]
