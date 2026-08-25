"""
Bounded Trust-Region Alternative Solver.
=========================================

Provides a trust-region optimization method for the same InversionProblem interface,
serving as an alternative to LMSolver for comparison experiments.

Solves:
    minimize 0.5 * ||r(theta)||^2
    subject to ||delta|| <= Delta, check_geometry_validity(theta+delta) == True

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

import config as cfg
from inversion.problem import InversionProblem
from geometry.parameters import check_geometry_validity


@dataclass
class TrustRegionConfig:
    """
    Configuration parameters for bounded trust-region solver.
    
    Attributes:
        max_iters: Maximum trust-region iterations.
        delta_init: Initial trust-region radius.
        delta_max: Maximum allowed trust-region radius.
        ftol: Relative objective reduction tolerance.
        gtol: Gradient infinity-norm convergence tolerance.
        domain_radius: Outer boundary radius of circular domain.
    """
    max_iters: int = 100
    delta_init: float = 0.1
    delta_max: float = 0.5
    ftol: float = 1e-6
    gtol: float = 1e-6
    domain_radius: float = cfg.DOMAIN_RADIUS


class TrustRegionSolver:
    """Trust-region solver alternative for 64-D B-spline inversion."""

    def __init__(self, config: Optional[TrustRegionConfig] = None) -> None:
        self.cfg = config or TrustRegionConfig()

    def solve(
        self, problem: InversionProblem, theta_0: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, str]:
        """
        Solve inversion problem using Powell's dogleg trust-region method.

        Args:
            problem: InversionProblem instance.
            theta_0: Initial theta guess (64,).

        Returns:
            (theta_opt, f_opt, converged, reason)
        """
        theta = np.asarray(theta_0, dtype=np.float64).copy()
        delta_tr = self.cfg.delta_init
        converged = False
        reason = "max_iters_reached"

        for k in range(self.cfg.max_iters):
            r, J = problem.residual_and_jacobian(theta)
            f = float(0.5 * np.sum(r ** 2))
            g = J.T @ r
            if np.linalg.norm(g) < self.cfg.gtol:
                converged = True
                reason = "gtol_reached"
                break

            # Gauss-Newton step
            JTJ = J.T @ J
            try:
                p_gn = np.linalg.solve(JTJ + 1e-8 * np.eye(cfg.N_THETA), -g)
            except np.linalg.LinAlgError:
                p_gn = -g

            gn_norm = np.linalg.norm(p_gn)
            if gn_norm <= delta_tr:
                step = p_gn
            else:
                step = (delta_tr / gn_norm) * p_gn

            theta_trial = theta + step
            valid, _ = check_geometry_validity(theta_trial, domain_radius=self.cfg.domain_radius)
            if not valid:
                delta_tr *= 0.5
                continue

            f_trial = problem.objective(theta_trial)
            actual_red = f - f_trial
            pred_red = float(-np.dot(g, step) - 0.5 * np.dot(step, JTJ @ step))

            rho = actual_red / (pred_red + 1e-14) if pred_red > 0 else -1.0

            if rho > 0.25:
                theta = theta_trial
                if rho > 0.75 and gn_norm >= 0.9 * delta_tr:
                    delta_tr = min(self.cfg.delta_max, 2.0 * delta_tr)
                if abs(f - f_trial) / max(f, 1e-14) < self.cfg.ftol:
                    converged = True
                    reason = "ftol_reached"
                    break
            else:
                delta_tr *= 0.5

        return theta, float(problem.objective(theta)), converged, reason
