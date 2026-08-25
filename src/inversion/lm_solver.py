"""
Levenberg-Marquardt Optimizer for 64-D B-spline EIT Inversion.
===============================================================

Solves:
    minimize  0.5 * ||r(theta)||^2
    subject to  check_geometry_validity(theta) == True

where:
    r(theta) = V_model(theta) - V_measured   (1920,)
    theta    in R^64

Gain-ratio adaptive damping (Nielsen / Marquardt scheme):
    (J^T J + lambda I) delta = -J^T r
    rho = actual_reduction / predicted_reduction

Feasibility guard:
    Rejects steps that violate domain containment or contain NaNs.
    Does NOT require convexity (concave shapes are valid).

No artificial polygon regularization. Pure measurement inverse problem.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import config as cfg
from inversion.problem import InversionProblem
from geometry.parameters import check_geometry_validity


@dataclass
class LMConfig:
    """Configuration for LMSolver."""
    max_iters: int = 100
    lambda_init: float = 1e-2
    lambda_max: float = 1e8
    lambda_min: float = 1e-12
    ftol: float = 1e-6          # relative change in objective
    gtol: float = 1e-6          # norm of gradient
    xtol: float = 1e-6          # norm of step delta
    nu: float = 2.0             # damping factor multiplier
    domain_radius: float = cfg.DOMAIN_RADIUS
    verbose: bool = False


@dataclass
class LMIterationRecord:
    """Record of a single LM iteration."""
    iter_idx: int
    objective: float
    grad_norm: float
    step_norm: float
    lambda_val: float
    rho: float
    accepted: bool
    reason: str


@dataclass
class LMResult:
    """Result of LM optimization run."""
    theta_opt: np.ndarray
    objective_opt: float
    n_iters: int
    converged: bool
    termination_reason: str
    history: List[LMIterationRecord] = field(default_factory=list)

    def summary(self) -> str:
        """
        Return a formatted human-readable summary string of the LM result.
        
        Returns:
            Multi-line string detailing convergence status, objective, and iterations.
        """
        return (
            f"LM Optimization Result:\n"
            f"  Converged:          {self.converged}\n"
            f"  Termination Reason: {self.termination_reason}\n"
            f"  Final Objective:    {self.objective_opt:.6e}\n"
            f"  Iterations:         {self.n_iters}\n"
        )


class LMSolver:
    """
    Levenberg-Marquardt solver for 64-D B-spline EIT reconstruction.
    """

    def __init__(self, config: Optional[LMConfig] = None) -> None:
        self.cfg = config or LMConfig()

    def solve(
        self, problem: InversionProblem, theta_0: np.ndarray
    ) -> LMResult:
        """
        Run LM optimization starting from initial guess theta_0.

        Args:
            problem: InversionProblem instance.
            theta_0: Initial B-spline parameter vector, shape (64,).

        Returns:
            LMResult dataclass with optimized theta and diagnostic history.
        """
        theta = np.asarray(theta_0, dtype=np.float64).copy()
        assert theta.shape == (cfg.N_THETA,), \
            f"Expected initial theta shape ({cfg.N_THETA},), got {theta.shape}"

        # Initial feasibility check
        valid, reason = check_geometry_validity(theta, domain_radius=self.cfg.domain_radius)
        if not valid:
            raise ValueError(f"Initial theta_0 is invalid: {reason}")

        lambda_val = self.cfg.lambda_init
        nu = self.cfg.nu
        history: List[LMIterationRecord] = []

        r, J = problem.residual_and_jacobian(theta)
        f = float(0.5 * np.sum(r ** 2))
        g = J.T @ r
        grad_norm = float(np.linalg.norm(g))

        converged = False
        term_reason = "max_iters_reached"

        for k in range(self.cfg.max_iters):
            if grad_norm < self.cfg.gtol:
                converged = True
                term_reason = f"gtol_reached (||g||={grad_norm:.2e} < {self.cfg.gtol})"
                break

            # Solve (J^T J + lambda I) delta = -g
            JTJ = J.T @ J
            A = JTJ + lambda_val * np.eye(cfg.N_THETA)
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, -g, rcond=None)[0]

            step_norm = float(np.linalg.norm(delta))
            if step_norm < self.cfg.xtol:
                converged = True
                term_reason = f"xtol_reached (||delta||={step_norm:.2e} < {self.cfg.xtol})"
                break

            theta_trial = theta + delta

            # Feasibility guard
            valid, reason = check_geometry_validity(
                theta_trial, domain_radius=self.cfg.domain_radius
            )
            if not valid:
                # Reject step and increase damping
                lambda_val *= nu
                nu *= 2.0
                history.append(LMIterationRecord(
                    iter_idx=k, objective=f, grad_norm=grad_norm,
                    step_norm=step_norm, lambda_val=lambda_val, rho=-1.0,
                    accepted=False, reason=f"geometry_invalid: {reason}"
                ))
                continue

            # Evaluate trial step
            f_trial = problem.objective(theta_trial)

            # Gain ratio calculation
            # predicted_reduction = 0.5 * delta^T (lambda delta - g)
            predicted_red = float(0.5 * np.dot(delta, lambda_val * delta - g))
            actual_red = f - f_trial
            rho = actual_red / (predicted_red + 1e-14) if predicted_red > 0 else -1.0

            if rho > 0:
                # Step accepted
                rel_f_change = abs(f - f_trial) / max(f, 1e-14)
                theta = theta_trial
                f = f_trial
                r, J = problem.residual_and_jacobian(theta)
                g = J.T @ r
                grad_norm = float(np.linalg.norm(g))

                # Update damping (decrease lambda)
                lambda_val = max(
                    self.cfg.lambda_min,
                    lambda_val * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3),
                )
                nu = self.cfg.nu

                history.append(LMIterationRecord(
                    iter_idx=k, objective=f, grad_norm=grad_norm,
                    step_norm=step_norm, lambda_val=lambda_val, rho=rho,
                    accepted=True, reason="accepted"
                ))

                if rel_f_change < self.cfg.ftol:
                    converged = True
                    term_reason = f"ftol_reached (rel_f={rel_f_change:.2e} < {self.cfg.ftol})"
                    break
            else:
                # Step rejected (increase damping)
                lambda_val = min(self.cfg.lambda_max, lambda_val * nu)
                nu *= 2.0

                history.append(LMIterationRecord(
                    iter_idx=k, objective=f, grad_norm=grad_norm,
                    step_norm=step_norm, lambda_val=lambda_val, rho=rho,
                    accepted=False, reason="rho_negative"
                ))

                if lambda_val >= self.cfg.lambda_max:
                    term_reason = "lambda_max_reached"
                    break

        return LMResult(
            theta_opt=theta,
            objective_opt=f,
            n_iters=len(history),
            converged=converged,
            termination_reason=term_reason,
            history=history,
        )


# Canonical alias for Levenberg-Marquardt solver
LevenbergMarquardtSolver = LMSolver
