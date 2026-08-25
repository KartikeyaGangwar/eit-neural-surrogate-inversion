"""
Multi-start Inversion Solver.
=============================

Runs LMSolver (or TrustRegionSolver) from multiple initial guesses theta_0
and ranks candidates by physical residual norm.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import config as cfg
from inversion.problem import InversionProblem
from inversion.lm_solver import LMSolver, LMConfig, LMResult
from inversion.initialization import BSplineInitializer


@dataclass
class MultiStartResult:
    """Multi-start optimization result container."""
    theta_best: np.ndarray
    f_best: float
    best_restart_idx: int
    all_results: List[LMResult]

    def summary(self) -> str:
        """
        Return a formatted human-readable summary of the multi-start optimization.
        
        Returns:
            Multi-line string detailing the best restart, final objective, and all attempts.
        """
        s = f"Multi-Start Optimization ({len(self.all_results)} restarts):\n"
        s += f"  Best Restart: #{self.best_restart_idx+1}\n"
        s += f"  Best Objective: {self.f_best:.6e}\n"
        for i, res in enumerate(self.all_results):
            status = "CONVERGED" if res.converged else "FAILED"
            s += f"  Restart #{i+1}: f={res.objective_opt:.6e}, iters={res.n_iters} [{status}]\n"
        return s


class MultiStartSolver:
    """
    Multi-start solver wrapping LMSolver over N initial guesses.
    """

    def __init__(
        self,
        lm_config: Optional[LMConfig] = None,
        n_restarts: int = cfg.N_RESTARTS,
        initializer: Optional[BSplineInitializer] = None,
    ) -> None:
        self.lm_solver = LMSolver(lm_config)
        self.n_restarts = n_restarts
        self.initializer = initializer or BSplineInitializer()

    def solve(
        self,
        problem: InversionProblem,
        inits: Optional[List[np.ndarray]] = None,
        verbose: bool = False,
    ) -> MultiStartResult:
        """
        Run multi-start inversion.

        Args:
            problem: InversionProblem instance.
            inits:   Optional explicit list of initial theta guesses.
                     If None, generated via self.initializer.
            verbose: If True, print progress logs per restart.

        Returns:
            MultiStartResult dataclass.
        """
        if inits is None:
            inits = self.initializer.generate_inits(n_restarts=self.n_restarts)

        all_results: List[LMResult] = []
        best_f = np.inf
        best_idx = 0
        best_theta = inits[0]

        for i, theta_0 in enumerate(inits):
            if verbose:
                print(f"[MultiStart] Running Restart #{i+1}/{len(inits)} ...")

            try:
                res = self.lm_solver.solve(problem, theta_0)
                all_results.append(res)
                if res.objective_opt < best_f:
                    best_f = res.objective_opt
                    best_theta = res.theta_opt
                    best_idx = len(all_results) - 1
            except Exception as exc:
                if verbose:
                    print(f"  Restart #{i+1} failed: {exc}")

        return MultiStartResult(
            theta_best=best_theta,
            f_best=best_f,
            best_restart_idx=best_idx,
            all_results=all_results,
        )
