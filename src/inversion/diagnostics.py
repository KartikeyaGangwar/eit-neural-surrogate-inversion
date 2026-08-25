"""
LM diagnostics and figure generation.
=====================================

Provides iteration logging, gradient diagnostics, and visual diagnostic tools
for LM inversion runs.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, Optional, List

from inversion.lm_solver import LMResult
import config as cfg


def log_lm_history(result: LMResult, logger: Optional[Any] = None) -> None:
    """Print or log structured summary of LM solver history."""
    summary = result.summary()
    if logger is not None and hasattr(logger, "info"):
        logger.info(summary)
    else:
        print(summary)


def gradient_diagnostics(
    problem: Any, theta: np.ndarray
) -> Dict[str, float]:
    """
    Compute gradient norm and conditioning diagnostics.

    Args:
        problem: InversionProblem instance.
        theta:   shape (64,)

    Returns:
        dict with grad_norm, j_cond, j_max_singular, j_min_singular
    """
    r, J = problem.residual_and_jacobian(theta)
    g = J.T @ r
    grad_norm = float(np.linalg.norm(g))

    s = np.linalg.svd(J, compute_uv=False)
    j_max = float(s[0]) if len(s) > 0 else 0.0
    j_min = float(s[-1]) if len(s) > 0 else 0.0
    cond = j_max / max(j_min, 1e-14)

    return {
        "grad_norm": grad_norm,
        "j_max_singular": j_max,
        "j_min_singular": j_min,
        "j_cond": cond,
    }
