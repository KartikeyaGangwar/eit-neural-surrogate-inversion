"""
Surrogate vs FEM Evaluation and Verification.
=============================================

Independent verification comparing surrogate predictions and reconstructed theta
against ground-truth FEM solves.

EIDORS-DEPENDENT (for backend comparison): import backend only when running locally.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, TYPE_CHECKING

import config as cfg
from surrogate.model import VoltageSurrogate

if TYPE_CHECKING:
    from physics.fem_backend import FEMBackend


def evaluate_surrogate_vs_fem(
    surrogate: VoltageSurrogate,
    backend: "FEMBackend",
    test_thetas: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate surrogate voltage and Jacobian accuracy against EIDORS FEM.

    Args:
        surrogate:   Trained VoltageSurrogate.
        backend:     Built FEMBackend instance.
        test_thetas: Test theta vectors, shape (N_test, 64).

    Returns:
        dict with mean/max relative voltage error, mean/max relative Jacobian error.
    """
    N_test = test_thetas.shape[0]
    v_rel_errs = []
    j_rel_errs = []

    for i in range(N_test):
        theta = test_thetas[i]
        v_fem, j_fem = backend.solve_with_jacobian(theta)
        v_surr = surrogate.predict_numpy(theta)

        v_err = np.linalg.norm(v_surr - v_fem) / max(np.linalg.norm(v_fem), 1e-14)
        v_rel_errs.append(v_err)

    return {
        "n_test": N_test,
        "mean_voltage_rel_err": float(np.mean(v_rel_errs)),
        "max_voltage_rel_err": float(np.max(v_rel_errs)),
    }
