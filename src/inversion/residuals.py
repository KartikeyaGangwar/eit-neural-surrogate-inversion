"""
Surrogate and Real Residual Providers.
======================================

Implements InversionProblem for:
  1. SurrogateResidualProvider (backed by VoltageSurrogate or EnsembleModel)
  2. FEMResidualProvider (backed by FEMBackend for validation)

NO artificial polygon regularization (perimeter, boundary, edge-uniformity) is added.

No EIDORS dependency at import time.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Tuple, Optional, Union

import config as cfg
from inversion.problem import InversionProblem
from surrogate.model import VoltageSurrogate, compute_surrogate_jacobian


class SurrogateResidualProblem(InversionProblem):
    """
    Inversion problem backed by a VoltageSurrogate neural network.

    r(theta) = V_surrogate(theta) - V_measured   (shape 1920,)
    J(theta) = dV_surrogate / dtheta              (shape 1920, 64)
    """

    def __init__(
        self,
        surrogate: VoltageSurrogate,
        v_measured: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            surrogate:  Trained VoltageSurrogate instance.
            v_measured: Target voltage measurements, shape (1920,).
            weights:    Optional observation weights, shape (1920,).
        """
        super().__init__()
        self.surrogate = surrogate
        self.v_measured = np.asarray(v_measured, dtype=np.float64).flatten()
        assert self.v_measured.shape == (cfg.N_MEAS,)
        self.weights = weights
        self.fem_call_count: int = 0  # Invariant: Surrogate evaluations do 0 FEM calls

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute measurement residual vector r(theta) = V_surrogate(theta) - V_measured.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Residual vector of shape (1920,).
        """
        v_pred = self.surrogate.predict_numpy(theta)  # (1920,)
        r = v_pred - self.v_measured
        if self.weights is not None:
            r = r * self.weights
        return r

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute analytical autodiff Jacobian J(theta) = dV_surrogate / dtheta.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Sensitivity matrix of shape (1920, 64).
        """
        t = torch.from_numpy(theta.astype(np.float32))
        with torch.no_grad():
            J = compute_surrogate_jacobian(self.surrogate, t).detach().cpu().numpy().astype(np.float64)
        if self.weights is not None:
            J = J * self.weights[:, np.newaxis]
        return J

    def residual_and_jacobian(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both residual vector and Jacobian matrix concurrently.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Tuple of (residual array shape (1920,), jacobian array shape (1920, 64)).
        """
        t = torch.from_numpy(theta.astype(np.float32))
        v_pred = self.surrogate.predict_numpy(theta)
        r = v_pred - self.v_measured
        with torch.no_grad():
            J = compute_surrogate_jacobian(self.surrogate, t).detach().cpu().numpy().astype(np.float64)
        if self.weights is not None:
            r = r * self.weights
            J = J * self.weights[:, np.newaxis]
        return r, J


class FEMResidualProblem(InversionProblem):
    """
    Inversion problem backed directly by EIDORS FEMBackend.
    Used for direct FEM inversion benchmarks / validation.
    """

    def __init__(
        self,
        fem_backend: Any,
        v_measured: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            fem_backend: Configured FEMBackend instance.
            v_measured: Target voltage measurements, shape (1920,).
            weights:    Optional observation weights, shape (1920,).
        """
        super().__init__()
        self.fem_backend = fem_backend
        self.v_measured = np.asarray(v_measured, dtype=np.float64).flatten()
        assert self.v_measured.shape == (cfg.N_MEAS,)
        self.weights = weights
        self.fem_call_count: int = 0

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute measurement residual using forward FEM solver.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Residual vector of shape (1920,).
        """
        self.fem_call_count += 1
        v_pred = self.fem_backend.solve(theta)
        r = v_pred - self.v_measured
        if self.weights is not None:
            r = r * self.weights
        return r

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute sensitivity Jacobian using adjoint FEM solver.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Sensitivity matrix of shape (1920, 64).
        """
        self.fem_call_count += 1
        J = self.fem_backend.jacobian(theta)
        if self.weights is not None:
            J = J * self.weights[:, np.newaxis]
        return J

    def residual_and_jacobian(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both FEM residual and adjoint Jacobian concurrently.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Tuple of (residual array shape (1920,), jacobian array shape (1920, 64)).
        """
        self.fem_call_count += 1
        v_pred, J = self.fem_backend.solve_with_jacobian(theta)
        r = v_pred - self.v_measured
        if self.weights is not None:
            r = r * self.weights
            J = J * self.weights[:, np.newaxis]
        return r, J

