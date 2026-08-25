"""
Ensemble-aware Inversion and Uncertainty Monitoring
=====================================================
Wraps an EnsembleModel to perform zero-online-FEM inverse reconstruction
using the ensemble mean forward prediction and Jacobian while monitoring
epistemic disagreement.
"""

from __future__ import annotations
import sys
import os
import numpy as np
import torch
from typing import Tuple, Optional

import config as cfg
from src.inversion.objective import InversionProblem
from src.ensemble.model import EnsembleModel


class EnsembleInversionProblem(InversionProblem):
    """
    InversionProblem backed by an EnsembleModel.

    r(theta) = v_ensemble_mean(theta) - v_measured
    J(theta) = j_ensemble_mean(theta)
    """

    def __init__(
        self,
        ensemble: EnsembleModel,
        v_measured: np.ndarray,
    ) -> None:
        self.ensemble = ensemble
        self.v_measured = np.asarray(v_measured, dtype=np.float64).flatten()
        assert self.v_measured.shape == (cfg.N_MEAS,)
        self.device = self.ensemble.members[0].v_mean.device

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute measurement residual vector using the ensemble mean forward prediction.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Residual vector of shape (1920,).
        """
        t = torch.from_numpy(theta.astype(np.float32)).to(self.device)
        with torch.no_grad():
            v_mean, _ = self.ensemble.predict(t)
        return v_mean.detach().cpu().numpy().astype(np.float64) - self.v_measured

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute analytical sensitivity Jacobian using the ensemble mean Jacobian.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Sensitivity matrix of shape (1920, 64).
        """
        t = torch.from_numpy(theta.astype(np.float32)).to(self.device)
        with torch.no_grad():
            _, j_mean, _ = self.ensemble.predict_with_jacobian(t)
        return j_mean.detach().cpu().numpy().astype(np.float64)

    def residual_and_jacobian(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble mean residual vector and Jacobian matrix concurrently.
        
        Args:
            theta: Parameter vector of shape (64,).
            
        Returns:
            Tuple of (residual array shape (1920,), jacobian array shape (1920, 64)).
        """
        t = torch.from_numpy(theta.astype(np.float32)).to(self.device)
        with torch.no_grad():
            v_mean, j_mean, _ = self.ensemble.predict_with_jacobian(t)
        r = v_mean.detach().cpu().numpy().astype(np.float64) - self.v_measured
        J = j_mean.detach().cpu().numpy().astype(np.float64)
        return r, J
