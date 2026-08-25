"""
Sobolev loss function for surrogate training.
=============================================

L_Sobolev = lambda_V * MSE(V_pred_norm, V_true_norm)
          + lambda_J * MSE(J_pred_norm, J_true_norm)

where:
    V_pred_norm  = (V_pred - v_mean) / v_std
    V_true_norm  = (V_true - v_mean) / v_std
    J_pred_norm  = d(V_pred_norm) / d(theta)   shape (B, 1920, 64)
    J_true_norm  = J_true / v_std[:, None]     shape (B, 1920, 64)

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

import config as cfg


class SobolevLoss(nn.Module):
    """
    Sobolev loss: combines normalized voltage MSE and normalized Jacobian MSE.

    L_total = lambda_voltage * L_V + lambda_jacobian * L_J

    Production Default:
        lambda_voltage = 1.0, lambda_jacobian = 0.01 (fixed, deterministic)
    """

    def __init__(
        self,
        lambda_voltage: float = cfg.LAMBDA_VOLTAGE,
        lambda_jacobian: float = cfg.LAMBDA_JACOBIAN,
        adaptive: bool = False,
        min_w_j: float = 0.1,
        max_w_j: float = 0.9,
    ) -> None:
        super().__init__()
        self.lambda_voltage = lambda_voltage
        self.lambda_jacobian = lambda_jacobian
        self.adaptive = adaptive
        self.min_w_j = min_w_j
        self.max_w_j = max_w_j
        self.mse = nn.MSELoss()

    def forward(
        self,
        v_pred_norm: torch.Tensor,
        v_true_norm: torch.Tensor,
        j_pred_norm: torch.Tensor,
        j_true_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total Sobolev loss.

        Args:
            v_pred_norm: (B, 1920)
            v_true_norm: (B, 1920)
            j_pred_norm: (B, 1920, 64)
            j_true_norm: (B, 1920, 64)

        Returns:
            (total_loss, voltage_loss, jacobian_loss)
        """
        assert v_pred_norm.shape == v_true_norm.shape, f"Shape mismatch: {v_pred_norm.shape} vs {v_true_norm.shape}"
        assert j_pred_norm.shape == j_true_norm.shape, f"Shape mismatch: {j_pred_norm.shape} vs {j_true_norm.shape}"

        v_loss = self.mse(v_pred_norm, v_true_norm)
        j_loss = self.mse(j_pred_norm, j_true_norm)

        if self.adaptive:
            # Bounded adaptive gradient ratio weighting
            v_val = v_loss.detach().item() + 1e-8
            j_val = j_loss.detach().item() + 1e-8
            ratio = j_val / (v_val + j_val)
            w_j = float(np.clip(ratio, self.min_w_j, self.max_w_j))
            w_v = 1.0 - w_j
            total_loss = w_v * v_loss + w_j * j_loss
        else:
            total_loss = self.lambda_voltage * v_loss + self.lambda_jacobian * j_loss

        return total_loss, v_loss, j_loss

