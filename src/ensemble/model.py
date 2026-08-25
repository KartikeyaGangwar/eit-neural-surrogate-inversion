"""
Deep Ensemble Container Class for VoltageSurrogate Models
=========================================================
Aggregates N independently trained VoltageSurrogate models to provide
predictive mean, inter-member standard deviation (epistemic uncertainty),
and ensemble mean Jacobian autodiff evaluation.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

import config as cfg
from src.surrogate.model import VoltageSurrogate, compute_surrogate_batch_jacobian


class EnsembleModel(nn.Module):
    """
    Deep Ensemble of K VoltageSurrogate members.
    """

    def __init__(self, members: List[VoltageSurrogate]) -> None:
        super().__init__()
        assert len(members) > 0, "Ensemble must have at least 1 member"
        self.members = nn.ModuleList(members)
        self.n_members = len(members)

    def predict_members(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all ensemble members.

        Args:
            theta: shape (B, 64) or (64,)

        Returns:
            v_members: shape (K, B, 1920) or (K, 1920)
        """
        squeeze = theta.ndim == 1
        if squeeze:
            theta = theta.unsqueeze(0)

        preds = [m.predict(theta) for m in self.members]
        v_members = torch.stack(preds, dim=0)

        return v_members.squeeze(1) if squeeze else v_members

    def predict(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ensemble mean prediction and member standard deviation.

        Args:
            theta: shape (B, 64) or (64,)

        Returns:
            (v_mean, v_std): mean voltage prediction and inter-member standard deviation.
        """
        v_members = self.predict_members(theta)
        v_mean = torch.mean(v_members, dim=0)
        v_std = torch.std(v_members, dim=0)
        return v_mean, v_std

    def predict_with_jacobian(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ensemble mean voltage, mean Jacobian, and uncertainty stats.

        Args:
            theta: shape (64,) or (B, 64)

        Returns:
            v_mean:      (1920,) or (B, 1920)
            j_mean:      (1920, 64) or (B, 1920, 64)
            uncertainty: dict with 'v_std', 'j_std', 'disagreement_norm'
        """
        squeeze = theta.ndim == 1
        if squeeze:
            theta = theta.unsqueeze(0)

        v_preds = []
        j_preds = []

        for m in self.members:
            v_preds.append(m.predict(theta))
            j_preds.append(compute_surrogate_batch_jacobian(m, theta))

        v_stack = torch.stack(v_preds, dim=0)
        j_stack = torch.stack(j_preds, dim=0)

        v_mean = torch.mean(v_stack, dim=0)
        v_std = torch.std(v_stack, dim=0)

        j_mean = torch.mean(j_stack, dim=0)
        j_std = torch.std(j_stack, dim=0)

        disagreement = torch.norm(v_std, dim=-1)

        uncertainty = {
            "v_std": v_std.squeeze(0) if squeeze else v_std,
            "j_std": j_std.squeeze(0) if squeeze else j_std,
            "disagreement_norm": disagreement.squeeze(0) if squeeze else disagreement,
        }

        if squeeze:
            return v_mean.squeeze(0), j_mean.squeeze(0), uncertainty
        return v_mean, j_mean, uncertainty
