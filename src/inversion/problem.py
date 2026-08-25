"""
Abstract Inversion Problem interface.
=====================================

Defines the mathematical problem interface consumed by LMSolver,
TrustRegionSolver, and MultiStartSolver:

    residual(theta) -> r  shape (1920,)
    jacobian(theta) -> J  shape (1920, 64)

    r = V_model(theta) - V_measured
    J = dV_model / dtheta

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import abc
import numpy as np
from typing import Tuple, Optional, Dict, Any

import config as cfg


class InversionProblem(abc.ABC):
    """
    Abstract base class for inverse problem implementations.

    Decouples the optimization algorithms (LM, Trust-Region, Multi-start)
    from the underlying model (Surrogate, Ensemble, or direct FEM).
    """

    def __init__(self) -> None:
        self.fem_call_count: int = 0

    @abc.abstractmethod
    def residual(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute measurement residual r(theta) = V_model(theta) - V_measured.

        Args:
            theta: shape (64,)

        Returns:
            r: shape (1920,)
        """

    @abc.abstractmethod
    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian J(theta) = dV_model / dtheta.

        Args:
            theta: shape (64,)

        Returns:
            J: shape (1920, 64)
        """

    def residual_and_jacobian(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residual and Jacobian in a single evaluation.

        Default implementation calls residual() and jacobian() sequentially.
        Subclasses may override for single-pass efficiency.
        """
        return self.residual(theta), self.jacobian(theta)

    def objective(self, theta: np.ndarray) -> float:
        """
        Compute scalar objective f(theta) = 0.5 * ||r(theta)||^2.

        Args:
            theta: shape (64,)

        Returns:
            f: float
        """
        r = self.residual(theta)
        return float(0.5 * np.sum(r ** 2))
