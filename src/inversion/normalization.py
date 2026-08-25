"""
Voltage measurement normalization utilities.
============================================

Ensures identical normalization conventions are applied to both surrogate predictions
and real/synthetic measurement vectors before entering optimization.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class VoltageNormalizer:
    """
    Normalizes EIT voltage vectors: v_norm = (v - mean) / std.
    """

    def __init__(self, v_mean: np.ndarray, v_std: np.ndarray) -> None:
        self.v_mean = np.asarray(v_mean, dtype=np.float64).flatten()
        self.v_std  = np.asarray(v_std,  dtype=np.float64).flatten()
        assert self.v_mean.shape == (1920,)
        assert self.v_std.shape  == (1920,)

    def normalize(self, v: np.ndarray) -> np.ndarray:
        """
        Normalize physical voltages to zero-mean, unit-variance representation.
        
        Args:
            v: Physical boundary voltage vector of shape (1920,).
            
        Returns:
            Normalized voltage vector of shape (1920,).
        """
        return (v - self.v_mean) / (self.v_std + 1e-8)

    def denormalize(self, v_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize unit-variance voltages back to physical volts/millivolts.
        
        Args:
            v_norm: Normalized voltage vector of shape (1920,).
            
        Returns:
            Physical voltage vector of shape (1920,).
        """
        return v_norm * (self.v_std + 1e-8) + self.v_mean
