"""
Parameter space scaling utilities (theta_physical <-> theta_scaled).
=====================================================================

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

import config as cfg


class ParameterScaler:
    """
    Scales theta between physical coordinates in [-1, 1] and normalized [-1, 1].
    """

    def __init__(self, bounds: Tuple[float, float] = (-cfg.DOMAIN_RADIUS, cfg.DOMAIN_RADIUS)) -> None:
        self.min_val, self.max_val = bounds

    def scale(self, theta_physical: np.ndarray) -> np.ndarray:
        """Physical -> [-1, 1] range."""
        return 2.0 * (theta_physical - self.min_val) / (self.max_val - self.min_val) - 1.0

    def unscale(self, theta_scaled: np.ndarray) -> np.ndarray:
        """[-1, 1] -> Physical range."""
        return 0.5 * (theta_scaled + 1.0) * (self.max_val - self.min_val) + self.min_val
