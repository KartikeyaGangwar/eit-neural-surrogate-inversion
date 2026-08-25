"""
Noise Simulation Models for EIT Boundary Measurements
=====================================================
Provides additive Gaussian measurement noise functions matching standard clinical/industrial EIT SNR definitions.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def add_gaussian_noise(
    voltage: np.ndarray,
    noise_fraction: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise proportional to signal standard deviation:
        V_noisy = V + eta,  where eta ~ N(0, (noise_fraction * std(V))^2 * I)

    Args:
        voltage: Clean measurement voltage vector (shape (N_MEAS,)).
        noise_fraction: Noise level delta (e.g. 0.01 for 1.0% noise / 40 dB SNR).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Noisy voltage vector of the same shape.
    """
    if noise_fraction <= 0.0:
        return voltage.copy()

    rng = np.random.default_rng(seed)
    signal_std = float(np.std(voltage))
    noise_sigma = noise_fraction * signal_std
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=voltage.shape)
    return voltage + noise
