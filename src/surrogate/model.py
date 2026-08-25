"""
VoltageSurrogate: 64-D -> 1920-D ResNet surrogate model.
=========================================================

Maps B-spline parameters theta (64,) to EIT voltage measurements (1920,).
Trained with Sobolev supervision: voltage loss + Jacobian loss.

Architecture:
    encoder:    Linear(64, hidden_dim) + LayerNorm + SiLU
    res_blocks: N x ResBlock(hidden_dim)
    head:       Linear(hidden_dim, 1920)

Jacobian computation:
    torch.func.jacfwd (forward-mode AD) — exact, no FD approximation.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.func import jacfwd, jvp, vmap
from typing import Optional, Tuple

import config as cfg


class ResBlock(nn.Module):
    """
    Residual block: Linear + LayerNorm + SiLU + Linear + residual.

    Input:  (batch, hidden_dim)
    Output: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate residual block forward computation with skip connection.
        
        Args:
            x: Input tensor of shape (batch, hidden_dim).
            
        Returns:
            Output tensor of shape (batch, hidden_dim).
        """
        return self.act(x + self.net(x))


class VoltageSurrogate(nn.Module):
    """
    Surrogate model: theta (64,) -> voltage (1920,).

    Trained with Sobolev loss:
        L = lambda_V * MSE(V_pred_norm, V_true_norm)
          + lambda_J * MSE(J_pred_norm, J_true_norm)

    Input normalization:
        theta is used as-is (control points in [-1, 1] approximately).

    Output normalization (registered buffers, set during training):
        v_mean: (1920,)   mean of training voltages
        v_std:  (1920,)   std  of training voltages

    The surrogate predicts NORMALIZED voltages. Denormalization is applied
    before returning predictions in predict() mode.
    """

    def __init__(
        self,
        n_input: int = cfg.N_THETA,
        n_output: int = cfg.N_MEAS,
        hidden_dim: int = cfg.SURROGATE_HIDDEN_DIM,
        n_blocks: int = cfg.SURROGATE_N_BLOCKS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.encoder = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, n_output)

        # Normalization statistics (set via set_normalization_stats)
        self.register_buffer("v_mean", torch.zeros(n_output))
        self.register_buffer("v_std",  torch.ones(n_output))
        self._norm_set = False

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalization_stats(
        self, v_mean: np.ndarray, v_std: np.ndarray
    ) -> None:
        """
        Set voltage normalization statistics from training data.

        Args:
            v_mean: shape (1920,)
            v_std:  shape (1920,)  — must be > 0
        """
        assert v_mean.shape == (self.n_output,)
        assert v_std.shape  == (self.n_output,)
        self.v_mean.copy_(torch.from_numpy(v_mean.astype(np.float32)))
        self.v_std.copy_(torch.from_numpy(v_std.astype(np.float32)))
        self._norm_set = True

    def normalize_voltage(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize voltage: (v - mean) / std."""
        return (v - self.v_mean) / (self.v_std + 1e-8)

    def denormalize_voltage(self, v_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize voltage: v_norm * std + mean."""
        return v_norm * (self.v_std + 1e-8) + self.v_mean

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning NORMALIZED predicted voltage.

        Args:
            theta: shape (..., 64)

        Returns:
            v_norm_pred: shape (..., 1920) — normalized predictions
        """
        x = self.encoder(theta)
        x = self.res_blocks(x)
        return self.head(x)

    def predict(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Predict PHYSICAL (denormalized) voltage.

        Args:
            theta: shape (..., 64)

        Returns:
            voltage: shape (..., 1920) — in physical units
        """
        return self.denormalize_voltage(self.forward(theta))

    def predict_numpy(self, theta: np.ndarray) -> np.ndarray:
        """
        Predict physical voltage from numpy theta.

        Args:
            theta: shape (64,) or (B, 64)

        Returns:
            voltage: shape (1920,) or (B, 1920)
        """
        squeeze = theta.ndim == 1
        if squeeze:
            theta = theta[np.newaxis]
        t = torch.from_numpy(theta.astype(np.float32)).to(self.v_mean.device)
        with torch.no_grad():
            v = self.predict(t).cpu().numpy()
        return v[0] if squeeze else v


def compute_surrogate_jacobian(
    model: VoltageSurrogate,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Compute surrogate Jacobian dV/dtheta using forward-mode AD.

    Args:
        model: VoltageSurrogate (in eval mode, no dropout).
        theta: shape (64,)  — single theta vector.

    Returns:
        J: shape (1920, 64)  — Jacobian of denormalized voltage w.r.t. theta.
    """
    assert theta.dim() == 1 and theta.shape[0] == cfg.N_THETA
    if theta.device != model.v_mean.device:
        theta = theta.to(model.v_mean.device)

    def predict_fn(t: torch.Tensor) -> torch.Tensor:
        """Evaluate single unbatched physical prediction for forward-mode AD."""
        return model.predict(t.unsqueeze(0)).squeeze(0)

    J = jacfwd(predict_fn)(theta)  # (1920, 64)
    return J


def compute_surrogate_batch_jacobian(
    model: VoltageSurrogate,
    theta_batch: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Compute physical surrogate Jacobian d(V_phys)/dtheta for a batch of theta vectors.

    Uses vmap + jacfwd in chunks of size chunk_size to prevent CUDA VRAM OOM on 4GB GPUs.

    Args:
        model:       VoltageSurrogate instance.
        theta_batch: shape (B, 64)
        chunk_size:  Micro-batch size for vmap AD (default 8).

    Returns:
        J_batch: shape (B, 1920, 64) — physical Jacobian d(V_phys)/dtheta
    """
    assert theta_batch.dim() == 2 and theta_batch.shape[1] == cfg.N_THETA
    if theta_batch.device != model.v_mean.device:
        theta_batch = theta_batch.to(model.v_mean.device)

    def single_jac(t: torch.Tensor) -> torch.Tensor:
        """Compute single physical Jacobian matrix via forward-mode autodiff."""
        return jacfwd(lambda x: model.predict(x.unsqueeze(0)).squeeze(0))(t)

    N = theta_batch.shape[0]
    j_chunks = []
    for i in range(0, N, chunk_size):
        sub_batch = theta_batch[i : i + chunk_size]
        sub_j = vmap(single_jac)(sub_batch)
        j_chunks.append(sub_j)

    return torch.cat(j_chunks, dim=0)


def compute_surrogate_batch_normalized_jacobian(
    model: VoltageSurrogate,
    theta_batch: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Compute normalized surrogate Jacobian d(V_norm)/dtheta for Sobolev loss supervision.

    Uses vmap + jacfwd in chunks of size chunk_size to prevent CUDA VRAM OOM on 4GB GPUs.

    Calculates:
        d(V_norm)/dtheta = d(model.forward(theta))/dtheta

    This guarantees that predicted normalized Jacobian match normalized target Jacobian
    J_norm = J_phys / v_std without any 400x scaling mismatch.

    Args:
        model:       VoltageSurrogate instance.
        theta_batch: shape (B, 64)
        chunk_size:  Micro-batch size for vmap AD (default 8).

    Returns:
        J_norm_batch: shape (B, 1920, 64) — normalized Jacobian d(V_norm)/dtheta
    """
    assert theta_batch.dim() == 2 and theta_batch.shape[1] == cfg.N_THETA
    if theta_batch.device != model.v_mean.device:
        theta_batch = theta_batch.to(model.v_mean.device)

    def single_norm_jac(t: torch.Tensor) -> torch.Tensor:
        """Compute single normalized Jacobian matrix via forward-mode autodiff."""
        return jacfwd(lambda x: model(x.unsqueeze(0)).squeeze(0))(t)

    N = theta_batch.shape[0]
    j_chunks = []
    for i in range(0, N, chunk_size):
        sub_batch = theta_batch[i : i + chunk_size]
        sub_j = vmap(single_norm_jac)(sub_batch)
        j_chunks.append(sub_j)

    return torch.cat(j_chunks, dim=0)


def compute_sobolev_jvp_loss_term(
    model: VoltageSurrogate,
    theta_batch: torch.Tensor,
    j_true_norm_batch: torch.Tensor,
    directions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Randomized Sobolev Jacobian loss via single JVP per sample.

    Instead of computing the full (B, 1920, 64) Jacobian via vmap+jacfwd
    (which requires 64 tangent vectors per sample and OOMs on 4 GB GPUs),
    this function:
      1. Samples ONE random unit direction d in R^64 per batch step.
      2. Computes J_pred·d = jvp(model, theta, d)[1]  — shape (B, 1920).
      3. Computes J_true·d from the stored FEM Jacobian.
      4. Returns MSE(J_pred·d, J_true·d).

    This has the same EXPECTED gradient w.r.t. model parameters as the full
    Frobenius Jacobian MSE (by isotropy of the random direction), but requires
    only ~1/64 of the peak VRAM since it runs a single forward pass per sample.

    VRAM cost: equivalent to a standard forward pass — always safe on 4 GB.

    Args:
        model:              VoltageSurrogate in training mode.
        theta_batch:        shape (B, 64)
        j_true_norm_batch:  shape (B, 1920, 64) — normalized FEM Jacobians
        directions:         Optional (B, 64) or (64,) pre-computed unit directions.
                            If None, fresh random unit vectors are sampled.

    Returns:
        j_loss: scalar tensor — MSE between predicted and true Jacobian projections.
    """
    B = theta_batch.shape[0]
    device = theta_batch.device

    # Sample random unit direction d: shape (B, 64)
    if directions is None:
        d = torch.randn(B, cfg.N_THETA, device=device, dtype=theta_batch.dtype)
        d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-12)
    elif directions.dim() == 1:
        d = directions.unsqueeze(0).expand(B, -1)
        d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-12)
    else:
        d = directions
        d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-12)

    # Compute J_pred * d per sample via jvp — shape (B, 1920)
    # vmap(jvp) runs one JVP per batch element: O(forward_pass) VRAM
    def single_jvp(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute d(model.forward)/dtheta at t in direction v.
        t: (64,), v: (64,) -> output: (1920,)
        """
        _, jvp_val = jvp(
            lambda x: model(x.unsqueeze(0)).squeeze(0),
            (t,),
            (v,),
        )
        return jvp_val  # (1920,)

    j_pred_d = vmap(single_jvp)(theta_batch, d)  # (B, 1920)

    # Ground truth projection: J_true_norm @ d for each sample
    # j_true_norm_batch: (B, 1920, 64), d: (B, 64) -> (B, 1920)
    j_true_d = torch.einsum('bmd,bd->bm', j_true_norm_batch, d)  # (B, 1920)

    # MSE over all B*1920 elements
    return torch.mean((j_pred_d - j_true_d) ** 2)


# Canonical alias for Sobolev-regularized VoltageSurrogate
SobolevSurrogate = VoltageSurrogate


