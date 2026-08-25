"""
Surrogate training pipeline.
============================

Trains VoltageSurrogate using Sobolev loss (voltage + Jacobian MSE) on an HDF5 dataset.
Includes AdamW optimizer, CosineAnnealingLR scheduler, checkpointing, and evaluation.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import config as cfg
from surrogate.model import (
    VoltageSurrogate,
    compute_surrogate_batch_jacobian,
    compute_surrogate_batch_normalized_jacobian,
    compute_sobolev_jvp_loss_term,
)
from dataset.storage import DatasetStorage


def train_surrogate(
    dataset_path: str,
    output_model_path: str,
    val_fraction: float = 0.1,
    train_steps: int = cfg.TRAIN_STEPS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = 1e-3,
    lambda_voltage: float = cfg.LAMBDA_VOLTAGE,
    lambda_jacobian: float = cfg.LAMBDA_JACOBIAN,
    device: Optional[str] = None,
    seed: int = cfg.DEFAULT_SEED,
    verbose: bool = True,
) -> VoltageSurrogate:
    """
    Train VoltageSurrogate on an HDF5 dataset with Sobolev loss.

    Args:
        dataset_path:       Path to HDF5 dataset (must contain theta, voltage, jacobian).
        output_model_path:  Path to save trained model checkpoint (.pt).
        val_fraction:       Validation data split fraction (default 0.1).
        train_steps:        Total optimizer steps (default 6000).
        batch_size:         Batch size (default 1024).
        lr:                 Initial learning rate.
        lambda_voltage:     Sobolev loss voltage weight.
        lambda_jacobian:    Sobolev loss Jacobian weight.
        device:             'cuda' or 'cpu'. Auto-detected if None.
        seed:               Random seed for data split and training.
        verbose:            If True, print progress logs.

    Returns:
        Trained VoltageSurrogate instance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[train_surrogate] Loading dataset: {dataset_path}")
        print(f"[train_surrogate] Selected Device: {device}")
        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[train_surrogate] GPU Hardware: {gpu_name} ({vram_total:.2f} GB total VRAM)")

    # Load dataset
    storage = DatasetStorage(Path(dataset_path), 0)
    theta_all, v_all, j_all = storage.load_all()

    N = theta_all.shape[0]
    n_val = int(N * val_fraction)
    n_train = N - n_val

    # Data splitting: if N=1000, use strict 800 train, 100 val, 100 test
    rng_split = np.random.RandomState(seed)
    indices = rng_split.permutation(N)

    if N == 1000:
        train_idx = indices[:800]
        val_idx   = indices[800:900]
        test_idx  = indices[900:1000]
        n_train, n_val, n_test = 800, 100, 100
    else:
        n_val = int(N * val_fraction)
        n_train = N - n_val
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        test_idx = np.array([])
        n_test = 0

    if verbose:
        print(f"[train_surrogate] Dataset Split: {n_train} Train | {n_val} Val | {n_test} Held-out Test")

    # Compute normalization statistics STRICTLY from training split
    v_train = v_all[train_idx]
    v_mean = np.mean(v_train, axis=0)
    v_std  = np.std(v_train,  axis=0)
    v_std  = np.maximum(v_std, 1e-6)

    # Normalize datasets
    v_train_norm = (v_all[train_idx] - v_mean) / v_std
    j_train_norm = j_all[train_idx] / v_std[np.newaxis, :, np.newaxis]

    v_val_norm   = (v_all[val_idx]   - v_mean) / v_std
    j_val_norm   = j_all[val_idx]   / v_std[np.newaxis, :, np.newaxis]

    # Convert to Tensors
    t_train = torch.from_numpy(theta_all[train_idx].astype(np.float32))
    v_train_t = torch.from_numpy(v_train_norm.astype(np.float32))
    j_train_t = torch.from_numpy(j_train_norm.astype(np.float32))

    t_val = torch.from_numpy(theta_all[val_idx].astype(np.float32)).to(device)
    v_val_t = torch.from_numpy(v_val_norm.astype(np.float32)).to(device)
    j_val_t = torch.from_numpy(j_val_norm.astype(np.float32)).to(device)

    eff_batch_size = min(batch_size, n_train)
    train_ds = TensorDataset(t_train, v_train_t, j_train_t)
    train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, drop_last=False)

    # Model setup
    model = VoltageSurrogate().to(device)
    model.set_normalization_stats(v_mean, v_std)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps)

    best_val_loss = float("inf")
    best_model_state = None

    history = {
        "step": [],
        "train_loss": [],
        "train_v_loss": [],
        "train_j_loss": [],
        "val_v_loss": [],
        "val_j_loss": [],
        "val_total_loss": [],
        "lr": [],
    }

    step = 0
    t0 = time.time()

    while step < train_steps:
        for b_theta, b_v, b_j in train_loader:
            if step >= train_steps:
                break

            b_theta = b_theta.to(device)
            b_v     = b_v.to(device)
            b_j     = b_j.to(device)

            model.train()
            optimizer.zero_grad()

            # --- Voltage loss (standard forward pass, same VRAM as always) ---
            b_v_pred = model(b_theta)
            v_loss = nn.functional.mse_loss(b_v_pred, b_v)

            # --- Sobolev Jacobian loss via randomized JVP (VRAM-safe on 4GB GPUs) ---
            # Replaces vmap+jacfwd which requires 64x tangent vectors simultaneously.
            # Samples ONE random direction per step, computes J_pred*d and J_true*d.
            # Expected loss is identical to full Frobenius MSE by isotropy of d.
            j_loss = compute_sobolev_jvp_loss_term(model, b_theta, b_j)

            loss = lambda_voltage * v_loss + lambda_jacobian * j_loss

            if step == 0 and verbose:
                print(f"[train_surrogate] Step 1 Tensor Shape Audit:")
                print(f"  b_v_pred: {b_v_pred.shape} | b_v_true: {b_v.shape}")
                print(f"  Sobolev JVP Loss (randomized, VRAM-safe): scalar")
                if device.startswith("cuda"):
                    print(f"  Step-0 VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            current_lr = scheduler.get_last_lr()[0]

            if step % 100 == 0 or step == train_steps:
                model.eval()
                with torch.no_grad():
                    val_v_pred = model(t_val)
                    # Use vmap+jacfwd with chunk_size=4 for exact Jacobian eval.
                    # torch.no_grad() eliminates the gradient-tracking overhead
                    # that causes OOM during training.
                    val_j_pred = compute_surrogate_batch_normalized_jacobian(
                        model, t_val, chunk_size=4
                    )
                    val_v_loss_val = nn.functional.mse_loss(val_v_pred, v_val_t).item()
                    val_j_loss_val = nn.functional.mse_loss(val_j_pred, j_val_t).item()
                    val_tot_val = lambda_voltage * val_v_loss_val + lambda_jacobian * val_j_loss_val

                if val_tot_val < best_val_loss:
                    best_val_loss = val_tot_val
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                history["step"].append(step)
                history["train_loss"].append(loss.item())
                history["train_v_loss"].append(v_loss.item())
                history["train_j_loss"].append(j_loss.item())
                history["val_v_loss"].append(val_v_loss_val)
                history["val_j_loss"].append(val_j_loss_val)
                history["val_total_loss"].append(val_tot_val)
                history["lr"].append(current_lr)

            if verbose and (step % 500 == 0 or step == train_steps):
                elapsed = time.time() - t0
                vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.startswith("cuda") else 0.0
                vram_str = f" | Peak VRAM: {vram_mb:.1f} MB" if device.startswith("cuda") else ""
                print(
                    f"Step {step}/{train_steps} [{elapsed:.1f}s] | "
                    f"Train Loss: {loss.item():.5f} (V: {v_loss.item():.5f}, J: {j_loss.item():.5f}) | "
                    f"Val V-Loss: {history['val_v_loss'][-1]:.5f}{vram_str}"
                )

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if verbose:
            print(f"[train_surrogate] Restored best validation model (Best Val Loss: {best_val_loss:.5f})")

    # Save checkpoint
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "v_mean": v_mean,
        "v_std": v_std,
        "history": history,
        "test_indices": test_idx.tolist(),
        "config": {
            "pipeline_version": "1.0.0-LOCKED",
            "n_input": cfg.N_THETA,
            "n_output": cfg.N_MEAS,
            "train_steps": train_steps,
            "lambda_voltage": lambda_voltage,
            "lambda_jacobian": lambda_jacobian,
            "val_fraction": val_fraction,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "seed": seed,
            "best_val_loss": best_val_loss,
        }
    }, out_path)

    if verbose:
        print(f"[train_surrogate] Saved self-contained checkpoint to {out_path}")

    model.eval()
    return model


def load_frozen_surrogate(
    checkpoint_path: str, device: Optional[str] = None
) -> VoltageSurrogate:
    """
    Load a trained VoltageSurrogate checkpoint in eval mode (frozen).

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device:          'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        VoltageSurrogate in eval mode with restored weights and normalization stats.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VoltageSurrogate().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_normalization_stats(ckpt["v_mean"], ckpt["v_std"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model
