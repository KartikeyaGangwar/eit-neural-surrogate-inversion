"""
Two-Stage Sobolev Surrogate Training Pipeline
=============================================
Executes the production two-stage training protocol:
- Stage 1: AdamW stochastic optimization (6,000 to 20,000 steps, cosine annealing lr 1e-3 -> 1e-5, B=512)
- Stage 2: Deterministic full-batch L-BFGS refinement (100 outer iterations, strong Wolfe line search)

Supervises boundary voltages and directional Jacobian-Vector Products (JVPs) with lambda_J = 0.01.
Includes precision execution timing and host hardware profiling.
"""

from __future__ import annotations
import sys
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.utils.hardware import print_hardware_header, PrecisionTimer, format_time
from src.surrogate.model import VoltageSurrogate, SobolevSurrogate
from src.surrogate.training import train_surrogate


def train_surrogate_pipeline(
    data_path: str = None,
    output_dir: str = None,
    train_steps: int = cfg.TRAIN_STEPS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = 1e-3,
    seed: int = cfg.DEFAULT_SEED,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_hardware_header(
        phase_title="Sobolev Surrogate Neural Network Training",
        device=device,
        phase_type="GPU Surrogate Training (Sobolev JVP)" if device.type == "cuda" else "CPU Surrogate Training",
    )
    
    output_dir = Path(output_dir) if output_dir else repo_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = output_dir / "surrogate_10k_final.pt"
    
    if data_path is None or not Path(data_path).exists():
        print(f"[NOTE] Dataset file not found at: {data_path}")
        print(f"To generate the dataset from scratch via EIDORS/Octave, run:")
        print(f"  python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5")
        print(f"\nA verified pretrained production model is already available at: models/surrogate_10k_final.pt")
        return
        
    print(f"Loading training data from: {data_path}")
    print(f"Optimizer: AdamW (Steps: {train_steps}, Batch Size: {batch_size}, LR: {lr})")
    print(f"Loss: Sobolev Regularized (lambda_V = {cfg.LAMBDA_VOLTAGE}, lambda_J = {cfg.LAMBDA_JACOBIAN})")
    
    timer = PrecisionTimer("Surrogate Training", synchronize_cuda=True).start()
    model = train_surrogate(
        dataset_path=data_path,
        output_model_path=str(out_model_path),
        train_steps=train_steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=str(device),
        verbose=True,
    )
    timer.stop()
    
    print("\n" + "=" * 80)
    print(f"Surrogate training completed in: {format_time(timer.elapsed_wall)} (CPU Time: {timer.elapsed_cpu:.2f}s)")
    print(f"Saved model checkpoint to: {out_model_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train Sobolev Surrogate Model")
    parser.add_argument("--dataset", "--data-path", type=str, default=str(repo_root / "data" / "dataset.h5"), help="Path to HDF5 dataset")
    parser.add_argument("--output", "--output-dir", type=str, default=str(repo_root / "models"), help="Directory to save model checkpoint")
    parser.add_argument("--steps", type=int, default=cfg.TRAIN_STEPS, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()
    
    train_surrogate_pipeline(
        data_path=args.dataset,
        output_dir=args.output,
        train_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
