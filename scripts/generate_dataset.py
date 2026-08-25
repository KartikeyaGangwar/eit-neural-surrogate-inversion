"""
Synthetic FEM Dataset Generator Script
======================================
Generates high-fidelity HDF5 datasets containing triples of:
  - 64-D B-spline shape parameters (theta)
  - 1920-D Complete Electrode Model boundary voltages (voltage)
  - 1920x64 physical geometric sensitivity Jacobians (jacobian)

Requires local GNU Octave + EIDORS 3.12 installation.
Includes high-precision execution timing and host CPU/worker profiling.

Usage Examples:
    # Standard generation (single-process subprocess backend)
    python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5

    # Fast persistent Octave session with in-memory model caching
    python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5 --optimized

    # Multi-worker parallel dataset generation
    python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5 --optimized --workers 4
"""

from __future__ import annotations
import sys
import os
import time
import argparse
from pathlib import Path

# Add project root and src to path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.utils.hardware import print_hardware_header, PrecisionTimer, format_time
from src.dataset.generator import DatasetGenerator
from src.dataset.sampler import BSplineSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chunked HDF5 dataset of (theta, voltage, J_theta) triples via EIDORS/Octave."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(repo_root / "data" / "dataset.h5"),
        help="Path to output HDF5 dataset file (default: data/dataset.h5)",
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=10000,
        help="Total number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=cfg.DATASET_CHUNK_SIZE,
        help=f"Number of samples per HDF5 write chunk / checkpoint (default: {cfg.DATASET_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=cfg.DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {cfg.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--eidors-path",
        type=str,
        default=None,
        help="Path to EIDORS startup.m directory (overrides config.EIDORS_PATH and EIDORS_PATH env var)",
    )
    parser.add_argument(
        "--mesh-maxsz",
        type=float,
        default=cfg.MESH_MAXSZ,
        help=f"FEM maximum element size in meters (default: {cfg.MESH_MAXSZ})",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Use PersistentOctaveSession with in-memory model caching and in-Octave BLAS chain rule reduction",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel worker processes for persistent generation (default: 2)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print verbose generation progress logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_hardware_header(
        phase_title=f"EIT FEM Dataset Generation ({args.n_samples} Samples)",
        device="cpu",
        phase_type="Multi-Worker CPU (Octave + EIDORS FEM)" if args.workers > 1 else "CPU (Octave + EIDORS FEM)",
        workers=args.workers,
    )
    print(f"  Target File:   {args.output}")
    print(f"  Chunk Size:    {args.chunk_size}")
    print(f"  Mesh Max Size: {args.mesh_maxsz} m")
    print(f"  Session Mode:  {'Optimized Persistent Octave' if args.optimized else 'Standard Subprocess'}")
    print("-" * 80)

    # Validate / build FEM backend
    try:
        if args.optimized:
            from src.physics.fem_backend_optimized import PersistentOctaveSession, FEMBackend as BaseFEMBackend
            print("[DatasetGenerator] Initializing Persistent Octave Session...")
            backend = BaseFEMBackend(
                eidors_path=args.eidors_path,
                mesh_maxsz=args.mesh_maxsz,
                verbose=args.verbose,
            )
            backend.build_model()
        else:
            from src.physics.fem_backend import FEMBackend
            print("[DatasetGenerator] Initializing Standard EIDORS/Octave Backend...")
            backend = FEMBackend(
                eidors_path=args.eidors_path,
                mesh_maxsz=args.mesh_maxsz,
                verbose=args.verbose,
            )
            backend.build_model()

    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize EIDORS / Octave backend:")
        print(f"  {exc}")
        print("\nNote: Dataset generation requires local GNU Octave and EIDORS 3.12 installed.")
        print("For ML training, evaluation, and inversion without EIDORS, use pretrained checkpoints.")
        sys.exit(1)

    # Initialize and execute generator
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = BSplineSampler(seed=args.seed)
    generator = DatasetGenerator(
        backend=backend,
        output_path=str(output_path),
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        sampler=sampler,
        verbose=args.verbose,
    )

    timer = PrecisionTimer(name=f"FEM Dataset Generation ({args.n_samples} Samples)", synchronize_cuda=False).start()
    generator.generate()
    timer.stop()

    print("\n" + "=" * 80)
    print(f"Dataset generation complete in: {format_time(timer.elapsed_wall)} (CPU Time: {timer.elapsed_cpu:.2f}s)")
    if args.n_samples > 0 and timer.elapsed_wall > 0:
        avg_ms = (timer.elapsed_wall / args.n_samples) * 1000.0
        print(f"Average FEM throughput:         {avg_ms:.2f} ms / sample ({args.n_samples / timer.elapsed_wall:.2f} samples/sec)")
    print(f"HDF5 dataset saved to:          {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
