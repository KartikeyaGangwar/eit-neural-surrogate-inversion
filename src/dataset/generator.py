"""
Chunked, resumable FEM dataset generator.
==========================================

Generates the HDF5 dataset of (theta, voltage, J_theta) triples
using the local EIDORS/FEM backend.

EIDORS-DEPENDENT: run on local Windows machine only.
The generated dataset.h5 can then be uploaded to Colab for ML training.

DO NOT EXECUTE until validation stages A-D have passed.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Dict, Any

import config as cfg
from dataset.sampler import BSplineSampler
from dataset.storage import DatasetStorage
from geometry.parameters import check_geometry_validity


class DatasetGenerator:
    """
    Chunked, resumable EIT FEM dataset generator.

    Writes (theta, voltage, J_theta) triples to HDF5 in chunks.
    Failed samples are logged to /failed group with error reason.
    Checkpoints after each chunk for crash-safe resumption.

    HDF5 schema written by this generator:
        /theta      (N, 64)       float32
        /voltage    (N, 1920)     float32
        /jacobian   (N, 1920, 64) float32
        /metadata   (N,)          structured dtype
        /config     JSON string (scalar dataset)
        /failed/    subgroup with failed sample records

    Example usage (LOCAL only — requires EIDORS)::

        from physics.fem_backend import FEMBackend
        from dataset.generator import DatasetGenerator

        backend = FEMBackend()
        backend.build_model()

        gen = DatasetGenerator(
            backend=backend,
            output_path='dataset.h5',
            n_samples=10000,
            chunk_size=100,
            seed=42,
        )
        gen.generate()
    """

    def __init__(
        self,
        backend,  # FEMBackend — imported lazily to avoid EIDORS at import time
        output_path: str,
        n_samples: int,
        chunk_size: int = cfg.DATASET_CHUNK_SIZE,
        seed: int = cfg.DEFAULT_SEED,
        sampler: Optional[BSplineSampler] = None,
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.output_path = Path(output_path)
        self.n_samples = n_samples
        self.chunk_size = chunk_size
        self.seed = seed
        self.verbose = verbose
        self.sampler = sampler or BSplineSampler(seed=seed)

        # Checkpoint file tracks how many samples have been written
        self.checkpoint_path = self.output_path.with_suffix(".checkpoint.json")

    def generate(self) -> None:
        """
        Generate n_samples (theta, voltage, J_theta) triples and write to HDF5.

        Resumes from last checkpoint if the output file already exists.
        """
        start_index = self._load_checkpoint()
        if start_index >= self.n_samples:
            self._log("Dataset already complete. Nothing to do.")
            return

        storage = DatasetStorage(self.output_path, self.n_samples)
        storage.initialize(self._build_config_dict())

        self._log(f"Generating {self.n_samples} samples (resuming from {start_index}).")
        self._log(f"Output: {self.output_path}")

        n_written = start_index
        n_failed = 0

        for chunk_start in range(start_index, self.n_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.n_samples)
            chunk_indices = list(range(chunk_start, chunk_end))
            self._log(f"  Chunk [{chunk_start}:{chunk_end}] ...")

            for idx in chunk_indices:
                sample_seed = self.seed * 10_000 + idx
                try:
                    theta, meta = self.sampler.sample_random_theta(seed=sample_seed)
                    meta["sample_index"] = idx
                    meta["seed"] = sample_seed

                    # FEM solve + Jacobian
                    t0 = time.perf_counter()
                    voltage, J_theta = self.backend.solve_with_jacobian(theta)
                    elapsed = time.perf_counter() - t0

                    meta["generation_time_s"] = elapsed
                    meta["source_type"] = cfg.DATASET_SOURCE_SYNTHETIC

                    storage.write_sample(
                        idx=idx,
                        theta=theta,
                        voltage=voltage,
                        jacobian=J_theta,
                        metadata=meta,
                    )
                    n_written += 1
                except Exception as exc:
                    n_failed += 1
                    self._log(f"  [FAILED] sample {idx}: {exc}")
                    storage.write_failed_sample(
                        idx=idx,
                        reason=str(exc),
                        traceback_str=traceback.format_exc(),
                    )

            # Checkpoint after each chunk
            self._save_checkpoint(chunk_end)
            self._log(f"  Written: {n_written}, Failed: {n_failed}")

        self._log("=" * 60)
        self._log(f"Dataset generation complete.")
        self._log(f"  Total written:  {n_written}")
        self._log(f"  Total failed:   {n_failed}")
        self._log(f"  Output:         {self.output_path}")

    def _build_config_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
            "n_theta": cfg.N_THETA,
            "n_meas": cfg.N_MEAS,
            "sigma_bg": cfg.SIGMA_BG,
            "sigma_inc": cfg.SIGMA_INC,
            "alpha": cfg.ALPHA,
            "mollifier_eps": cfg.MOLLIFIER_EPS,
            "n_boundary_samp": cfg.N_BOUNDARY_SAMP,
            "mesh_maxsz": cfg.MESH_MAXSZ,
            "domain_radius": cfg.DOMAIN_RADIUS,
            "n_elec": cfg.N_ELEC,
            "source_type": cfg.DATASET_SOURCE_SYNTHETIC,
        }

    def _load_checkpoint(self) -> int:
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            idx = int(data.get("completed_through", 0))
            self._log(f"Resuming from checkpoint: completed_through = {idx}")
            return idx
        return 0

    def _save_checkpoint(self, completed_through: int) -> None:
        with open(self.checkpoint_path, "w") as f:
            json.dump({"completed_through": completed_through}, f)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DatasetGenerator] {msg}")
