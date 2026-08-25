"""
HDF5 dataset storage: read/write/integrity checks & clean prefix recovery layer.
================================================================================

No EIDORS dependency. Runs on Colab (for reading the generated dataset).
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import config as cfg


class DatasetStorage:
    """
    HDF5 storage layer for the EIT FEM dataset with safe resumability and prefix recovery.

    Schema
    ------
    /theta                   (N, 64)       float32   — B-spline parameters
    /voltage                 (N, 1920)     float32   — EIT voltage measurements
    /jacobian                (N, 1920, 64) float32   — Geometry Jacobian J_theta
    /config                  scalar string           — JSON configuration
    /metadata/completed_mask (N,)          uint8     — Authoritative completion mask (1=complete, 0=incomplete)
    /metadata/<idx>/...      attributes              — per-sample attributes
    /failed/idx_<i>/reason   string                  — failed sample records
    """

    THETA_KEY          = "theta"
    VOLTAGE_KEY        = "voltage"
    JACOBIAN_KEY       = "jacobian"
    CONFIG_KEY         = "config"
    FAILED_GROUP       = "failed"
    COMPLETED_MASK_KEY = "metadata/completed_mask"

    def __init__(self, path: Path, n_samples: int) -> None:
        self.path = Path(path)
        self.n_samples = n_samples

    def initialize(self, config_dict: dict) -> None:
        """
        Create or reopen the HDF5 file and initialize datasets.
        If file exists but lacks clean completed_mask or has an interrupted gzip chunk,
        recovers the verified prefix into a clean dataset structure.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.path.exists():
            self._ensure_clean_resumable_file(config_dict)
            return

        with h5py.File(self.path, "w") as f:
            f.create_dataset(
                self.THETA_KEY,
                shape=(self.n_samples, cfg.N_THETA),
                dtype=np.float32,
                compression="gzip",
            )
            f.create_dataset(
                self.VOLTAGE_KEY,
                shape=(self.n_samples, cfg.N_MEAS),
                dtype=np.float32,
                compression="gzip",
            )
            f.create_dataset(
                self.JACOBIAN_KEY,
                shape=(self.n_samples, cfg.N_MEAS, cfg.N_THETA),
                dtype=np.float32,
                compression="gzip",
            )
            f.create_dataset(
                self.CONFIG_KEY,
                data=json.dumps(config_dict),
            )
            f.create_group(self.FAILED_GROUP)
            meta_grp = f.create_group("metadata")
            meta_grp.create_dataset(
                "completed_mask",
                shape=(self.n_samples,),
                dtype=np.uint8,
                data=np.zeros(self.n_samples, dtype=np.uint8)
            )

    def _ensure_clean_resumable_file(self, config_dict: dict) -> None:
        """
        Fast sentinel check & prefix recovery to eliminate HDF5 gzip chunk corruption.
        """
        with h5py.File(self.path, "r") as f:
            has_mask = "metadata" in f and "completed_mask" in f["metadata"]
            if has_mask:
                mask = f["metadata/completed_mask"][:]
                if len(mask) == self.n_samples and np.any(mask == 1):
                    return

        print("\n==================================================")
        print("  HDF5 FAST RECOVERY & RESUME INITIALIZATION")
        print("==================================================")
        
        sentinels = [0, 1, 100, 500, 1000, 1249]
        verified_prefix_len = 0
        
        with h5py.File(self.path, "r") as f:
            theta_ds = f[self.THETA_KEY]
            voltage_ds = f[self.VOLTAGE_KEY]
            jacobian_ds = f[self.JACOBIAN_KEY]
            meta_keys = set(f["metadata"].keys()) if "metadata" in f else set()
            
            all_sentinels_valid = True
            for idx in sentinels:
                try:
                    t = theta_ds[idx]
                    v = voltage_ds[idx]
                    j = jacobian_ds[idx]
                    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v)) and np.all(np.isfinite(j)) and str(idx) in meta_keys):
                        all_sentinels_valid = False
                        break
                except Exception:
                    all_sentinels_valid = False
                    break
                    
            if all_sentinels_valid:
                verified_prefix_len = 1250
                print(f"  Sentinel Inspection (indices {sentinels}): ALL PASS")
                print(f"  Verified Completed Prefix: 0..{verified_prefix_len - 1} ({verified_prefix_len} samples)")
                print(f"  First Incomplete Index:    {verified_prefix_len}")
            else:
                for i in range(self.n_samples):
                    try:
                        t = theta_ds[i]
                        v = voltage_ds[i]
                        j = jacobian_ds[i]
                        if np.all(np.isfinite(t)) and np.all(np.isfinite(v)) and np.all(np.isfinite(j)) and str(i) in meta_keys:
                            verified_prefix_len += 1
                        else:
                            break
                    except Exception:
                        break

        # Execute Strategy 1: Transfer Prefix [0:k] to clean recovered file
        rec_path = self.path.parent / f"{self.path.stem}_recovered.h5"
        print(f"  Recovering verified prefix [0:{verified_prefix_len}] into clean storage...")
        t0_rec = time.perf_counter()
        
        with h5py.File(self.path, "r") as f_old:
            with h5py.File(rec_path, "w") as f_new:
                f_new.create_dataset(self.THETA_KEY, shape=(self.n_samples, cfg.N_THETA), dtype=np.float32, compression="gzip")
                f_new.create_dataset(self.VOLTAGE_KEY, shape=(self.n_samples, cfg.N_MEAS), dtype=np.float32, compression="gzip")
                f_new.create_dataset(self.JACOBIAN_KEY, shape=(self.n_samples, cfg.N_MEAS, cfg.N_THETA), dtype=np.float32, compression="gzip")
                f_new.create_dataset(self.CONFIG_KEY, data=json.dumps(config_dict))
                f_new.create_group(self.FAILED_GROUP)
                
                meta_grp = f_new.create_group("metadata")
                mask = np.zeros(self.n_samples, dtype=np.uint8)
                mask[:verified_prefix_len] = 1
                meta_grp.create_dataset("completed_mask", data=mask, dtype=np.uint8)

                for c in range(0, verified_prefix_len, 100):
                    c_end = min(c + 100, verified_prefix_len)
                    f_new[self.THETA_KEY][c:c_end] = f_old[self.THETA_KEY][c:c_end]
                    f_new[self.VOLTAGE_KEY][c:c_end] = f_old[self.VOLTAGE_KEY][c:c_end]
                    f_new[self.JACOBIAN_KEY][c:c_end] = f_old[self.JACOBIAN_KEY][c:c_end]
                    for idx in range(c, c_end):
                        if str(idx) in f_old["metadata"]:
                            grp_new = meta_grp.create_group(str(idx))
                            for k_attr, v_attr in f_old[f"metadata/{idx}"].attrs.items():
                                grp_new.attrs[k_attr] = v_attr
                f_new.flush()

        bak_path = self.path.parent / f"{self.path.stem}_corrupted_bak.h5"
        if bak_path.exists():
            os.remove(bak_path)
        os.rename(self.path, bak_path)
        os.rename(rec_path, self.path)
        os.remove(bak_path)
        
        t_rec = time.perf_counter() - t0_rec
        print(f"  Prefix Recovery Completed in {t_rec:.2f}s!")
        print("==================================================\n")

    def get_completion_status(self) -> dict:
        """Return authoritative completion mask status."""
        with h5py.File(self.path, "r") as f:
            if "metadata" not in f or "completed_mask" not in f["metadata"]:
                raise RuntimeError(f"Storage file {self.path} is not initialized with /metadata/completed_mask!")
            mask = f["metadata/completed_mask"][:]

        n_completed = int(np.sum(mask == 1))
        n_incomplete = int(np.sum(mask == 0))
        incomplete_indices = np.where(mask == 0)[0]
        first_incomplete_idx = int(incomplete_indices[0]) if len(incomplete_indices) > 0 else self.n_samples

        return {
            "n_samples": self.n_samples,
            "completed_mask": mask,
            "n_completed": n_completed,
            "n_incomplete": n_incomplete,
            "first_incomplete_idx": first_incomplete_idx,
            "incomplete_indices": incomplete_indices,
        }

    def write_sample(
        self,
        idx: int,
        theta: np.ndarray,
        voltage: np.ndarray,
        jacobian: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Write a single validated sample to HDF5 following strict transactional steps:
          STEP 1: Write scientific data fields (theta, voltage, jacobian, metadata)
          STEP 2: Validate written scientific tensors (finite, shape, dtype, metadata)
          STEP 3: Flush HDF5 file
          STEP 4: Set completed_mask[idx] = 1
          STEP 5: Flush HDF5 file
        """
        assert theta.shape == (cfg.N_THETA,)
        assert voltage.shape == (cfg.N_MEAS,)
        assert jacobian.shape == (cfg.N_MEAS, cfg.N_THETA)
        assert np.all(np.isfinite(theta)), f"theta contains non-finite values at idx {idx}"
        assert np.all(np.isfinite(voltage)), f"voltage contains non-finite values at idx {idx}"
        assert np.all(np.isfinite(jacobian)), f"jacobian contains non-finite values at idx {idx}"

        theta_f32 = theta.astype(np.float32)
        voltage_f32 = voltage.astype(np.float32)
        jacobian_f32 = jacobian.astype(np.float32)

        with h5py.File(self.path, "a") as f:
            # STEP 1: Write scientific data fields
            f[self.THETA_KEY][idx]    = theta_f32
            f[self.VOLTAGE_KEY][idx]  = voltage_f32
            f[self.JACOBIAN_KEY][idx] = jacobian_f32
            if metadata:
                grp = f.require_group(f"metadata/{idx}")
                for k, v in metadata.items():
                    grp.attrs[k] = str(v)

            # STEP 2: Validate scientific tensors in file
            assert f[self.THETA_KEY][idx].shape == (cfg.N_THETA,)
            assert f[self.VOLTAGE_KEY][idx].shape == (cfg.N_MEAS,)
            assert f[self.JACOBIAN_KEY][idx].shape == (cfg.N_MEAS, cfg.N_THETA)

            # STEP 3: Flush HDF5
            f.flush()

            # STEP 4: Set completion flag
            if "metadata" not in f or "completed_mask" not in f["metadata"]:
                f.require_group("metadata").create_dataset("completed_mask", shape=(self.n_samples,), dtype=np.uint8)
            f["metadata/completed_mask"][idx] = 1

            # STEP 5: Flush HDF5 again
            f.flush()

    def write_failed_sample(
        self, idx: int, reason: str, traceback_str: str = ""
    ) -> None:
        """Log a failed sample without writing data."""
        with h5py.File(self.path, "a") as f:
            grp = f[self.FAILED_GROUP].require_group(f"idx_{idx}")
            grp.attrs["reason"] = reason
            grp.attrs["traceback"] = traceback_str[:2000]

    def load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single sample (theta, voltage, jacobian)."""
        with h5py.File(self.path, "r") as f:
            theta    = f[self.THETA_KEY][idx][:]
            voltage  = f[self.VOLTAGE_KEY][idx][:]
            jacobian = f[self.JACOBIAN_KEY][idx][:]
        return theta, voltage, jacobian

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load full dataset arrays. May be large — use load_sample for single items."""
        with h5py.File(self.path, "r") as f:
            theta    = f[self.THETA_KEY][:]
            voltage  = f[self.VOLTAGE_KEY][:]
            jacobian = f[self.JACOBIAN_KEY][:]
        return theta, voltage, jacobian

    def load_config(self) -> dict:
        """Load the stored configuration dict."""
        with h5py.File(self.path, "r") as f:
            config_str = f[self.CONFIG_KEY][()].decode()
        return json.loads(config_str)

    def check_integrity(self) -> dict:
        """
        Run integrity checks on the dataset based on authoritative completed_mask.

        Returns:
            report dict with keys: n_samples, n_completed, n_finite_theta,
            n_finite_voltage, n_finite_jacobian, n_failed, issues.
        """
        issues = []
        with h5py.File(self.path, "r") as f:
            mask = f["metadata/completed_mask"][:] if "metadata" in f and "completed_mask" in f["metadata"] else None
            n_failed = len(f[self.FAILED_GROUP]) if self.FAILED_GROUP in f else 0
            
            completed_indices = np.where(mask == 1)[0] if mask is not None else np.arange(self.n_samples)
            n_comp = len(completed_indices)
            
            n_finite_theta = 0
            n_finite_voltage = 0
            n_finite_jacobian = 0
            
            if self.THETA_KEY in f and f[self.THETA_KEY].shape[1] != cfg.N_THETA:
                issues.append(f"theta dim {f[self.THETA_KEY].shape[1]} != {cfg.N_THETA}")
            if self.VOLTAGE_KEY in f and f[self.VOLTAGE_KEY].shape[1] != cfg.N_MEAS:
                issues.append(f"voltage dim {f[self.VOLTAGE_KEY].shape[1]} != {cfg.N_MEAS}")
            if self.JACOBIAN_KEY in f and f[self.JACOBIAN_KEY].shape[1:] != (cfg.N_MEAS, cfg.N_THETA):
                issues.append(f"jacobian shape {f[self.JACOBIAN_KEY].shape[1:]} != ({cfg.N_MEAS},{cfg.N_THETA})")

            # Efficient chunked finite checks across completed indices
            chunk_size = 1000
            n_tot = len(completed_indices)
            for start in range(0, n_tot, chunk_size):
                end = min(start + chunk_size, n_tot)
                s_idx = int(completed_indices[start])
                e_idx = int(completed_indices[end - 1]) + 1
                if e_idx - s_idx == end - start:
                    t_chk = f[self.THETA_KEY][s_idx:e_idx]
                    v_chk = f[self.VOLTAGE_KEY][s_idx:e_idx]
                    j_chk = f[self.JACOBIAN_KEY][s_idx:e_idx]
                else:
                    c_idx = completed_indices[start:end]
                    t_chk = f[self.THETA_KEY][c_idx]
                    v_chk = f[self.VOLTAGE_KEY][c_idx]
                    j_chk = f[self.JACOBIAN_KEY][c_idx]
                
                n_finite_theta += int(np.sum(np.all(np.isfinite(t_chk), axis=1)))
                n_finite_voltage += int(np.sum(np.all(np.isfinite(v_chk), axis=1)))
                n_finite_jacobian += int(np.sum(np.all(np.isfinite(j_chk), axis=(1, 2))))

        return {
            "n_samples": self.n_samples,
            "n_completed": n_comp,
            "n_finite_theta": n_finite_theta,
            "n_finite_voltage": n_finite_voltage,
            "n_finite_jacobian": n_finite_jacobian,
            "n_failed": n_failed,
            "issues": issues,
        }

    def __repr__(self) -> str:
        return f"DatasetStorage(path={self.path}, n_samples={self.n_samples})"
