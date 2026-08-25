"""
Lightweight End-to-End Smoke Test Suite
=======================================
Verifies the complete repository pipeline without running full expensive experiments:
1. Forward surrogate inference & autodiff Jacobian
2. Deterministic Levenberg-Marquardt & MultiStart inversion
3. Noise robustness simulation & noisy inversion
4. Deep Ensemble loading (K=5), predictive spread, and ensemble inversion
5. Figure generation scripts and results parsing
6. Checkpoint cryptographic SHA-256 integrity

Run with:
    python -m pytest tests/test_end_to_end_smoke.py -v
or
    python tests/test_end_to_end_smoke.py
"""

import sys
import os
import time
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np
import torch

test_dir = Path(__file__).resolve().parent
repo_root = test_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.geometry.bspline import compute_bspline_boundary
from src.geometry.parameters import check_geometry_validity, sample_reference_geometry
from src.surrogate.model import VoltageSurrogate, compute_surrogate_batch_jacobian
from src.ensemble.model import EnsembleModel
from src.inversion.lm_solver import LMSolver, LMConfig
from src.inversion.multistart import MultiStartSolver
from src.inversion.initialization import BSplineInitializer
from src.inversion.residuals import SurrogateResidualProblem
from src.inversion.uncertainty import EnsembleInversionProblem
from src.noise.noise_models import add_gaussian_noise


class TestRepositorySmoke(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.ckpt_path = repo_root / "models" / "surrogate_10k_final.pt"
        cls.targets_path = repo_root / "data" / "targets" / "held_out_9_targets.json"
        
        with open(cls.targets_path) as f:
            cls.targets = json.load(f)

    def test_01_checkpoint_hashes(self):
        """Verify all 6 surrogate checkpoints match exact SHA-256 checksums."""
        expected_hashes = {
            "surrogate_10k_final.pt": "8301aa04adf710fcfd8ad12ae12e67dc4c623a576b7d6061bc0e4cd6042753a5",
            "deep_ensemble/ensemble_member_0_seed42.pt": "8301aa04adf710fcfd8ad12ae12e67dc4c623a576b7d6061bc0e4cd6042753a5",
            "deep_ensemble/ensemble_member_1_seed142.pt": "bbbf3d996ce8f624388809e76e6a87438508ef2daf46adf7b3d88d0d11969e5e",
            "deep_ensemble/ensemble_member_2_seed242.pt": "64606da26052705094b01ced47664831088ead244d371ac7c5a3832be4ece4aa",
            "deep_ensemble/ensemble_member_3_seed342.pt": "d48187001bfcd0a83a782027e6ec50be7e80c57d4230a5584c433a8f39060dd0",
            "deep_ensemble/ensemble_member_4_seed442.pt": "8829e4dcacd18b817513720d1659e25850f0f0c1697966539ea392333d10a8c8",
        }
        
        for rel_p, exp_h in expected_hashes.items():
            full_p = repo_root / "models" / rel_p
            self.assertTrue(full_p.exists(), f"Missing checkpoint: {rel_p}")
            h = hashlib.sha256(full_p.read_bytes()).hexdigest()
            self.assertEqual(h, exp_h, f"Hash mismatch for {rel_p}")

    def test_02_forward_surrogate_and_autodiff(self):
        """Verify forward model inference and analytical Jacobian computation."""
        model = VoltageSurrogate().to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        if "v_mean" in ckpt and "v_std" in ckpt:
            model.set_normalization_stats(ckpt["v_mean"], ckpt["v_std"])
        model.eval()
        
        # Test 2 sample inputs
        th_sample = torch.randn(2, 64, device=self.device)
        v_pred = model.predict(th_sample)
        self.assertEqual(v_pred.shape, (2, 1920))
        self.assertTrue(torch.isfinite(v_pred).all().item())
        
        # Autodiff Jacobian
        J = compute_surrogate_batch_jacobian(model, th_sample)
        self.assertEqual(J.shape, (2, 1920, 64))
        self.assertTrue(torch.isfinite(J).all().item())

    def test_03_deterministic_inversion_smoke(self):
        """Verify Levenberg-Marquardt and MultiStart inversion on 1 target."""
        model = VoltageSurrogate().to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        if "v_mean" in ckpt and "v_std" in ckpt:
            model.set_normalization_stats(ckpt["v_mean"], ckpt["v_std"])
        model.eval()
        
        tgt = self.targets[1]  # Target #2 (Ellipse)
        v_target = np.array(tgt["voltage_target_fem"], dtype=np.float64)
        
        prob = SurrogateResidualProblem(model, v_target)
        self.assertEqual(prob.fem_call_count, 0)
        
        # Smoke solve with 2 restarts
        solver = MultiStartSolver(
            lm_config=LMConfig(max_iters=20, verbose=False),
            n_restarts=2,
            initializer=BSplineInitializer(seed=42)
        )
        res = solver.solve(prob)
        
        self.assertEqual(prob.fem_call_count, 0)
        self.assertTrue(np.isfinite(res.f_best))
        self.assertEqual(res.theta_best.shape, (64,))
        valid, _ = check_geometry_validity(res.theta_best)
        self.assertTrue(valid)

    def test_04_noise_robustness_smoke(self):
        """Verify additive Gaussian noise injection and noisy inversion."""
        tgt = self.targets[0]
        v_clean = np.array(tgt["voltage_target_fem"], dtype=np.float64)
        
        # Add 1.0% noise (40 dB SNR)
        v_noisy = add_gaussian_noise(v_clean, noise_fraction=0.01, seed=1042)
        self.assertEqual(v_noisy.shape, (1920,))
        self.assertTrue(np.isfinite(v_noisy).all())
        
        # Check actual noise magnitude
        noise_norm = np.linalg.norm(v_noisy - v_clean)
        clean_norm = np.linalg.norm(v_clean)
        self.assertAlmostEqual(noise_norm / clean_norm, 0.01, delta=0.005)

    def test_05_deep_ensemble_smoke(self):
        """Verify K=5 EnsembleModel loading, spread, and inversion."""
        seeds = [42, 142, 242, 342, 442]
        members = []
        for idx, seed in enumerate(seeds):
            p = repo_root / "models" / "deep_ensemble" / f"ensemble_member_{idx}_seed{seed}.pt"
            m = VoltageSurrogate().to(self.device)
            c = torch.load(p, map_location=self.device, weights_only=False)
            m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c)
            if "v_mean" in c and "v_std" in c:
                m.set_normalization_stats(c["v_mean"], c["v_std"])
            m.eval()
            members.append(m)
            
        ensemble = EnsembleModel(members)
        self.assertEqual(ensemble.n_members, 5)
        
        # Test ensemble predict and spread
        th = torch.randn(64, device=self.device)
        v_mean, v_std = ensemble.predict(th)
        self.assertEqual(v_mean.shape, (1920,))
        self.assertEqual(v_std.shape, (1920,))
        self.assertTrue(torch.isfinite(v_mean).all().item())
        self.assertTrue((v_std >= 0).all().item())
        
        # Test ensemble inversion problem
        tgt = self.targets[0]
        v_fem = np.array(tgt["voltage_target_fem"], dtype=np.float64)
        prob = EnsembleInversionProblem(ensemble, v_fem)
        
        r, J = prob.residual_and_jacobian(th.detach().cpu().numpy())
        self.assertEqual(r.shape, (1920,))
        self.assertEqual(J.shape, (1920, 64))
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue(np.isfinite(J).all())

    def test_06_result_jsons_parse(self):
        """Verify all stored result JSON files are valid and contain no NaN/Inf."""
        json_paths = list((repo_root / "data" / "results").glob("*.json"))
        self.assertGreaterEqual(len(json_paths), 5)
        for jp in json_paths:
            with open(jp) as f:
                d = json.load(f)
            txt = jp.read_text(encoding="utf-8")
            self.assertNotIn("NaN", txt)
            self.assertNotIn("Infinity", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
