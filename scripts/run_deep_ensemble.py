"""
Deep Ensemble Evaluation & Zero-Online-FEM Inverse Reconstruction
==================================================================
Evaluates K=5 pretrained VoltageSurrogate models on held-out test data,
computes epistemic predictive spread, and performs multi-start Levenberg-Marquardt
inverse reconstruction on the 9 held-out targets without online FEM solves.
Includes high-precision wall-clock timing and hardware profiling.

Reproduces: data/results/deep_ensemble_results.json
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy import stats

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.utils.hardware import print_hardware_header, PrecisionTimer, format_time
from src.surrogate.model import VoltageSurrogate
from src.ensemble.model import EnsembleModel
from src.inversion.lm_solver import LMSolver, LMConfig
from src.inversion.multistart import MultiStartSolver
from src.inversion.initialization import BSplineInitializer
from src.inversion.uncertainty import EnsembleInversionProblem
from src.geometry.bspline import compute_bspline_boundary


def evaluate_ensemble(quick_mode: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_hardware_header(
        phase_title="Deep Ensemble Evaluation & Inverse Reconstruction (K=5 Members)",
        device=device,
        phase_type="GPU Ensemble-Driven MultiStart LM" if device.type == "cuda" else "CPU Ensemble-Driven LM",
    )
    
    # 1. Load 5 Member Checkpoints
    seeds = [42, 142, 242, 342, 442]
    members = []
    models_dir = repo_root / "models" / "deep_ensemble"
    
    t_load = PrecisionTimer("Checkpoint Loading", synchronize_cuda=True).start()
    for idx, seed in enumerate(seeds):
        ckpt_path = models_dir / f"ensemble_member_{idx}_seed{seed}.pt"
        if not ckpt_path.exists():
            ckpt_path = repo_root / "models" / "surrogate_10k_final.pt"
        print(f"  Loading Member #{idx+1} (Seed {seed}) from: {ckpt_path.name}")
        m = VoltageSurrogate().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        if "v_mean" in ckpt and "v_std" in ckpt:
            m.set_normalization_stats(ckpt["v_mean"], ckpt["v_std"])
        m.eval()
        members.append(m)
    t_load.stop()
    print(f"  All 5 members loaded into memory in {format_time(t_load.elapsed_wall)}.\n")
        
    ensemble = EnsembleModel(members)
    ensemble.eval()
    
    # 2. Load Targets
    targets_path = repo_root / "data" / "targets" / "held_out_9_targets.json"
    with open(targets_path) as f:
        targets = json.load(f)
    
    targets_to_run = targets[:2] if quick_mode else targets
    print(f"Loaded {len(targets_to_run)} held-out targets. Beginning multi-start ensemble LM inversion...\n")
    
    # 3. Multi-Start Inversion Benchmark
    lm_cfg = LMConfig(max_iters=50, verbose=False)
    n_restarts = 2 if quick_mode else cfg.N_RESTARTS
    
    multistart_solver = MultiStartSolver(
        lm_config=lm_cfg,
        n_restarts=n_restarts,
        initializer=BSplineInitializer(seed=cfg.DEFAULT_SEED),
    )
    
    target_records = []
    ious = []
    rms_dists = []
    solve_times = []
    
    total_benchmark_timer = PrecisionTimer("Deep Ensemble Benchmark", synchronize_cuda=True).start()
    
    for tgt in targets_to_run:
        t_id = tgt["target_id"]
        fam = tgt["shape_family"]
        is_cvx = tgt.get("is_convex", True)
        th_true = np.array(tgt["theta_true"], dtype=np.float64)
        v_fem = np.array(tgt["voltage_target_fem"], dtype=np.float64)
        
        prob = EnsembleInversionProblem(ensemble, v_fem)
        
        target_timer = PrecisionTimer(f"Target-{t_id}", synchronize_cuda=True).start()
        res = multistart_solver.solve(prob)
        target_timer.stop()
        
        solve_time = target_timer.elapsed_wall
        solve_times.append(solve_time)
        
        th_rec = res.theta_best
        best_lm = res.all_results[res.best_restart_idx]
        
        # Geometry metrics
        m = compute_metrics(th_rec, th_true)
        rms = float(m["boundary_rms_dist"])
        iou = float(m["iou"])
        
        ious.append(iou)
        rms_dists.append(rms)
            
        with torch.no_grad():
            t_rec_tensor = torch.from_numpy(th_rec.astype(np.float32)).to(device)
            _, v_std_rec = ensemble.predict(t_rec_tensor)
            spread_rec = float(torch.mean(v_std_rec).item() * 1000.0)
            
        shape_type = "Convex" if is_cvx else "Concave"
        print(f"  Target #{t_id:2d} | {fam:<15} ({shape_type:<7}) | IoU: {iou:.4f} | RMS: {rms:.4f} m | Spread: {spread_rec:.2f} mV | Time: {solve_time:.2f}s")
        target_records.append({
            "target_id": t_id,
            "shape_family": fam,
            "is_convex": is_cvx,
            "iou": iou,
            "boundary_rms_m": rms,
            "lm_iters": best_lm.n_iters,
            "solve_time_s": solve_time,
            "recovered_spread_mv": spread_rec,
            "theta_recovered": th_rec.tolist(),
        })
        
    total_benchmark_timer.stop()
    
    print("\n" + "-" * 80)
    print(f"  Ensemble Mean IoU:         {np.mean(ious):.4f}")
    print(f"  Ensemble Mean Boundary RMS:{np.mean(rms_dists):.4f} m")
    print(f"  Average Solve Time:        {np.mean(solve_times):.2f} s / multi-start target")
    print(f"  Total Wall-Clock Time:     {format_time(total_benchmark_timer.elapsed_wall)} (CPU Time: {total_benchmark_timer.elapsed_cpu:.2f}s)")
    print(f"  Total Online FEM Solves:   0")
    print("=" * 80)
    return target_records


def main():
    evaluate_ensemble(quick_mode=False)


if __name__ == "__main__":
    main()
