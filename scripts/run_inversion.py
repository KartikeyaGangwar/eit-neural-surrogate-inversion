"""
Zero-Online-FEM Inverse Reconstruction Runner
=============================================
Performs multi-start Levenberg-Marquardt shape reconstruction on the 9 held-out targets
using purely the trained Sobolev surrogate model (zero online FEM forward solves).
Includes high-precision wall-clock timing and hardware profiling.
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.utils.hardware import print_hardware_header, PrecisionTimer, format_time
from src.geometry.bspline import compute_bspline_boundary, theta_to_control_points
from src.surrogate.model import VoltageSurrogate, SobolevSurrogate
from src.inversion.lm_solver import LMSolver, LMConfig
from src.inversion.multistart import MultiStartSolver
from src.inversion.initialization import BSplineInitializer
from src.inversion.residuals import SurrogateResidualProblem
from src.inversion.reconstruction import compute_metrics


def run_inversion_benchmark(quick_mode: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_hardware_header(
        phase_title="Zero-Online-FEM Inverse Reconstruction (9 Held-Out Targets)",
        device=device,
        phase_type="GPU Surrogate-Driven LM" if device.type == "cuda" else "CPU Surrogate-Driven LM",
    )
    
    model_path = repo_root / "models" / "surrogate_10k_final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    model = SobolevSurrogate(
        n_input=cfg.N_THETA,
        n_output=cfg.N_MEAS,
        hidden_dim=cfg.SURROGATE_HIDDEN_DIM,
        n_blocks=cfg.SURROGATE_N_BLOCKS
    ).to(device)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    targets_path = repo_root / "data" / "targets" / "held_out_9_targets.json"
    with open(targets_path) as f:
        targets_data = json.load(f)
        
    targets_to_run = targets_data[:2] if quick_mode else targets_data
    print(f"Loaded {len(targets_to_run)} held-out target(s). Beginning {cfg.N_RESTARTS}-start LM inversion...\n")
    
    lm_cfg = LMConfig(max_iters=50, verbose=False)
    multistart_solver = MultiStartSolver(
        lm_config=lm_cfg,
        n_restarts=2 if quick_mode else cfg.N_RESTARTS,
        initializer=BSplineInitializer(seed=cfg.DEFAULT_SEED),
    )
    
    ious = []
    rms_dists = []
    solve_times = []
    
    benchmark_timer = PrecisionTimer(name="Deterministic Inversion Benchmark", synchronize_cuda=True).start()
    
    for t_idx, tgt in enumerate(targets_to_run):
        t_id = tgt["target_id"]
        fam = tgt["shape_family"]
        is_cvx = tgt.get("is_convex", True)
        th_true = np.array(tgt["theta_true"], dtype=np.float64)
        v_target = np.array(tgt.get("voltage_target_fem", tgt.get("target_voltage")), dtype=np.float64)
        
        problem = SurrogateResidualProblem(model, v_target)
        assert problem.fem_call_count == 0
        
        target_timer = PrecisionTimer(name=f"Target-{t_id}", synchronize_cuda=True).start()
        ms_res = multistart_solver.solve(problem, verbose=False)
        target_timer.stop()
        
        elapsed = target_timer.elapsed_wall
        solve_times.append(elapsed)
        
        assert problem.fem_call_count == 0
        th_rec = ms_res.theta_best
        best_lm = ms_res.all_results[ms_res.best_restart_idx]
        
        geo_metrics = compute_metrics(th_rec, th_true)
        iou = float(geo_metrics["iou"])
        rms = float(geo_metrics["boundary_rms_dist"])
        
        ious.append(iou)
        rms_dists.append(rms)
        
        shape_type = "Convex" if is_cvx else "Concave"
        print(f"  Target #{t_id:2d} | {fam:<15} ({shape_type:<7}) | IoU: {iou:.4f} | RMS: {rms:.4f} m | Iters: {best_lm.n_iters:2d} | Solve Time: {elapsed:.2f}s")
        
    benchmark_timer.stop()
    
    print("\n" + "-" * 80)
    print(f"  Overall Mean IoU:          {np.mean(ious):.4f}")
    print(f"  Overall Median IoU:        {np.median(ious):.4f}")
    print(f"  Overall Mean Boundary RMS: {np.mean(rms_dists):.4f} m")
    print(f"  Average Solve Time:        {np.mean(solve_times):.2f} s / multi-start target")
    print(f"  Total Wall-Clock Time:     {format_time(benchmark_timer.elapsed_wall)} (CPU Time: {benchmark_timer.elapsed_cpu:.2f}s)")
    print(f"  Total Online FEM Solves:   0 (N_FEM^online = 0 strictly maintained)")
    print("=" * 80)


def main():
    run_inversion_benchmark(quick_mode=False)


if __name__ == "__main__":
    main()
