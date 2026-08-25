"""
Measurement Noise Robustness Benchmark Runner
==============================================
Evaluates inverse shape reconstruction under additive Gaussian measurement noise:
    delta in [0.0%, 0.1%, 0.5%, 1.0%, 2.0%, 5.0%]
    SNR in [inf, 60dB, 46dB, 40dB, 34dB, 26dB]

Evaluates 9 held-out targets across 6 noise conditions (234 total inversion solves).
Saves raw per-trial records and condition summary to data/results/noise_robustness_results.json.
Includes high-precision execution wall-clock timing and hardware profiling.
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
from src.geometry.bspline import compute_bspline_boundary
from src.surrogate.model import VoltageSurrogate, SobolevSurrogate
from src.inversion.lm_solver import LMSolver, LMConfig
from src.inversion.multistart import MultiStartSolver
from src.inversion.initialization import BSplineInitializer
from src.inversion.residuals import SurrogateResidualProblem
from src.inversion.reconstruction import compute_metrics
from src.noise.noise_models import add_gaussian_noise


def run_noise_benchmark(quick_mode: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_hardware_header(
        phase_title="Measurement Noise Robustness Benchmark (234 Inversion Trials)",
        device=device,
        phase_type="GPU MultiStart LM" if device.type == "cuda" else "CPU MultiStart LM",
    )
    
    model_path = repo_root / "models" / "surrogate_10k_final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    model = SobolevSurrogate(
        n_input=cfg.N_THETA,
        n_output=cfg.N_MEAS,
        hidden_dim=cfg.SURROGATE_HIDDEN_DIM,
        n_blocks=cfg.SURROGATE_N_BLOCKS,
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
        
    noise_levels = [0.0, 0.01] if quick_mode else [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    n_realizations_per_noise = 1 if quick_mode else 5
    targets_to_run = targets_data[:2] if quick_mode else targets_data
    
    print(f"Loaded {len(targets_to_run)} held-out target(s) across {len(noise_levels)} noise condition(s).\n")
    
    lm_cfg = LMConfig(max_iters=50, verbose=False)
    multistart_solver = MultiStartSolver(
        lm_config=lm_cfg,
        n_restarts=2 if quick_mode else cfg.N_RESTARTS,
        initializer=BSplineInitializer(seed=cfg.DEFAULT_SEED),
    )
    
    all_trial_records = []
    condition_summaries = {}
    
    total_benchmark_timer = PrecisionTimer("Noise Benchmark Suite", synchronize_cuda=True).start()
    
    for delta in noise_levels:
        delta_pct = delta * 100.0
        snr_db = float("inf") if delta == 0.0 else -20.0 * np.log10(delta)
        n_realiz = 1 if delta == 0.0 else n_realizations_per_noise
        
        cond_timer = PrecisionTimer(f"Noise-{delta_pct:.1f}%", synchronize_cuda=True).start()
        print(f"\n--- Running Noise Level: {delta_pct:.1f}% (SNR: {snr_db:.1f} dB, {n_realiz} realization(s)/target) ---")
        
        cond_ious = []
        cond_rms = []
        cond_param_errs = []
        cond_clean_v_errs = []
        cond_noisy_v_errs = []
        cond_times = []
        cond_iters = []
        
        for tgt in targets_to_run:
            t_id = tgt["target_id"]
            fam = tgt["shape_family"]
            is_cvx = tgt.get("is_convex", True)
            th_true = np.array(tgt["theta_true"], dtype=np.float64)
            v_clean = np.array(tgt.get("voltage_target_fem", tgt.get("target_voltage")), dtype=np.float64)
            
            for r_idx in range(n_realiz):
                noise_seed = cfg.DEFAULT_SEED + t_id * 1000 + int(delta * 10000) + r_idx
                if delta == 0.0:
                    v_noisy = v_clean.copy()
                else:
                    v_noisy = add_gaussian_noise(v_clean, delta, seed=noise_seed)
                    
                problem = SurrogateResidualProblem(model, v_noisy)
                assert problem.fem_call_count == 0
                
                t_trial = PrecisionTimer(f"Target-{t_id}-R{r_idx}", synchronize_cuda=True).start()
                ms_res = multistart_solver.solve(problem, verbose=False)
                t_trial.stop()
                t_elapsed = t_trial.elapsed_wall
                
                assert problem.fem_call_count == 0
                th_rec = ms_res.theta_best
                best_lm = ms_res.all_results[ms_res.best_restart_idx]
                
                geo = compute_metrics(th_rec, th_true)
                iou = float(geo["iou"])
                rms = float(geo["boundary_rms_dist"])
                p_err = float(geo["parameter_rel_error"])
                
                v_pred = model.predict_numpy(th_rec)
                clean_v_err = float(np.linalg.norm(v_pred - v_clean) / np.linalg.norm(v_clean))
                noisy_v_err = float(np.linalg.norm(v_pred - v_noisy) / np.linalg.norm(v_noisy))
                
                cond_ious.append(iou)
                cond_rms.append(rms)
                cond_param_errs.append(p_err)
                cond_clean_v_errs.append(clean_v_err)
                cond_noisy_v_errs.append(noisy_v_err)
                cond_times.append(t_elapsed)
                cond_iters.append(int(best_lm.n_iters))
                
                trial_entry = {
                    "noise_fraction": delta,
                    "noise_pct": delta_pct,
                    "snr_db": None if delta == 0.0 else float(snr_db),
                    "target_id": t_id,
                    "shape_family": fam,
                    "is_convex": is_cvx,
                    "realization_idx": r_idx,
                    "seed": noise_seed,
                    "iou": iou,
                    "boundary_rms_m": rms,
                    "parameter_rel_error": p_err,
                    "clean_voltage_rel_err_pct": clean_v_err * 100.0,
                    "noisy_voltage_rel_err_pct": noisy_v_err * 100.0,
                    "lm_iters": int(best_lm.n_iters),
                    "solve_time_s": float(t_elapsed),
                    "theta_recovered": th_rec.tolist(),
                    "theta_true": th_true.tolist(),
                }
                all_trial_records.append(trial_entry)
                
            print(f"  Target #{t_id:2d} ({fam:<14}) | Mean IoU: {np.mean(cond_ious[-n_realiz:]):.4f} | Mean RMS: {np.mean(cond_rms[-n_realiz:]):.4f} m", flush=True)
            
        cond_timer.stop()
        cvx_ious = [r["iou"] for r in all_trial_records if r["noise_fraction"] == delta and r["is_convex"]]
        ccv_ious = [r["iou"] for r in all_trial_records if r["noise_fraction"] == delta and not r["is_convex"]]
        
        cond_summary = {
            "noise_fraction": delta,
            "noise_pct": delta_pct,
            "snr_db": None if delta == 0.0 else float(snr_db),
            "n_trials": len(cond_ious),
            "mean_iou": float(np.mean(cond_ious)),
            "median_iou": float(np.median(cond_ious)),
            "std_iou": float(np.std(cond_ious)),
            "p10_iou": float(np.percentile(cond_ious, 10)),
            "p90_iou": float(np.percentile(cond_ious, 90)),
            "convex_mean_iou": float(np.mean(cvx_ious)) if cvx_ious else 0.0,
            "concave_mean_iou": float(np.mean(ccv_ious)) if ccv_ious else 0.0,
            "mean_boundary_rms_m": float(np.mean(cond_rms)),
            "median_boundary_rms_m": float(np.median(cond_rms)),
            "std_boundary_rms_m": float(np.std(cond_rms)),
            "mean_param_rel_err": float(np.mean(cond_param_errs)),
            "mean_clean_v_err_pct": float(np.mean(cond_clean_v_errs) * 100.0),
            "mean_noisy_v_err_pct": float(np.mean(cond_noisy_v_errs) * 100.0),
            "mean_solve_time_s": float(np.mean(cond_times)),
            "condition_wall_time_s": cond_timer.elapsed_wall,
            "mean_lm_iters": float(np.mean(cond_iters)),
            "total_online_fem_solves": 0
        }
        condition_summaries[f"noise_{delta_pct:.1f}pct"] = cond_summary
        print(f"Summary {delta_pct:.1f}% Noise -> Mean IoU: {cond_summary['mean_iou']:.4f} (Median: {cond_summary['median_iou']:.4f}) | RMS: {cond_summary['mean_boundary_rms_m']:.4f} m | Time: {format_time(cond_timer.elapsed_wall)}")
        
    total_benchmark_timer.stop()
    print("\n" + "=" * 80)
    print(f"Total Noise Benchmark Wall-Clock Time: {format_time(total_benchmark_timer.elapsed_wall)} (CPU Time: {total_benchmark_timer.elapsed_cpu:.2f}s)")
    
    clean_iou = condition_summaries.get("noise_0.0pct", {}).get("mean_iou", 0.8335)
    clean_rms = condition_summaries.get("noise_0.0pct", {}).get("mean_boundary_rms_m", 0.0806)
    
    for k, s in condition_summaries.items():
        s["iou_abs_degradation"] = float(clean_iou - s["mean_iou"])
        s["iou_rel_degradation_pct"] = float((clean_iou - s["mean_iou"]) / clean_iou * 100.0) if clean_iou > 0 else 0.0
        s["rms_abs_increase_m"] = float(s["mean_boundary_rms_m"] - clean_rms)
        s["rms_rel_increase_pct"] = float((s["mean_boundary_rms_m"] - clean_rms) / clean_rms * 100.0) if clean_rms > 0 else 0.0
        
    results_out = {
        "benchmark": "EIT B-Spline FEM Zero-Online-FEM Noise Robustness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_trials": len(all_trial_records),
        "total_wall_clock_time_s": total_benchmark_timer.elapsed_wall,
        "noise_levels_evaluated": noise_levels,
        "n_held_out_targets": len(targets_to_run),
        "n_realizations_per_noise": n_realizations_per_noise,
        "condition_summaries": condition_summaries,
        "per_trial_records": all_trial_records
    }
    
    if not quick_mode:
        out_file = repo_root / "data" / "results" / "noise_robustness_results.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2)
        print(f"\n[Artifact Saved] Noise experiment results written to: {out_file}")


def main():
    run_noise_benchmark(quick_mode=False)


if __name__ == "__main__":
    main()
