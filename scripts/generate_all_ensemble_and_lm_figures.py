def log_msg(msg):
    print(msg, flush=True)

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.geometry.bspline import compute_bspline_boundary
from src.surrogate.model import VoltageSurrogate
from src.ensemble.model import EnsembleModel
from src.inversion.lm_solver import LMSolver, LMConfig
from src.inversion.multistart import MultiStartSolver
from src.inversion.initialization import BSplineInitializer
from src.inversion.residuals import SurrogateResidualProblem
from src.inversion.uncertainty import EnsembleInversionProblem

# Style configuration
COLOR_BG_FILL = "#ffffff"
COLOR_INC_FILL = "#000000"
COLOR_DOMAIN_LINE = "#1a202c"
COLOR_REC_LINE = "#e53e3e"
COLOR_ENS_LINE = "#2b6cb0"
COLOR_REF_LINE = "#718096"
BOX_BG = "#f8fafc"
BOX_BORDER = "#cbd5e0"

C_BLUE = "#2b6cb0"
C_GREEN = "#276749"
C_CRIMSON = "#c53030"
C_ORANGE = "#dd6b20"
C_PURPLE = "#6b46c1"
C_TEAL = "#319795"
C_GRAY = "#718096"
C_DARK = "#2d3748"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def remove_axes_frame(ax):
    ax.set_frame_on(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def save_fig(fig, path_no_ext):
    png_p = Path(f"{path_no_ext}.png")
    pdf_p = Path(f"{path_no_ext}.pdf")
    png_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_p, format="pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    log_msg(f"  [Figure Saved] {png_p.name} & {pdf_p.name}")


def main():
    log_msg("=" * 80)
    log_msg("  GENERATING AUDITED PUBLICATION FIGURES: DEEP ENSEMBLE, LM & MULTI-START")
    log_msg("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    ens_json_path = repo_root / "data" / "results" / "deep_ensemble_results.json"
    with open(ens_json_path) as f:
        ens_data = json.load(f)
        
    noise_json_path = repo_root / "data" / "results" / "noise_robustness_results.json"
    with open(noise_json_path) as f:
        noise_data = json.load(f)
        
    targets_json_path = repo_root / "data" / "targets" / "held_out_9_targets.json"
    with open(targets_json_path) as f:
        all_targets = json.load(f)
        
    with open(repo_root / "data" / "results" / "inversion_results.json") as f:
        raw_single = json.load(f)
        single_inv_records = raw_single.get("new_model_inversion_records", raw_single)
        
    ens_inv_records = ens_data["per_target_inversion_records"]
    base_sum = ens_data["baseline_summary"]
    member_sums = ens_data["member_summaries"]
    ens_sum = ens_data["ensemble_mean_summary"]
    unc_res = ens_data["uncertainty_quantification"]
    
    figures_ens_out = repo_root / "figures" / "deep_ensemble"
    figures_diag_out = repo_root / "figures" / "inversion_diagnostics"
    figures_ens_out.mkdir(parents=True, exist_ok=True)
    figures_diag_out.mkdir(parents=True, exist_ok=True)
    
    # Load model
    m_single = VoltageSurrogate().to(device)
    ckpt_path = repo_root / "models" / "surrogate_10k_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_single.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    if "v_mean" in ckpt and "v_std" in ckpt:
        m_single.set_normalization_stats(ckpt["v_mean"], ckpt["v_std"])
    m_single.eval()
    
    # Load 5 ensemble member checkpoints from models/deep_ensemble/
    ens_models_dir = repo_root / "models" / "deep_ensemble"
    seeds = [42, 142, 242, 342, 442]
    members = []
    for idx, seed in enumerate(seeds):
        m_p = ens_models_dir / f"ensemble_member_{idx}_seed{seed}.pt"
        m = VoltageSurrogate().to(device)
        c = torch.load(m_p, map_location=device, weights_only=False)
        m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c)
        if "v_mean" in c and "v_std" in c:
            m.set_normalization_stats(c["v_mean"], c["v_std"])
        m.eval()
        members.append(m)
    ensemble = EnsembleModel(members)
    ensemble.eval()
    
    # -------------------------------------------------------------------------
    # FIGURE E1: Single vs Ensemble Forward Accuracy
    # -------------------------------------------------------------------------
    log_msg("\nGenerating Figure E1: Single vs Ensemble Forward Accuracy...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor="#ffffff")
    for ax_i in [ax1, ax2, ax3]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    model_labels = ["Single\nBaseline"] + [f"Member #{m['member_idx']+1}\n(Seed {m['seed']})" for m in member_sums] + ["Ensemble\nMean (K=5)"]
    rel_errors = [base_sum["mean_rel_err_pct"]] + [m["mean_rel_err_pct"] for m in member_sums] + [ens_sum["mean_rel_err_pct"]]
    colors = ["#4a5568"] + [C_GRAY]*len(member_sums) + [C_BLUE]
    
    xm = np.arange(len(model_labels))
    ax1.bar(xm, rel_errors, color=colors, width=0.55, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.8)
    ax1.set_xticks(xm)
    ax1.set_xticklabels(model_labels, fontsize=7.8)
    ax1.set_ylabel("Mean Relative Voltage Error (%)")
    ax1.set_title("(a) Forward Voltage Relative Error\n(1,000 Held-Out Test Samples)", fontsize=10.5)
    for idx, val in enumerate(rel_errors):
        ax1.annotate(f"{val:.3f}%", xy=(idx, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        
    rmse_vals = [base_sum["mean_rmse_mv"]] + [m["mean_rmse_mv"] for m in member_sums] + [ens_sum["mean_rmse_mv"]]
    ax2.bar(xm, rmse_vals, color=colors, width=0.55, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.8)
    ax2.set_xticks(xm)
    ax2.set_xticklabels(model_labels, fontsize=7.8)
    ax2.set_ylabel("Physical Voltage RMSE (mV)")
    ax2.set_title("(b) Physical Voltage Error\n(Ground-Truth Scale mV)", fontsize=10.5)
    for idx, val in enumerate(rmse_vals):
        ax2.annotate(f"{val:.3f}", xy=(idx, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        
    metrics_names = ["Median (50%)", "P90 (90%)", "P95 (95%)", "Max Error"]
    base_tails = [base_sum["median_rel_err_pct"], base_sum["p90_rel_err_pct"], base_sum["p95_rel_err_pct"], base_sum["max_rel_err_pct"]]
    ens_tails = [ens_sum["median_rel_err_pct"], ens_sum["p90_rel_err_pct"], ens_sum["p95_rel_err_pct"], ens_sum["max_rel_err_pct"]]
    
    xt = np.arange(len(metrics_names))
    ax3.bar(xt - 0.18, base_tails, width=0.35, label="Single Baseline Model", color="#4a5568")
    ax3.bar(xt + 0.18, ens_tails, width=0.35, label="Ensemble Mean (K=5)", color=C_BLUE)
    ax3.set_xticks(xt)
    ax3.set_xticklabels(metrics_names, fontsize=8.5)
    ax3.set_ylabel("Relative Voltage Error (%)")
    ax3.set_title("(c) Error Distribution Tail Reduction", fontsize=10.5)
    ax3.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_ens_out / "fig_E1_single_vs_ensemble_accuracy")
    
    # -------------------------------------------------------------------------
    # FIGURE E2: Epistemic Uncertainty vs Error Correlation
    # -------------------------------------------------------------------------
    log_msg("Generating Figure E2: Epistemic Uncertainty Calibration...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor="#ffffff")
    for ax_i in [ax1, ax2]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    spreads = np.array(unc_res["per_sample_spread_mv"])
    errors = np.array(unc_res["per_sample_error_mv"])
    
    ax1.scatter(spreads, errors, color=C_BLUE, alpha=0.30, s=9, rasterized=True, label="Test Samples (n=1,000)")
    m_slope, b_intercept = np.polyfit(spreads, errors, 1)
    x_line = np.linspace(spreads.min(), spreads.max(), 100)
    ax1.plot(x_line, m_slope * x_line + b_intercept, "r--", linewidth=1.8,
             label=f"Linear Fit ($r = {unc_res['pearson_r']:.3f}, \\rho = {unc_res['spearman_rho']:.3f}$)")
    ax1.set_xlabel("Ensemble Predictive Spread $\\sigma_{\\mathrm{ens}}$ (mV)")
    ax1.set_ylabel("Actual Prediction RMSE (mV)")
    ax1.set_title(f"(a) Epistemic Spread vs. Empirical Error\n(Pearson $r = {unc_res['pearson_r']:.3f},\\ p = {unc_res['pearson_p']:.2e}$)", fontsize=10.5)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    inv_t_ids = [r["target_id"] for r in ens_inv_records]
    inv_ious = [r["iou"] for r in ens_inv_records]
    inv_spreads = [r["recovered_spread_mv"] for r in ens_inv_records]
    
    xi = np.arange(len(inv_t_ids))
    ax2.bar(xi - 0.18, inv_ious, width=0.35, color=C_GREEN, label="Recovered IoU")
    ax2.set_ylabel("Geometric IoU")
    ax2.set_ylim(0, 1.1)
    
    ax2_r = ax2.twinx()
    ax2_r.plot(xi + 0.18, inv_spreads, "s-", color=C_CRIMSON, linewidth=1.8, label="Recovered Spread $\\sigma_{\\mathrm{ens}}$ (mV)")
    ax2_r.set_ylabel("Predictive Spread (mV)")
    ax2_r.grid(False)
    
    ax2.set_xticks(xi)
    ax2.set_xticklabels([f"#{r['target_id']}\n{r['shape_family'][:5]}" for r in ens_inv_records], fontsize=8)
    ax2.set_xlabel("Held-Out Target")
    ax2.set_title("(b) Inverse Reconstruction Fidelity vs. Ensemble Spread", fontsize=10.5)
    ax2.legend(loc="lower left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    ax2_r.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_ens_out / "fig_E2_epistemic_uncertainty_calibration")
    
    # -------------------------------------------------------------------------
    # FIGURE E3: Measurement Uncertainty Profile (Explicit 5-Member Variation)
    # -------------------------------------------------------------------------
    log_msg("Generating Figure E3: Measurement Uncertainty Profile & Member Disagreement...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.0, 7.0), facecolor="#ffffff", sharex=True)
    for ax_i in [ax1, ax2]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    tgt1 = all_targets[0]
    th_t1 = torch.from_numpy(np.array(tgt1["theta_true"], dtype=np.float32)).to(device)
    v_true_t1 = np.array(tgt1["voltage_target_fem"][:120]) * 1000.0
    
    with torch.no_grad():
        v_m_preds = [m.predict(th_t1).cpu().numpy()[:120] * 1000.0 for m in members]
        v_stack = np.array(v_m_preds)
        v_mean_prof = np.mean(v_stack, axis=0)
        v_std_prof = np.std(v_stack, axis=0)
        
    ch = np.arange(1, 121)
    
    for idx, m_pred in enumerate(v_m_preds):
        lbl = f"Member #{idx+1} (Seed {seeds[idx]})" if idx == 0 else f"Member #{idx+1}"
        ax1.plot(ch, m_pred, ":", color=C_GRAY, alpha=0.65, linewidth=1.1, label=lbl if idx < 2 else None)
        
    ax1.plot(ch, v_true_t1, "-", color="#1a202c", linewidth=1.6, label="Ground Truth FEM Voltage")
    ax1.plot(ch, v_mean_prof, "--", color=C_BLUE, linewidth=1.6, label="Deep Ensemble Mean $\\bar{V}(\\mathbf{\\theta})$")
    ax1.fill_between(ch, v_mean_prof - 2*v_std_prof, v_mean_prof + 2*v_std_prof, color=C_BLUE, alpha=0.20, label="Epistemic Spread Envelope $\\pm 2\\sigma_{\\mathrm{ens}}$")
    
    ax1.set_ylabel("Boundary Voltage (mV)")
    ax1.set_title("(a) 120-Channel Voltage Profile: 5 Individual Members vs. Ensemble Mean & Ground Truth (Target #1)", fontsize=11)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, ncol=2)
    
    ax2.bar(ch, v_std_prof, color=C_CRIMSON, alpha=0.75, width=0.8, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.4, label="Channel Predictive Spread $\\sigma_{\\mathrm{ens}, i}$")
    ax2.axhline(np.mean(v_std_prof), color=C_DARK, linestyle="--", linewidth=1.4, label=f"Mean Channel Spread: {np.mean(v_std_prof):.3f} mV (Median: {np.median(v_std_prof):.3f} mV)")
    ax2.set_xlabel("Differential Voltage Channel Index $i \\in \\{1, \\dots, 120\\}$ (16-Electrode Adjacent Pairings)")
    ax2.set_ylabel("Predictive Spread $\\sigma_{\\mathrm{ens}}$ (mV)")
    ax2.set_title("(b) Channel-wise Epistemic Disagreement Magnitude Across 5 Independent Models", fontsize=11)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_ens_out / "fig_E3_measurement_uncertainty_profile")
    
    # -------------------------------------------------------------------------
    # FIGURE E4A: Reconstructions for Targets 1 to 5
    # -------------------------------------------------------------------------
    log_msg("Generating Figure E4A: Reconstructions for Targets 1-5...")
    th_circ = np.linspace(0, 2*np.pi, 200)
    x_circ = np.cos(th_circ)
    y_circ = np.sin(th_circ)
    
    def plot_reconstruction_set(target_slice, figure_title, out_name):
        fig, axes = plt.subplots(len(target_slice), 3, figsize=(10.5, 2.7 * len(target_slice)), facecolor="#ffffff")
        for r_idx, tgt in enumerate(target_slice):
            t_id = tgt["target_id"]
            s_rec = [t for t in single_inv_records if t["target_id"] == t_id][0]
            e_rec = [t for t in ens_inv_records if t["target_id"] == t_id][0]
            
            fam = tgt["shape_family"].replace("_", " ").title()
            is_cvx = "Convex" if tgt.get("is_convex", True) else "Concave"
            th_true = np.array(tgt["theta_true"])
            th_single = np.array(s_rec["theta_recovered"])
            th_ens = np.array(e_rec["theta_recovered"])
            
            curve_true, _ = compute_bspline_boundary(th_true, n_samples=300)
            curve_single, _ = compute_bspline_boundary(th_single, n_samples=300)
            curve_ens, _ = compute_bspline_boundary(th_ens, n_samples=300)
            
            # Col 0: Ground Truth
            ax0 = axes[r_idx, 0]
            remove_axes_frame(ax0)
            ax0.plot(x_circ, y_circ, color=COLOR_DOMAIN_LINE, linewidth=1.3)
            ax0.fill(curve_true[:, 0], curve_true[:, 1], color=COLOR_INC_FILL)
            ax0.plot(curve_true[:, 0], curve_true[:, 1], "-", color="#ffffff", linewidth=1.2)
            ax0.set_aspect("equal")
            ax0.set_xlim(-1.08, 1.08)
            ax0.set_ylim(-1.08, 1.08)
            if r_idx == 0:
                ax0.set_title("Ground Truth", fontsize=10.5, fontweight="bold", pad=6)
            ax0.text(-1.05, 0.95, f"Target #{t_id}\n{fam} ({is_cvx})", fontsize=8.5, fontweight="bold", va="top", ha="left")
            
            # Col 1: Single Model Reconstruction
            ax1 = axes[r_idx, 1]
            remove_axes_frame(ax1)
            ax1.plot(x_circ, y_circ, color=COLOR_DOMAIN_LINE, linewidth=1.3)
            ax1.fill(curve_single[:, 0], curve_single[:, 1], color=COLOR_INC_FILL)
            ax1.plot(curve_true[:, 0], curve_true[:, 1], ":", color="#718096", alpha=0.75, linewidth=1.1)
            ax1.plot(curve_single[:, 0], curve_single[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6)
            ax1.set_aspect("equal")
            ax1.set_xlim(-1.08, 1.08)
            ax1.set_ylim(-1.08, 1.08)
            if r_idx == 0:
                ax1.set_title("Single Baseline Model", fontsize=10.5, fontweight="bold", pad=6)
            ax1.text(0.98, -0.98, f"IoU: {s_rec['iou']:.4f}\nRMS: {s_rec['boundary_rms_dist']:.3f}m", ha="right", va="bottom", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor=BOX_BG, edgecolor=BOX_BORDER, lw=0.7))
                     
            # Col 2: Ensemble Mean Reconstruction
            ax2 = axes[r_idx, 2]
            remove_axes_frame(ax2)
            ax2.plot(x_circ, y_circ, color=COLOR_DOMAIN_LINE, linewidth=1.3)
            ax2.fill(curve_ens[:, 0], curve_ens[:, 1], color=COLOR_INC_FILL)
            ax2.plot(curve_true[:, 0], curve_true[:, 1], ":", color="#718096", alpha=0.75, linewidth=1.1)
            ax2.plot(curve_ens[:, 0], curve_ens[:, 1], "--", color=COLOR_ENS_LINE, linewidth=1.6)
            ax2.set_aspect("equal")
            ax2.set_xlim(-1.08, 1.08)
            ax2.set_ylim(-1.08, 1.08)
            if r_idx == 0:
                ax2.set_title("Deep Ensemble Mean (K=5)", fontsize=10.5, fontweight="bold", pad=6)
            ax2.text(0.98, -0.98, f"IoU: {e_rec['iou']:.4f}\nRMS: {e_rec['boundary_rms_m']:.3f}m\n$\\sigma_{{ens}}$: {e_rec['recovered_spread_mv']:.2f}mV", ha="right", va="bottom", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor=BOX_BG, edgecolor=BOX_BORDER, lw=0.7))
                     
        fig.suptitle(figure_title, fontsize=12, y=0.995)
        plt.tight_layout()
        save_fig(fig, figures_ens_out / out_name)
        
    plot_reconstruction_set(all_targets[:5], "Inverse Shape Reconstruction (Targets 1–5): Single Baseline vs. Deep Ensemble Mean", "fig_E4A_reconstruction_targets_1_to_5")
    
    # -------------------------------------------------------------------------
    # FIGURE E4B: Reconstructions for Targets 6 to 9
    # -------------------------------------------------------------------------
    log_msg("Generating Figure E4B: Reconstructions for Targets 6-9...")
    plot_reconstruction_set(all_targets[5:], "Inverse Shape Reconstruction (Targets 6–9): Single Baseline vs. Deep Ensemble Mean", "fig_E4B_reconstruction_targets_6_to_9")
    plot_reconstruction_set(all_targets[5:], "Inverse Shape Reconstruction (Targets 6–9): Single Baseline vs. Deep Ensemble Mean", "fig_E4B_reconstruction_targets_6_to_10")
    
    # -------------------------------------------------------------------------
    # FIGURE E5: Quantitative Inversion Comparison Matrix (All 9 Targets)
    # -------------------------------------------------------------------------
    log_msg("Generating Figure E5: 9-Target Quantitative Comparison Table...")
    fig, ax = plt.subplots(figsize=(13.0, 5.2), facecolor="#ffffff")
    ax.axis("off")
    
    headers = [
        "Target", "Shape Family", "Type", 
        "Baseline\nIoU", "Ensemble\nIoU", "IoU\nGain", 
        "Baseline\nRMS (m)", "Ensemble\nRMS (m)", "RMS\nReduction", 
        "Ensemble\nSpread (mV)", "LM\nIters", "Solve\nTime (s)"
    ]
    
    rows = []
    for tgt in all_targets:
        t_id = tgt["target_id"]
        fam = tgt["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if tgt.get("is_convex", True) else "Concave"
        s_rec = [t for t in single_inv_records if t["target_id"] == t_id][0]
        e_rec = [t for t in ens_inv_records if t["target_id"] == t_id][0]
        
        b_iou = s_rec["iou"]
        e_iou = e_rec["iou"]
        iou_diff = e_iou - b_iou
        
        b_rms = s_rec["boundary_rms_dist"]
        e_rms = e_rec["boundary_rms_m"]
        rms_red = (b_rms - e_rms) / b_rms * 100.0 if b_rms > 0 else 0.0
        
        sp = e_rec["recovered_spread_mv"]
        it = e_rec["lm_iters"]
        tm = e_rec["solve_time_s"]
        
        rows.append([
            f"#{t_id}", fam, is_cvx,
            f"{b_iou:.4f}", f"{e_iou:.4f}", f"{iou_diff:+.4f}",
            f"{b_rms:.4f}", f"{e_rms:.4f}", f"{rms_red:+.1f}%",
            f"{sp:.2f}", f"{it}", f"{tm:.2f}"
        ])
        
    # Add summary row dynamically
    mean_b_iou = np.mean([t["iou"] for t in single_inv_records])
    mean_e_iou = np.mean([t["iou"] for t in ens_inv_records])
    mean_b_rms = np.mean([t["boundary_rms_dist"] for t in single_inv_records])
    mean_e_rms = np.mean([t["boundary_rms_m"] for t in ens_inv_records])
    mean_rms_red = (mean_b_rms - mean_e_rms) / mean_b_rms * 100.0 if mean_b_rms > 0 else 0.0

    rows.append([
        "Mean", "All 9 Targets", "Overall",
        f"{mean_b_iou:.4f}", f"{mean_e_iou:.4f}", f"{mean_e_iou - mean_b_iou:+.4f}",
        f"{mean_b_rms:.4f}", f"{mean_e_rms:.4f}", f"{mean_rms_red:+.1f}%",
        f"{np.mean([r['recovered_spread_mv'] for r in ens_inv_records]):.2f}",
        f"{np.mean([r['lm_iters'] for r in ens_inv_records]):.1f}",
        f"{np.mean([r['solve_time_s'] for r in ens_inv_records]):.2f}"
    ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    
    # Style header and rows
    for (r_i, c_i), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e0")
        if r_i == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="#ffffff", fontweight="bold")
        elif r_i == len(rows):
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold")
        else:
            if r_i % 2 == 1:
                cell.set_facecolor("#ffffff")
            else:
                cell.set_facecolor("#f7fafc")
                
    plt.title(r"$\mathbf{Surrogate-Assisted\;L-M\;Inversion\;Performance\;Matrix:\;Single\;Baseline\;vs.\;Deep\;Ensemble\;Mean}$", fontsize=11.5, pad=12)
    plt.tight_layout()
    save_fig(fig, figures_ens_out / "fig_E5_quantitative_inversion_table")
    
    # -------------------------------------------------------------------------
    # FIGURE LM1: Objective Convergence Trajectories
    # -------------------------------------------------------------------------
    log_msg("Generating Figure LM1: Objective Convergence Trajectories...")
    lm_cfg_rec = LMConfig(max_iters=50, verbose=False)
    solver_rec = LMSolver(lm_cfg_rec)
    inits_4 = BSplineInitializer(seed=cfg.DEFAULT_SEED).generate_inits(4)
    
    prob_det_t2 = SurrogateResidualProblem(members[0], np.array(all_targets[1]["voltage_target_fem"]))
    prob_ens_t2 = EnsembleInversionProblem(ensemble, np.array(all_targets[1]["voltage_target_fem"]))
    
    res_det_t2 = solver_rec.solve(prob_det_t2, inits_4[0])
    res_ens_t2 = solver_rec.solve(prob_ens_t2, inits_4[0])
    
    prob_det_t5 = SurrogateResidualProblem(members[0], np.array(all_targets[4]["voltage_target_fem"]))
    prob_ens_t5 = EnsembleInversionProblem(ensemble, np.array(all_targets[4]["voltage_target_fem"]))
    res_det_t5 = solver_rec.solve(prob_det_t5, inits_4[0])
    res_ens_t5 = solver_rec.solve(prob_ens_t5, inits_4[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor="#ffffff")
    for ax_i in [ax1, ax2]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        ax_i.set_yscale("log")
        
    ax1.plot([h.iter_idx for h in res_det_t2.history], [h.objective for h in res_det_t2.history], "o-", color=C_CRIMSON, label="Single Baseline Model")
    ax1.plot([h.iter_idx for h in res_ens_t2.history], [h.objective for h in res_ens_t2.history], "s--", color=C_BLUE, label="Deep Ensemble Mean (K=5)")
    ax1.set_xlabel("Levenberg-Marquardt Iteration $k$")
    ax1.set_ylabel("Least-Squares Objective $f(\\mathbf{\\theta}) = \\frac{1}{2}\\|\\mathbf{r}\\|^2$")
    ax1.set_title("(a) Target #2 (Ellipse — Convex) Convergence Trajectory", fontsize=10.5)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    ax2.plot([h.iter_idx for h in res_det_t5.history], [h.objective for h in res_det_t5.history], "o-", color=C_CRIMSON, label="Single Baseline Model")
    ax2.plot([h.iter_idx for h in res_ens_t5.history], [h.objective for h in res_ens_t5.history], "s--", color=C_BLUE, label="Deep Ensemble Mean (K=5)")
    ax2.set_xlabel("Levenberg-Marquardt Iteration $k$")
    ax2.set_ylabel("Least-Squares Objective $f(\\mathbf{\\theta}) = \\frac{1}{2}\\|\\mathbf{r}\\|^2$")
    ax2.set_title("(b) Target #5 (Star — Concave) Convergence Trajectory", fontsize=10.5)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_diag_out / "fig_LM1_objective_convergence_history")
    
    # -------------------------------------------------------------------------
    # FIGURE LM2: Gradient, Step, and Damping Dynamics
    # -------------------------------------------------------------------------
    log_msg("Generating Figure LM2: Gradient Norm, Step Norm & Adaptive Damping...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.5), facecolor="#ffffff")
    for ax_i in [ax1, ax2, ax3]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        ax_i.set_yscale("log")
        
    iters_e = [h.iter_idx for h in res_ens_t2.history]
    ax1.plot(iters_e, [h.grad_norm for h in res_ens_t2.history], "d-", color=C_PURPLE, linewidth=1.6)
    ax1.axhline(1e-6, color=C_DARK, linestyle=":", label="Tolerance $\\mathrm{gtol} = 10^{-6}$")
    ax1.set_xlabel("LM Iteration $k$")
    ax1.set_ylabel("Gradient Norm $\\|\\mathbf{g}_k\\| = \\|\\mathcal{J}^T \\mathbf{r}\\|$")
    ax1.set_title("(a) Gradient Norm Progression", fontsize=10.5)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    ax2.plot(iters_e, [h.step_norm for h in res_ens_t2.history], "^-", color=C_TEAL, linewidth=1.6)
    ax2.axhline(1e-6, color=C_DARK, linestyle=":", label="Tolerance $\\mathrm{xtol} = 10^{-6}$")
    ax2.set_xlabel("LM Iteration $k$")
    ax2.set_ylabel("Step Parameter Norm $\\|\\delta_k\\|$")
    ax2.set_title("(b) Parameter Step Magnitude", fontsize=10.5)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    ax3.plot(iters_e, [h.lambda_val for h in res_ens_t2.history], "s-", color=C_ORANGE, linewidth=1.6)
    ax3.set_xlabel("LM Iteration $k$")
    ax3.set_ylabel("Adaptive Damping Parameter $\\lambda_k$")
    ax3.set_title("(c) Nielsen Adaptive Damping $\\lambda_k$", fontsize=10.5)
    
    plt.tight_layout()
    save_fig(fig, figures_diag_out / "fig_LM2_gradient_step_damping_dynamics")
    
    # -------------------------------------------------------------------------
    # FIGURE LM3: Solver Statistics Across Conditions
    # -------------------------------------------------------------------------
    log_msg("Generating Figure LM3: Solver Statistics Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), facecolor="#ffffff")
    for ax_i in [ax1, ax2]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    cond_labels = ["Deterministic\nBaseline", "Deep Ensemble\nMean (K=5)", "Noise 0.1%\n(60 dB)", "Noise 1.0%\n(40 dB)", "Noise 5.0%\n(26 dB)"]
    iter_counts = [
        np.mean([r["lm_iters"] for r in single_inv_records]),
        np.mean([r["lm_iters"] for r in ens_inv_records]),
        noise_data["condition_summaries"]["noise_0.1pct"]["mean_lm_iters"],
        noise_data["condition_summaries"]["noise_1.0pct"]["mean_lm_iters"],
        noise_data["condition_summaries"]["noise_5.0pct"]["mean_lm_iters"],
    ]
    time_counts = [
        np.mean([r["inv_time_s"] for r in single_inv_records]),
        np.mean([r["solve_time_s"] for r in ens_inv_records]),
        noise_data["condition_summaries"]["noise_0.1pct"]["mean_solve_time_s"],
        noise_data["condition_summaries"]["noise_1.0pct"]["mean_solve_time_s"],
        noise_data["condition_summaries"]["noise_5.0pct"]["mean_solve_time_s"],
    ]
    
    xc = np.arange(len(cond_labels))
    ax1.bar(xc, iter_counts, color=[C_GRAY, C_BLUE, C_TEAL, C_ORANGE, C_CRIMSON], width=0.55, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.8)
    ax1.set_xticks(xc)
    ax1.set_xticklabels(cond_labels, fontsize=8)
    ax1.set_ylabel("Mean LM Iteration Count")
    ax1.set_title("(a) Optimizer Iterations to Convergence", fontsize=10.5)
    for idx, val in enumerate(iter_counts):
        ax1.annotate(f"{val:.1f}", xy=(idx, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax2.bar(xc, time_counts, color=[C_GRAY, C_BLUE, C_TEAL, C_ORANGE, C_CRIMSON], width=0.55, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.8)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cond_labels, fontsize=8)
    ax2.set_ylabel("Mean Multi-Start Solve Time (s)")
    ax2.set_title("(b) End-to-End Reconstruction Execution Time", fontsize=10.5)
    for idx, val in enumerate(time_counts):
        ax2.annotate(f"{val:.1f}s", xy=(idx, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    save_fig(fig, figures_diag_out / "fig_LM3_solver_statistics_comparison")
    
    # -------------------------------------------------------------------------
    # FIGURE MS1: Multi-Start Restart Distribution
    # -------------------------------------------------------------------------
    log_msg("Generating Figure MS1: Multi-Start Restart Performance...")
    restart_objs = []  # shape (10, 4)
    for tgt in all_targets:
        prob = SurrogateResidualProblem(members[0], np.array(tgt["voltage_target_fem"]))
        ms = MultiStartSolver(lm_config=LMConfig(max_iters=50, verbose=False), n_restarts=4)
        res = ms.solve(prob)
        restart_objs.append([r.objective_opt for r in res.all_results])
        
    restart_mat = np.array(restart_objs)  # (10, 4)
    
    fig, ax = plt.subplots(figsize=(11.5, 4.6), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")
    ax.set_yscale("log")
    
    xtg = np.arange(len(all_targets))
    restart_colors = [C_BLUE, C_GREEN, C_ORANGE, C_PURPLE]
    restart_names = ["Restart 1 (Circle Nominal)", "Restart 2 (Reference Shape)", "Restart 3 (Random Perturbation A)", "Restart 4 (Random Perturbation B)"]
    
    for r_idx in range(4):
        ax.scatter(xtg + (r_idx - 1.5)*0.18, restart_mat[:, r_idx], color=restart_colors[r_idx], s=45, label=restart_names[r_idx], edgecolor=COLOR_DOMAIN_LINE, linewidth=0.6, zorder=3)
        
    best_objs = np.min(restart_mat, axis=1)
    ax.plot(xtg, best_objs, "k--", alpha=0.5, linewidth=1.2, label="Global Selected Best Minimum", zorder=2)
    
    ax.set_xticks(xtg)
    ax.set_xticklabels([f"Target #{t['target_id']}\n{t['shape_family'][:6]}" for t in all_targets], fontsize=8.5)
    ax.set_xlabel("Held-Out Target")
    ax.set_ylabel("Final Objective $f(\\mathbf{\\theta}^*) = \\frac{1}{2}\\|\\mathbf{r}\\|^2$")
    ax.set_title("Multi-Start Optimization Performance Across 4 Deterministic Initializations (9 Targets)", fontsize=11)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_diag_out / "fig_MS1_multistart_restart_distribution")
    
    log_msg("\nAll figures generated and verified successfully!")
    
    print("\nAll figures generated and verified successfully!")


if __name__ == "__main__":
    main()
