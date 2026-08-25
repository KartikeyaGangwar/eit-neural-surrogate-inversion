"""
Publication-Quality Visualization for Measurement Noise Robustness
===================================================================
Generates:
1. figures/noise_robustness/fig_N1_noise_vs_iou_rms.png / .pdf
2. figures/noise_robustness/fig_N2_noise_reconstruction_gallery.png / .pdf
3. figures/noise_robustness/fig_N3_voltage_residual_degradation.png / .pdf
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.geometry.bspline import compute_bspline_boundary

# Aesthetics
COLOR_BG_FILL = "#ffffff"
COLOR_INC_FILL = "#000000"
COLOR_DOMAIN_LINE = "#1a202c"
COLOR_REC_LINE = "#e53e3e"
COLOR_REF_LINE = "#718096"
BOX_BG = "#f8fafc"
BOX_BORDER = "#cbd5e0"

C_BLUE = "#2b6cb0"
C_GREEN = "#276749"
C_CRIMSON = "#c53030"
C_ORANGE = "#dd6b20"
C_PURPLE = "#6b46c1"
C_GRAY = "#718096"

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
    fig.savefig(png_p, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_p, format="pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Exported: {png_p.name} & {pdf_p.name}")


def main():
    results_path = repo_root / "data" / "results" / "noise_robustness_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
        
    with open(results_path) as f:
        data = json.load(f)
        
    cond_sums = data["condition_summaries"]
    trials = data["per_trial_records"]
    figures_out = repo_root / "figures" / "noise_robustness"
    figures_out.mkdir(parents=True, exist_ok=True)
    
    noise_pcts = [cond_sums[k]["noise_pct"] for k in cond_sums]
    mean_ious = [cond_sums[k]["mean_iou"] for k in cond_sums]
    median_ious = [cond_sums[k]["median_iou"] for k in cond_sums]
    p10_ious = [cond_sums[k]["p10_iou"] for k in cond_sums]
    p90_ious = [cond_sums[k]["p90_iou"] for k in cond_sums]
    cvx_ious = [cond_sums[k]["convex_mean_iou"] for k in cond_sums]
    ccv_ious = [cond_sums[k]["concave_mean_iou"] for k in cond_sums]
    
    mean_rms = [cond_sums[k]["mean_boundary_rms_m"] for k in cond_sums]
    median_rms = [cond_sums[k]["median_boundary_rms_m"] for k in cond_sums]
    rel_degrad_pct = [cond_sums[k]["iou_rel_degradation_pct"] for k in cond_sums]
    
    # -------------------------------------------------------------------------
    # FIGURE N1: Aggregate Metrics vs Noise
    # -------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor="#ffffff")
    
    for ax_i in [ax1, ax2, ax3]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    # (a) IoU vs Noise
    ax1.plot(noise_pcts, mean_ious, "o-", color=C_BLUE, linewidth=2.0, markersize=6, label="Overall Mean IoU", zorder=4)
    ax1.plot(noise_pcts, median_ious, "s--", color=C_GREEN, linewidth=1.6, markersize=5, label="Overall Median IoU", zorder=4)
    ax1.plot(noise_pcts, cvx_ious, "^:", color=C_ORANGE, linewidth=1.5, markersize=5, label="Convex Targets", zorder=3)
    ax1.plot(noise_pcts, ccv_ious, "v:", color=C_PURPLE, linewidth=1.5, markersize=5, label="Concave Targets", zorder=3)
    ax1.fill_between(noise_pcts, p10_ious, p90_ious, color=C_BLUE, alpha=0.15, label="P10–P90 Range", zorder=2)
    
    for tr in trials:
        n_p = tr["noise_pct"]
        ax1.scatter(n_p + np.random.normal(0, 0.04), tr["iou"], color="#a0aec0", alpha=0.25, s=8, zorder=1)
        
    ax1.set_xlabel("Measurement Additive Gaussian Noise $\\delta$ (%)")
    ax1.set_ylabel("Geometric IoU")
    ax1.set_ylim(0.40, 1.0)
    ax1.set_title("(a) Geometric IoU vs. Noise Level\n($\\mathrm{SNR} \\in [\\infty, 60, 46, 40, 34, 26]\\ \\mathrm{dB}$)", fontsize=10.5)
    ax1.legend(loc="lower left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    # (b) Boundary RMS vs Noise
    ax2.plot(noise_pcts, mean_rms, "s-", color=C_CRIMSON, linewidth=2.0, markersize=6, label="Mean Boundary RMS", zorder=4)
    ax2.plot(noise_pcts, median_rms, "o--", color=C_PURPLE, linewidth=1.6, markersize=5, label="Median Boundary RMS", zorder=4)
    ax2.set_xlabel("Measurement Additive Gaussian Noise $\\delta$ (%)")
    ax2.set_ylabel("Boundary RMS Distance (m)")
    ax2.set_title("(b) Boundary Reconstruction Error vs. Noise\n(Domain Radius $R = 1.0\\ \\mathrm{m}$)", fontsize=10.5)
    ax2.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    # (c) Relative Degradation Percentage
    ax3.bar(np.arange(len(noise_pcts)), rel_degrad_pct, color=C_BLUE, width=0.55, alpha=0.85, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.8)
    ax3.set_xticks(np.arange(len(noise_pcts)))
    ax3.set_xticklabels([f"{p:.1f}%" for p in noise_pcts])
    ax3.set_xlabel("Measurement Noise Level $\\delta$")
    ax3.set_ylabel("Relative IoU Degradation from Clean Baseline (%)")
    ax3.set_title("(c) Relative IoU Degradation\n($(\\mathrm{IoU}_0 - \\mathrm{IoU}_\\delta) / \\mathrm{IoU}_0 \\times 100\\%$)", fontsize=10.5)
    
    for idx, val in enumerate(rel_degrad_pct):
        ax3.annotate(f"{val:.1f}%", xy=(idx, val), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_N1_noise_vs_iou_rms")
    
    # -------------------------------------------------------------------------
    # FIGURE N2: Representative Reconstructions across Noise Levels
    # -------------------------------------------------------------------------
    rep_targets = [2, 7, 9]  # Ellipse (cvx), Random Concave (ccv), Crescent (ccv)
    shown_noises = [0.0, 0.005, 0.01, 0.02, 0.05]
    
    fig, axes = plt.subplots(len(rep_targets), len(shown_noises) + 1, figsize=(16.0, 8.5), facecolor="#ffffff")
    
    th_domain = np.linspace(0, 2*np.pi, 200)
    x_circ = np.cos(th_domain)
    y_circ = np.sin(th_domain)
    
    for r_idx, t_id in enumerate(rep_targets):
        tgt_trials = [t for t in trials if t["target_id"] == t_id]
        tgt_clean = [t for t in tgt_trials if t["noise_fraction"] == 0.0][0]
        th_true = np.array(tgt_clean["theta_true"])
        curve_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        fam = tgt_clean["shape_family"].replace("_", " ").title()
        
        # Col 0: Ground Truth
        ax_gt = axes[r_idx, 0]
        remove_axes_frame(ax_gt)
        ax_gt.plot(x_circ, y_circ, color=COLOR_DOMAIN_LINE, linewidth=1.3)
        ax_gt.fill(curve_true[:, 0], curve_true[:, 1], color=COLOR_INC_FILL)
        ax_gt.plot(curve_true[:, 0], curve_true[:, 1], "-", color="#ffffff", linewidth=1.2)
        ax_gt.set_aspect("equal")
        ax_gt.set_xlim(-1.08, 1.08)
        ax_gt.set_ylim(-1.08, 1.08)
        if r_idx == 0:
            ax_gt.set_title("Ground Truth", fontsize=10, fontweight="bold", pad=6)
        ax_gt.text(-1.05, 0.95, f"Target #{t_id}\n({fam})", fontsize=8.5, fontweight="bold", va="top", ha="left")
        
        # Cols 1..5: Noise levels
        for c_idx, delta in enumerate(shown_noises):
            ax_rec = axes[r_idx, c_idx + 1]
            remove_axes_frame(ax_rec)
            
            cand_trials = [t for t in tgt_trials if t["noise_fraction"] == delta]
            best_trial = max(cand_trials, key=lambda x: x["iou"])
            
            th_rec = np.array(best_trial["theta_recovered"])
            curve_rec, _ = compute_bspline_boundary(th_rec, n_samples=300)
            
            ax_rec.plot(x_circ, y_circ, color=COLOR_DOMAIN_LINE, linewidth=1.3)
            ax_rec.fill(curve_rec[:, 0], curve_rec[:, 1], color=COLOR_INC_FILL)
            ax_rec.plot(curve_true[:, 0], curve_true[:, 1], ":", color="#718096", alpha=0.75, linewidth=1.1)
            ax_rec.plot(curve_rec[:, 0], curve_rec[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6)
            ax_rec.set_aspect("equal")
            ax_rec.set_xlim(-1.08, 1.08)
            ax_rec.set_ylim(-1.08, 1.08)
            
            delta_pct = delta * 100.0
            if r_idx == 0:
                header = "Clean (0.0%)" if delta == 0.0 else f"Noise {delta_pct:.1f}%\n({-20*np.log10(delta):.0f} dB SNR)"
                ax_rec.set_title(header, fontsize=9.5, fontweight="bold", pad=6)
                
            info_str = f"IoU: {best_trial['iou']:.4f}\nRMS: {best_trial['boundary_rms_m']:.3f}m"
            ax_rec.text(0.98, -0.98, info_str, ha="right", va="bottom", fontsize=7.2,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=BOX_BG, edgecolor=BOX_BORDER, lw=0.7))
                        
    fig.suptitle(r"$\mathbf{Surrogate-Assisted\;L-M\;Inverse\;Reconstructions\;Under\;Increasing\;Measurement\;Noise}$", fontsize=12.5, y=0.995)
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_N2_noise_reconstruction_gallery")
    
    # -------------------------------------------------------------------------
    # FIGURE N3: Voltage Residual vs Clean Fidelity
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), facecolor="#ffffff")
    for ax_i in [ax1, ax2]:
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_color("#4a5568")
        ax_i.spines["bottom"].set_color("#4a5568")
        
    noisy_v_errs = [cond_sums[k]["mean_noisy_v_err_pct"] for k in cond_sums]
    clean_v_errs = [cond_sums[k]["mean_clean_v_err_pct"] for k in cond_sums]
    
    ax1.plot(noise_pcts, noisy_v_errs, "o-", color=C_ORANGE, linewidth=1.8, label="Residual w.r.t Noisy Input ($\\|V_{\\mathrm{surr}} - V_{\\mathrm{noisy}}\\|$)")
    ax1.plot(noise_pcts, clean_v_errs, "s--", color=C_BLUE, linewidth=1.8, label="Error w.r.t Clean Physical ($V_{\\mathrm{clean}}$)")
    ax1.set_xlabel("Measurement Noise Level $\\delta$ (%)")
    ax1.set_ylabel("Relative Voltage Error (%)")
    ax1.set_title("(a) Optimization Residual vs. True Voltage Fidelity", fontsize=10.5)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    mean_iters = [cond_sums[k]["mean_lm_iters"] for k in cond_sums]
    mean_times = [cond_sums[k]["mean_solve_time_s"] for k in cond_sums]
    
    ax2.bar(np.arange(len(noise_pcts)) - 0.18, mean_iters, width=0.35, color=C_PURPLE, label="Mean LM Iterations")
    ax2.set_ylabel("LM Iterations")
    ax2_r = ax2.twinx()
    ax2_r.plot(np.arange(len(noise_pcts)), mean_times, "d-", color=C_CRIMSON, linewidth=2.0, label="Solve Time (s)")
    ax2_r.set_ylabel("Solve Time (s)")
    ax2_r.grid(False)
    
    ax2.set_xticks(np.arange(len(noise_pcts)))
    ax2.set_xticklabels([f"{p:.1f}%" for p in noise_pcts])
    ax2.set_xlabel("Measurement Noise Level $\\delta$")
    ax2.set_title("(b) Optimizer Convergence and Solve Time", fontsize=10.5)
    ax2.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    ax2_r.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_N3_voltage_residual_degradation")


if __name__ == "__main__":
    main()
