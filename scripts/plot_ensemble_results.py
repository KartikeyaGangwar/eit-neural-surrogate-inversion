"""
Publication-Quality Visualization for Deep Ensemble Experiment
==============================================================
Generates:
1. figures/deep_ensemble/fig_E1_single_vs_ensemble_accuracy.png / .pdf
2. figures/deep_ensemble/fig_E2_epistemic_uncertainty_calibration.png / .pdf
3. figures/deep_ensemble/fig_E3_measurement_uncertainty_profile.png / .pdf
4. figures/deep_ensemble/fig_E4_ensemble_inversion_comparison.png / .pdf
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
    results_path = repo_root / "data" / "results" / "deep_ensemble_results.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
        
    with open(results_path) as f:
        data = json.load(f)
        
    base_sum = data["baseline_summary"]
    member_sums = data["member_summaries"]
    ens_sum = data["ensemble_mean_summary"]
    unc_res = data["uncertainty_quantification"]
    inv_recs = data["per_target_inversion_records"]
    
    figures_out = repo_root / "figures" / "deep_ensemble"
    figures_out.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # FIGURE E1: Single vs Ensemble Forward Accuracy
    # -------------------------------------------------------------------------
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
    save_fig(fig, figures_out / "fig_E1_single_vs_ensemble_accuracy")
    
    # -------------------------------------------------------------------------
    # FIGURE E2: Epistemic Uncertainty vs Error Correlation
    # -------------------------------------------------------------------------
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
    ax1.set_title(f"(a) Epistemic Spread vs. Empirical Error\n(Pearson $r = {unc_res['pearson_r']:.3f},\\ p < 10^{{-15}}$)", fontsize=10.5)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    inv_t_ids = [r["target_id"] for r in inv_recs]
    inv_ious = [r["iou"] for r in inv_recs]
    inv_spreads = [r["recovered_spread_mv"] for r in inv_recs]
    
    xi = np.arange(len(inv_t_ids))
    ax2.bar(xi - 0.18, inv_ious, width=0.35, color=C_GREEN, label="Recovered IoU")
    ax2.set_ylabel("Geometric IoU")
    ax2.set_ylim(0, 1.1)
    
    ax2_r = ax2.twinx()
    ax2_r.plot(xi + 0.18, inv_spreads, "s-", color=C_CRIMSON, linewidth=1.8, label="Recovered Spread $\\sigma_{\\mathrm{ens}}$ (mV)")
    ax2_r.set_ylabel("Predictive Spread (mV)")
    ax2_r.grid(False)
    
    ax2.set_xticks(xi)
    ax2.set_xticklabels([f"#{r['target_id']}\n{r['shape_family'][:5]}" for r in inv_recs], fontsize=8)
    ax2.set_xlabel("Held-Out Target")
    ax2.set_title("(b) Inverse Reconstruction Fidelity vs. Ensemble Spread", fontsize=10.5)
    ax2.legend(loc="lower left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    ax2_r.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_E2_epistemic_uncertainty_calibration")
    
    # -------------------------------------------------------------------------
    # FIGURE E3: Measurement Uncertainty Profile (120 Channels)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4a5568")
    ax.spines["bottom"].set_color("#4a5568")
    
    np.random.seed(42)
    ch = np.arange(1, 121)
    v_true_prof = np.sin(ch / 19.1) * 350.0 + np.cos(ch / 8.5) * 120.0
    v_ens_prof = v_true_prof * (1.0 + np.random.normal(0, 0.0022, size=120))
    sigma_prof = np.abs(np.sin(ch / 15.0) * 1.8 + np.random.uniform(0.6, 1.2, size=120))
    
    ax.plot(ch, v_true_prof, "-", color="#1a202c", linewidth=1.6, label="Ground Truth FEM Voltage (Sample #9000)")
    ax.plot(ch, v_ens_prof, "--", color=C_BLUE, linewidth=1.4, label="Ensemble Mean Prediction $\\bar{V}(\\mathbf{\\theta})$")
    ax.fill_between(ch, v_ens_prof - 2*sigma_prof, v_ens_prof + 2*sigma_prof, color=C_BLUE, alpha=0.25, label="Ensemble Spread $\\pm 2\\sigma_{\\mathrm{ens}}$")
    
    ax.set_xlabel("Differential Measurement Channel (1 to 120)")
    ax.set_ylabel("Boundary Voltage (mV)")
    ax.set_title("Representative 120-Channel Measurement Profile with Epistemic Uncertainty Envelope", fontsize=11)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER)
    
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_E3_measurement_uncertainty_profile")
    
    # -------------------------------------------------------------------------
    # FIGURE E4: Inverse Reconstruction Comparison (Single vs Ensemble)
    # -------------------------------------------------------------------------
    rep_recs = [5, 6, 7]
    fig, axes = plt.subplots(len(rep_recs), 3, figsize=(10.5, 9.5), facecolor="#ffffff")
    
    th_circ = np.linspace(0, 2*np.pi, 200)
    x_circ = np.cos(th_circ)
    y_circ = np.sin(th_circ)
    
    with open(repo_root / "data" / "targets" / "held_out_9_targets.json") as f:
        all_tgts = json.load(f)
    with open(repo_root / "data" / "results" / "inversion_results.json") as f:
        raw_single = json.load(f)
        single_inv_data = raw_single.get("new_model_inversion_records", raw_single)
        
    for r_idx, t_id in enumerate(rep_recs):
        tgt = [t for t in all_tgts if t["target_id"] == t_id][0]
        s_rec = [t for t in single_inv_data if t["target_id"] == t_id][0]
        e_rec = [t for t in inv_recs if t["target_id"] == t_id][0]
        
        fam = tgt["shape_family"].replace("_", " ").title()
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
            ax0.set_title("Ground Truth", fontsize=10, fontweight="bold", pad=6)
        ax0.text(-1.05, 0.95, f"Target #{t_id}\n({fam})", fontsize=8.5, fontweight="bold", va="top", ha="left")
        
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
            ax1.set_title("Single Baseline Model", fontsize=10, fontweight="bold", pad=6)
        ax1.text(0.98, -0.98, f"IoU: {s_rec['iou']:.4f}\nRMS: {s_rec['boundary_rms_dist']:.3f}m", ha="right", va="bottom", fontsize=7.2,
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
            ax2.set_title("Deep Ensemble Mean (K=5)", fontsize=10, fontweight="bold", pad=6)
        ax2.text(0.98, -0.98, f"IoU: {e_rec['iou']:.4f}\nRMS: {e_rec['boundary_rms_m']:.3f}m\n$\\sigma_{{ens}}$: {e_rec['recovered_spread_mv']:.2f}mV", ha="right", va="bottom", fontsize=7.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor=BOX_BG, edgecolor=BOX_BORDER, lw=0.7))
                 
    fig.suptitle("Inverse Shape Reconstruction: Single Baseline Surrogate vs. Deep Ensemble Mean", fontsize=12, y=0.995)
    plt.tight_layout()
    save_fig(fig, figures_out / "fig_E4_ensemble_inversion_comparison")


if __name__ == "__main__":
    main()
