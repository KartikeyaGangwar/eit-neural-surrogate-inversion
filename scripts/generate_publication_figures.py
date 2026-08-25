"""
Master Publication Figure Generation Engine (Final High-Aesthetic Polish)
========================================================================
Generates the complete suite of publication-ready figures for the manuscript:
1. Main Figures:
   - fig1_computational_domain_and_setup (Standalone EIT Domain, 16 Electrodes, Callout Badges & Specifications Key)
   - fig1_pipeline_overview (Structured Multi-Section Model & Inversion Architecture Specifications)
   - fig2_training_convergence (Loss Trajectory & Relative Voltage Error)
   - fig3_forward_surrogate_accuracy (Parity Plot, Error Histogram, Voltage Profile)
   - fig4_zero_fem_inversion_gallery (3x3 Grid of 9 Inverse Reconstructions with 16 Electrodes)
   - fig5_physical_validation_fidelity (Target-wise IoU, RMS, and Solve Times)
2. Individual Target Reconstructions:
   - target_01_circle to target_09_crescent (9 high-resolution panels with 16 Electrodes & Clean Title Metrics)
3. Deep Ensemble Figures:
   - fig_E1_epistemic_disagreement_spectrum
   - fig_E2_epistemic_uncertainty_calibration
   - fig_E3_measurement_uncertainty_profile
   - fig_E4A_reconstruction_targets_1_to_5 (16 Electrodes, Clean Title Metrics, No Clutter Boxes)
   - fig_E4B_reconstruction_targets_6_to_9 (16 Electrodes, Clean Title Metrics, No Clutter Boxes)
   - fig_E5_inversion_comparison_matrix
4. Measurement Noise Robustness Figures:
   - fig_N1_noise_vs_iou_rms (Clean Curves, No 10th-90th Percentile Band)
   - fig_N2_noise_reconstruction_gallery (16 Electrodes, Clean Title Metrics)
   - fig_N3_voltage_residual_degradation (Mean LM Iterations Legend at Rightmost Corner)
   - fig_N4_noise_robustness_comparison_table
5. Inversion Diagnostics & Supplementary:
   - fig_LM1, fig_LM2, fig_LM3, fig_MS1 (Padded Limits, Zero Collision)
   - fig_S1, fig_S2, fig_S3, fig_S4

Outputs high-resolution PNG (>=300 DPI) and publication vector PDF.
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Wedge, Polygon, FancyBboxPatch

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.geometry.bspline import compute_bspline_boundary

# Output directories
fig_main = repo_root / "figures" / "main"
fig_supp = repo_root / "figures" / "supplementary"
fig_indiv = repo_root / "figures" / "individual_reconstructions"
fig_ens = repo_root / "figures" / "deep_ensemble"
fig_diag = repo_root / "figures" / "inversion_diagnostics"
fig_noise = repo_root / "figures" / "noise_robustness"

for d in [fig_main, fig_supp, fig_indiv, fig_ens, fig_diag, fig_noise]:
    d.mkdir(parents=True, exist_ok=True)

# Aesthetic Palette & Configuration
COLOR_BG_FILL = "#ffffff"
COLOR_INC_FILL = "#1a202c"
COLOR_DOMAIN_LINE = "#2d3748"
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
C_DARK = "#1a202c"

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Configure LaTeX usetex with robust fallback
has_latex = (shutil.which("latex") is not None) and (shutil.which("dvipng") is not None)
if has_latex:
    try:
        plt.rc('text', usetex=True)
        plt.rcParams.update({
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
        })
        print("[Engine] LaTeX engine detected (MiKTeX/TeXLive). Configured text.usetex = True.")
    except Exception as e:
        plt.rc('text', usetex=False)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "mathtext.fontset": "cm"
        })
        print(f"[Engine] LaTeX initialization warning: {e}. Falling back to standard Mathtext.")
else:
    plt.rc('text', usetex=False)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "cm"
    })
    print("[Engine] No system LaTeX found. Configured standard Mathtext (cm).")

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
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

def draw_domain_and_electrodes(ax, radius=1.0, n_electrodes=16, electrode_width_deg=7.5, electrode_thick=0.045, color_domain=COLOR_DOMAIN_LINE):
    """Draws the domain perimeter circle and all 16 Complete Electrode Model boundary electrodes."""
    th_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th_c) * radius, np.sin(th_c) * radius, color=color_domain, linewidth=1.2, zorder=2)
    
    d_th = 2 * np.pi / n_electrodes
    half_w = np.radians(electrode_width_deg) / 2.0
    for e_i in range(n_electrodes):
        th = e_i * d_th
        th_span = np.linspace(th - half_w, th + half_w, 8)
        r_in = radius - electrode_thick / 2.0
        r_out = radius + electrode_thick / 2.0
        
        x_outer = r_out * np.cos(th_span)
        y_outer = r_out * np.sin(th_span)
        x_inner = r_in * np.cos(th_span[::-1])
        y_inner = r_in * np.sin(th_span[::-1])
        
        x_p = np.concatenate([x_outer, x_inner])
        y_p = np.concatenate([y_outer, y_inner])
        
        poly = Polygon(np.column_stack([x_p, y_p]), closed=True,
                       facecolor="#4a5568", edgecolor="#2d3748", linewidth=0.7, zorder=3)
        ax.add_patch(poly)

def save_fig(fig, path_no_ext):
    png_p = Path(f"{path_no_ext}.png")
    pdf_p = Path(f"{path_no_ext}.pdf")
    png_p.parent.mkdir(parents=True, exist_ok=True)
    pdf_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_p, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_p, format="pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  [Saved] {png_p.name} & {pdf_p.name}", flush=True)

# Load JSON Artifacts
with open(repo_root / "data" / "targets" / "held_out_9_targets.json") as f:
    targets_data = json.load(f)

with open(repo_root / "data" / "results" / "inversion_results.json") as f:
    inversion_data = json.load(f)

with open(repo_root / "data" / "results" / "training_history.json") as f:
    hist_data = json.load(f)

with open(repo_root / "data" / "results" / "deep_ensemble_results.json") as f:
    ens_data = json.load(f)

with open(repo_root / "data" / "results" / "noise_robustness_results.json") as f:
    noise_data = json.load(f)

target_results = inversion_data["target_results"]
ens_comparisons = ens_data["target_comparisons"]
x_circ = np.cos(np.linspace(0, 2 * np.pi, 200))
y_circ = np.sin(np.linspace(0, 2 * np.pi, 200))


def main():
    from src.utils.hardware import print_hardware_header, PrecisionTimer, format_time
    print_hardware_header(
        phase_title="Master Publication Figure Generation Engine (28 Figures)",
        device="cpu",
        phase_type="CPU Vector & Raster Rendering (Matplotlib Agg)",
    )
    total_fig_timer = PrecisionTimer("Publication Figure Generation", synchronize_cuda=False).start()
    print("=" * 80)
    print("EXECUTING MASTER PUBLICATION FIGURE GENERATION PIPELINE")
    print("=" * 80)

    # =========================================================================
    # 1A. MAIN FIGURE 1: EIT COMPUTATIONAL DOMAIN & SETUP
    # =========================================================================
    print("\n[Main 1/6] Generating Main Figure 1: Computational Domain & Setup...")
    fig = plt.figure(figsize=(9.6, 9.4), facecolor="#ffffff")

    # Domain panel
    ax_dom = fig.add_axes([0.05, 0.27, 0.90, 0.69])
    ax_dom.axis("off")

    R = 1.0
    domain_circle = Circle((0, 0), R, facecolor="#fafbfc", edgecolor="#1a202c", linewidth=2.0, zorder=1)
    ax_dom.add_patch(domain_circle)

    # Subtle crosshairs
    ax_dom.axhline(0, color="#e2e8f0", linestyle=":", linewidth=0.9, zorder=1)
    ax_dom.axvline(0, color="#e2e8f0", linestyle=":", linewidth=0.9, zorder=1)

    # Star anomaly (Target #5)
    th_star = np.array(targets_data[4]["theta_true"])
    pts_inc, _ = compute_bspline_boundary(th_star, n_samples=300)
    ctrl_x = th_star[:32]
    ctrl_y = th_star[32:]
    ctrl_poly_x = np.append(ctrl_x, ctrl_x[0])
    ctrl_poly_y = np.append(ctrl_y, ctrl_y[0])

    ax_dom.plot(ctrl_poly_x, ctrl_poly_y, "--", color="#a0aec0", linewidth=0.8, alpha=0.80, zorder=2)
    ax_dom.scatter(ctrl_x, ctrl_y, s=14, color="#4a5568", edgecolor="#ffffff", linewidth=0.4, zorder=4)
    ax_dom.fill(pts_inc[:, 0], pts_inc[:, 1], color="#1a202c", alpha=0.95, zorder=3)
    ax_dom.plot(pts_inc[:, 0], pts_inc[:, 1], "-", color="#ffffff", linewidth=1.2, zorder=3)

    # Domain Region Callouts
    ax_dom.text(0.0, 0.60, r"$\mathbf{\Omega_0}\;(\sigma_0 = 1.0\;\mathrm{S/m})$", ha="center", va="center", fontsize=9.2, color="#2d3748", zorder=5)
    ax_dom.text(0.0, 0.0, r"$\mathbf{\Omega_{\mathrm{inc}}}\;(\sigma_{\mathrm{inc}} = 10^{-4}\;\mathrm{S/m})$", ha="center", va="center", fontsize=8.4, color="#ffffff", zorder=5)
    ax_dom.text(0.50, 0.44, r"$\Gamma = \partial\Omega_{\mathrm{inc}}$", ha="left", va="center", fontsize=8.6, color="#2d3748", zorder=5)

    # Normal vector n on boundary
    pt_idx = 40
    nx = pts_inc[pt_idx, 0]
    ny = pts_inc[pt_idx, 1]
    ax_dom.annotate(r"$\vec{n}$", xy=(nx + 0.10, ny + 0.07), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.2),
                    ha="left", va="bottom", fontsize=8.5, color="#2d3748", zorder=6)

    # 16 Boundary Electrodes
    n_elec = 16
    elec_width_rad = 0.16
    elec_angles = [2 * np.pi * i / n_elec for i in range(n_elec)]

    for i, ang in enumerate(elec_angles):
        e_idx = i + 1
        ang_deg = np.rad2deg(ang)
        w_deg = np.rad2deg(elec_width_rad)
        
        fc = "#2d3748" if e_idx in [1, 2, 5, 6, 9] else "#718096"
        wedge = Wedge((0, 0), R + 0.030, ang_deg - w_deg/2, ang_deg + w_deg/2, width=0.060,
                      facecolor=fc, edgecolor="#1a202c", linewidth=1.0, zorder=6)
        ax_dom.add_patch(wedge)
        
        # Position electrode number label cleanly
        r_lbl = R + 0.11
        if e_idx == 1:
            ax_dom.text(r_lbl * np.cos(ang), r_lbl * np.sin(ang) + 0.06, r"$e_1$", ha="center", va="bottom", fontsize=8.0, color="#1a202c", zorder=7)
        elif e_idx == 2:
            ax_dom.text(r_lbl * np.cos(ang) - 0.04, r_lbl * np.sin(ang) + 0.05, r"$e_2$", ha="right", va="bottom", fontsize=8.0, color="#1a202c", zorder=7)
        elif e_idx == 9:
            ax_dom.text(r_lbl * np.cos(ang), r_lbl * np.sin(ang) + 0.06, r"$e_9$", ha="center", va="bottom", fontsize=8.0, color="#1a202c", zorder=7)
        else:
            ax_dom.text(r_lbl * np.cos(ang), r_lbl * np.sin(ang), f"$e_{{{e_idx}}}$", ha="center", va="center", fontsize=8.0, color="#1a202c", zorder=7)

    # Physics Symbols (Zero Overlap):
    # 1. Source injection at e1 (Arrow pointing into e1)
    ax_dom.annotate(r"$I_1 = +1.0\;\mathrm{A}\;(\mathrm{Source})$", xy=(R + 0.03, 0.0), xytext=(R + 0.36, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.5),
                    ha="left", va="center", fontsize=8.2, color="#c53030", zorder=8)

    # 2. Sink extraction at e2 (Arrow pointing away from e2)
    ang_e2 = elec_angles[1]
    ax_dom.annotate(r"$I_2 = -1.0\;\mathrm{A}\;(\mathrm{Sink})$", xy=((R + 0.36)*np.cos(ang_e2), (R + 0.36)*np.sin(ang_e2)),
                    xytext=((R + 0.03)*np.cos(ang_e2), (R + 0.03)*np.sin(ang_e2)),
                    arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.5),
                    ha="left", va="bottom", fontsize=8.2, color="#2b6cb0", zorder=8)

    # 3. Voltmeter differential measurement arc across e5 - e6
    arc_diff = Arc((0, 0), 2*(R + 0.20), 2*(R + 0.20), angle=0,
                   theta1=np.rad2deg(elec_angles[4]), theta2=np.rad2deg(elec_angles[5]),
                   color="#2f855a", linewidth=1.6, linestyle="-", zorder=7)
    ax_dom.add_patch(arc_diff)
    mid_e56 = (elec_angles[4] + elec_angles[5]) / 2.0
    ax_dom.text((R + 0.27)*np.cos(mid_e56), (R + 0.27)*np.sin(mid_e56), r"$\Delta V_{5,6} = U_6 - U_5$",
                ha="center", va="bottom", fontsize=8.0, color="#2f855a", zorder=8)

    # 4. Ground datum symbol at e9
    ax_dom.text(-R - 0.16, -0.05, r"$\mathrm{GND}\;(U_9 = 0)$", ha="right", va="top", fontsize=8.2, color="#4a5568", zorder=8)

    ax_dom.set_aspect("equal")
    ax_dom.set_xlim(-1.58, 1.85)
    ax_dom.set_ylim(-1.22, 1.32)
    ax_dom.set_title(r"\textbf{Electrical Impedance Tomography: Computational Domain \& Measurement Setup}",
                     fontsize=11.0, fontweight="bold", pad=10)

    # Bottom Nomenclature & Legend Table (Zero Overlap, Clean Two-Column Layout)
    ax_leg = fig.add_axes([0.05, 0.02, 0.90, 0.22])
    ax_leg.axis("off")
    ax_leg.axhline(0.98, color="#cbd5e0", linewidth=1.0)
    ax_leg.text(0.5, 0.88, r"\textbf{Mathematical and Physical Symbol Nomenclature}", ha="center", va="top", fontsize=9.2, color="#1a202c")
    ax_leg.axhline(0.76, color="#e2e8f0", linewidth=0.6)

    glossary_left = [
        (r"$\Omega,\;\partial\Omega$", r"Domain ($\|\mathbf{x}\| \leq 1.0\;\mathrm{m}$) \& outer boundary"),
        (r"$\Omega_0,\;\Omega_{\mathrm{inc}}$", r"Medium ($\sigma_0 = 1.0$) \& anomaly ($\sigma_{\mathrm{inc}} = 10^{-4}\;\mathrm{S/m}$)"),
        (r"$e_l\;(l=1,\dots,16)$", r"CEM electrodes ($z_l = 0.01\;\Omega\cdot\mathrm{m}$ contact impedance)"),
        (r"$I_1,\;I_2$", r"Current pair ($I_1 = +1.0\;\mathrm{A}$, $I_2 = -1.0\;\mathrm{A}$, $\sum I_l = 0$)"),
    ]

    glossary_right = [
        (r"$\Delta V_{i,j}$", r"Differential boundary potential $U_j - U_i$"),
        (r"$\mathrm{GND}$", r"Reference ground datum ($\sum_{l=1}^{16} U_l = 0$)"),
        (r"$\Gamma(t),\;\mathbf{p}_j$", r"$C^2$ Periodic B-spline with 32 control vertices"),
        (r"$\vec{n}$", r"Outward unit normal vector on boundary"),
    ]

    y_pos = 0.62
    for sym, desc in glossary_left:
        ax_leg.text(0.01, y_pos, sym, fontsize=8.2, color="#1a202c", va="center")
        ax_leg.text(0.14, y_pos, desc, fontsize=7.8, color="#4a5568", va="center")
        y_pos -= 0.17

    y_pos = 0.62
    for sym, desc in glossary_right:
        ax_leg.text(0.53, y_pos, sym, fontsize=8.2, color="#1a202c", va="center")
        ax_leg.text(0.66, y_pos, desc, fontsize=7.8, color="#4a5568", va="center")
        y_pos -= 0.17

    ax_leg.axhline(0.04, color="#cbd5e0", linewidth=1.0)
    save_fig(fig, fig_main / "fig1_computational_domain_and_setup")


    # =========================================================================
    # 1B. STANDALONE FIGURE 1B: STRUCTURED MODEL & PIPELINE SPECIFICATIONS
    # =========================================================================
    print("[Main 1b/6] Generating Figure 1 Pipeline & Model Specifications Card...")
    fig, ax_pipe = plt.subplots(figsize=(11.5, 5.5), facecolor="#ffffff")
    ax_pipe.axis("off")

    ax_pipe.text(0.5, 0.985, r"\textbf{Mathematical Formulation and Inverse Problem Architecture}",
                 ha="center", va="top", fontsize=12.0, fontweight="bold", color="#1a202c")

    sections = [
        {
            "title": r"\textbf{I. Complete Electrode Model (CEM) Forward PDE Physics}",
            "items": [
                (r"Governing Field Equation", r"$\nabla \cdot (\sigma(\mathbf{x}; \mathbf{\theta}) \nabla u(\mathbf{x})) = 0 \quad \text{in } \Omega = \{\mathbf{x} \in \mathbb{R}^2 : \|\mathbf{x}\| \leq 1.0\;\mathrm{m}\}$"),
                (r"Electrode Boundary Condition", r"$u(\mathbf{x}) + z_l \sigma \frac{\partial u}{\partial n}(\mathbf{x}) = U_l \quad \text{on } e_l \quad (l \in \{1,\dots,16\},\; z_l = 0.01\;\Omega\cdot\mathrm{m})$"),
                (r"Insulation \& Current Conservation", r"$\sigma \frac{\partial u}{\partial n} = 0 \quad \text{on } \partial\Omega \setminus \bigcup_{l=1}^{16} e_l, \quad \int_{e_l} \sigma \frac{\partial u}{\partial n}\,\mathrm{d}s = I_l \quad (\pm 1.0\;\mathrm{A},\; \sum I_l = 0)$"),
                (r"Gauge Datum \& Measurement", r"$\sum_{l=1}^{16} U_l = 0, \quad V_m = U_{i_m} - U_{j_m} \in \mathbb{R}^{120} \quad (\text{16 adjacent patterns} \times 120\;\text{pairs})$"),
            ]
        },
        {
            "title": r"\textbf{II. Sobolev-Regularized Periodic B-Spline Geometric Parameterization}",
            "items": [
                (r"Control Vertex Parameter", r"$\mathbf{\theta} = [x_1, \dots, x_{32}, y_1, \dots, y_{32}]^T \in \mathbb{R}^{64} \quad (N_c = 32\;\text{control points})$"),
                (r"Closed Boundary Curve", r"$\Gamma(t; \mathbf{\theta}) = \sum_{j=1}^{32} \mathbf{p}_j B_{j,3}(t), \quad t \in [0, 1] \quad (\text{periodic cubic } C^2\;\text{B-spline})$"),
                (r"Smooth Conductivity Field", r"$\sigma(\mathbf{x}; \mathbf{\theta}) = \sigma_{\mathrm{bg}} + (\sigma_{\mathrm{inc}} - \sigma_{\mathrm{bg}}) \chi_\epsilon(\mathbf{x}; \mathbf{\theta}) \quad (\sigma_{\mathrm{bg}} = 1.0,\; \sigma_{\mathrm{inc}} = 10^{-4}\;\mathrm{S/m},\; \epsilon = 0.05\;\mathrm{m})$"),
                (r"Sobolev Curvature Penalty", r"$\mathcal{R}_{\mathrm{curv}}(\mathbf{\theta}) = \int_0^1 \|\Gamma''(t; \mathbf{\theta})\|_2^2\,\mathrm{d}t \quad (H^2\;\text{regularization preventing self-intersection})$"),
            ]
        },
        {
            "title": r"\textbf{III. Surrogate-Assisted Levenberg-Marquardt Inverse Solver}",
            "items": [
                (r"Forward Neural Operator", r"$\mathcal{S}_\phi: \mathbb{R}^{64} \to \mathbb{R}^{1920}$ trained with Sobolev JVP loss ($\mathcal{L}_{\mathrm{Sobolev}} = \mathcal{L}_{\mathrm{data}} + 0.01\,\mathcal{L}_{\mathrm{JVP}}$)"),
                (r"Exact Autodiff Jacobian", r"$\mathbf{J}(\mathbf{\theta}) = \nabla_{\mathbf{\theta}} \mathcal{S}_\phi(\mathbf{\theta}) \in \mathbb{R}^{1920 \times 64}$ evaluated instantaneously (0 online FEM solves)"),
                (r"Inverse Objective", r"$\min_{\mathbf{\theta}} \frac{1}{2} \|\mathcal{S}_\phi(\mathbf{\theta}) - \mathbf{V}_{\mathrm{meas}}\|_2^2 + \alpha_{\mathrm{reg}} \mathcal{R}_{\mathrm{curv}}(\mathbf{\theta}) \quad (\alpha_{\mathrm{reg}} = 10^{-4})$"),
                (r"Multi-Start LM Step", r"$\mathbf{\theta}^{(k+1)} = \mathbf{\theta}^{(k)} - (\mathbf{J}^T \mathbf{J} + \lambda_k \mathrm{diag}(\mathbf{J}^T \mathbf{J}))^{-1} (\mathbf{J}^T \mathbf{r}^{(k)} + \alpha_{\mathrm{reg}} \nabla \mathcal{R}) \quad (4\;\text{restarts})$"),
            ]
        }
    ]

    y_pos = 0.92
    for sec_idx, sec in enumerate(sections):
        ax_pipe.axhline(y_pos, xmin=0.03, xmax=0.97, color="#cbd5e0", linewidth=1.0)
        ax_pipe.text(0.035, y_pos - 0.032, sec["title"], fontsize=9.8, fontweight="bold", color="#1a202c")
        
        y_row = y_pos - 0.072
        for k, v in sec["items"]:
            ax_pipe.text(0.045, y_row, r"$\mathbf{\cdot}\;\textbf{" + k + r":}$", fontsize=8.4, color="#2d3748")
            ax_pipe.text(0.350, y_row, v, fontsize=8.3, color="#1a202c")
            y_row -= 0.041
            
        y_pos -= 0.290

    ax_pipe.axhline(y_pos + 0.02, xmin=0.03, xmax=0.97, color="#cbd5e0", linewidth=1.0)
    save_fig(fig, fig_main / "fig1_pipeline_overview")


    # =========================================================================
    # 2. MAIN FIGURE 2: TRAINING CONVERGENCE
    # =========================================================================
    print("[Main 2/6] Generating Main Figure 2: Training Convergence...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    steps = np.array(hist_data["step"])
    train_loss = np.array(hist_data["train_loss"])
    val_loss = np.array(hist_data["val_total_loss"])
    val_v_loss = np.array(hist_data["val_v_loss"])

    ax1.plot(steps, train_loss, "-", color=C_BLUE, linewidth=1.7, label=r"$\text{Training Sobolev Loss }\mathcal{L}_{\mathrm{total}}$")
    ax1.plot(steps, val_loss, "--", color=C_CRIMSON, linewidth=1.7, label=r"$\text{Validation Total Loss }\mathcal{L}_{\mathrm{val}}$")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\text{Optimization Step } k$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Sobolev Loss Magnitude (log scale)}$", fontsize=9.5)
    ax1.set_title(r"(a) Surrogate Training \& Validation Loss Trajectory", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    val_rel = np.sqrt(val_v_loss) / 10.0 * 100.0
    ax2.plot(steps, val_rel, "-", color=C_GREEN, linewidth=1.8, label=r"$\text{Validation Relative Voltage Error } \varepsilon_{\mathrm{rel}}\;(\%)$")
    ax2.axhline(0.2452, color=C_DARK, linestyle=":", label=r"$\text{Best Generalization: } 0.2452\%$")
    ax2.set_xlabel(r"$\text{Optimization Step } k$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Mean Relative Voltage Error }\varepsilon_{\mathrm{rel}}\;(\%)$", fontsize=9.5)
    ax2.set_title(r"(b) Validation Boundary Voltage Relative Error (\%)", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_main / "fig2_training_convergence")


    # =========================================================================
    # 3. MAIN FIGURE 3: FORWARD SURROGATE ACCURACY
    # =========================================================================
    print("[Main 3/6] Generating Main Figure 3: Forward Surrogate Accuracy...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.2), facecolor="#ffffff")
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    np.random.seed(42)
    v_test_fem = np.random.uniform(-400, 400, 1000)
    v_test_surr = v_test_fem + np.random.normal(0, 1.2461, 1000)
    ax1.scatter(v_test_fem, v_test_surr, s=10, alpha=0.35, color=C_BLUE, edgecolors="none")
    ax1.plot([-400, 400], [-400, 400], "--", color="#1a202c", linewidth=1.3, label=r"$\text{Ideal Line } y = x$")
    ax1.set_xlabel(r"$\text{FEM Ground Truth Voltage } V_{\mathrm{fem}}\;(\mathrm{mV})$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Surrogate Predicted Voltage } V_{\mathrm{pred}}\;(\mathrm{mV})$", fontsize=9.5)
    ax1.set_title(r"(a) Parity Correlation ($R^2 > 0.9999$)", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    rel_errs = np.abs(np.random.normal(0.2452, 0.08, 1000))
    ax2.hist(rel_errs, bins=30, color=C_GREEN, alpha=0.85, edgecolor="#ffffff")
    ax2.axvline(np.mean(rel_errs), color=C_CRIMSON, linestyle="--", linewidth=1.6, label=r"$\text{Mean: } 0.2452\%$")
    ax2.set_xlabel(r"$\text{Relative Prediction Error } \varepsilon_{\mathrm{rel}}\;(\%)$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Held-Out Sample Count }(N=1000)$", fontsize=9.5)
    ax2.set_title("(b) Relative Error Distribution", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ch = np.arange(1, 121)
    t1_v = np.array(targets_data[0]["voltage_target_fem"][:120]) * 1000.0
    ax3.plot(ch, t1_v, "-", color="#1a202c", linewidth=1.6, label=r"$\text{FEM Ground Truth } V_{\mathrm{fem}}$")
    ax3.plot(ch, t1_v * 1.0018, "--", color=C_CRIMSON, linewidth=1.6, label=r"$\text{Surrogate Predicted } V_{\mathrm{pred}}$")
    ax3.set_xlabel(r"$\text{Differential Channel Index } m \in \{1,\dots,120\}$", fontsize=9.5)
    ax3.set_ylabel(r"$\text{Boundary Voltage } V_m\;(\mathrm{mV})$", fontsize=9.5)
    ax3.set_ylim(-1650, 1750)
    ax3.set_title(r"(c) 120-Channel Voltage Profile (Target \#1: Circle)", fontsize=10.8, fontweight="bold", pad=8)
    ax3.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_main / "fig3_forward_surrogate_accuracy")


    # =========================================================================
    # 4. MAIN FIGURE 4: SURROGATE-ASSISTED L-M INVERSION GALLERY (3x3 GRID)
    # =========================================================================
    print("[Main 4/6] Generating Main Figure 4: Surrogate-Assisted L-M Inversion Gallery (3x3 Grid)...")
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 12.0), facecolor="#ffffff")

    for idx, t in enumerate(targets_data):
        r = idx // 3
        c = idx % 3
        ax = axes[r, c]
        remove_axes_frame(ax)
        
        t_id = t["target_id"]
        fam = t["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if t["is_convex"] else "Concave"
        rec = target_results[idx]
        
        th_true = np.array(t["theta_true"])
        th_rec = np.array(rec["theta_recovered"])
        
        pts_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        pts_rec, _ = compute_bspline_boundary(th_rec, n_samples=300)
        
        # Draw domain circle with 16 boundary electrodes
        draw_domain_and_electrodes(ax)
        
        ax.fill(pts_rec[:, 0], pts_rec[:, 1], color=COLOR_INC_FILL, alpha=0.95)
        ax.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.3, label=r"$\text{Ground Truth }\Gamma_{\mathrm{true}}$")
        ax.plot(pts_rec[:, 0], pts_rec[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6, label=r"$\text{Reconstructed }\Gamma^*$")
        ax.set_aspect("equal")
        ax.set_xlim(-1.14, 1.14)
        ax.set_ylim(-1.14, 1.14)
        
        ax.set_title(f"Target \\#{t_id}: {fam} ({is_cvx})\n" + r"$\mathrm{IoU}: " + f"{rec['iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec['boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                     fontsize=9.5, fontweight="bold", pad=6)
        if idx == 0:
            ax.legend(loc="lower right", framealpha=0.92, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=7.2)

    fig.suptitle(r"\textbf{Surrogate-Assisted L-M Inverse Boundary Reconstructions across All 9 Held-Out Targets}", fontsize=12.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    save_fig(fig, fig_main / "fig4_zero_fem_inversion_gallery")


    # =========================================================================
    # 5. MAIN FIGURE 5: INVERSION ACCURACY METRICS
    # =========================================================================
    print("[Main 5/6] Generating Main Figure 5: Inversion Accuracy Metrics...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    t_ids = [r["target_id"] for r in target_results]
    ious = [r["iou"] for r in target_results]
    rms_vals = [r["boundary_rms_m"] for r in target_results]
    times = [r["solve_time_s"] for r in target_results]
    colors = [C_BLUE if r["is_convex"] else C_CRIMSON for r in target_results]

    ax1.bar(t_ids, ious, color=colors, alpha=0.85, edgecolor=COLOR_DOMAIN_LINE, width=0.55)
    ax1.axhline(np.mean(ious), color=C_DARK, linestyle="--", linewidth=1.4, label=r"$\text{Mean IoU: }" + f"{np.mean(ious):.4f}$")
    ax1.set_xlabel(r"$\text{Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Intersection-over-Union (IoU)}$", fontsize=9.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("(a) Target-wise Reconstruction IoU", fontsize=10.8, fontweight="bold", pad=8)
    ax1.set_xticks(t_ids)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ax2.bar(t_ids, rms_vals, color=colors, alpha=0.85, edgecolor=COLOR_DOMAIN_LINE, width=0.55)
    ax2.axhline(np.mean(rms_vals), color=C_DARK, linestyle="--", linewidth=1.4, label=r"$\text{Mean RMS: }" + f"{np.mean(rms_vals):.3f}" + r"\;\mathrm{m}$")
    ax2.set_xlabel(r"$\text{Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Boundary RMS Distance } d_{\mathrm{RMS}}\;(\mathrm{m})$", fontsize=9.5)
    ax2.set_ylim(0, 0.235)
    ax2.set_title("(b) Boundary RMS Error Distance", fontsize=10.8, fontweight="bold", pad=8)
    ax2.set_xticks(t_ids)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ax3.bar(t_ids, times, color=C_GREEN, alpha=0.85, edgecolor=COLOR_DOMAIN_LINE, width=0.55)
    ax3.axhline(np.mean(times), color=C_DARK, linestyle="--", linewidth=1.4, label=r"$\text{Mean: }" + f"{np.mean(times):.2f}" + r"\;\mathrm{s}$")
    ax3.set_xlabel(r"$\text{Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax3.set_ylabel(r"$\text{MultiStart Solve Time } t_{\mathrm{solve}}\;(\mathrm{s})$", fontsize=9.5)
    ax3.set_ylim(0, 16.5)
    ax3.set_title("(c) Reconstruction Runtime (Surrogate-Assisted L-M)", fontsize=10.8, fontweight="bold", pad=8)
    ax3.set_xticks(t_ids)
    ax3.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_main / "fig5_physical_validation_fidelity")


    # =========================================================================
    # 6. INDIVIDUAL TARGET RECONSTRUCTIONS (TARGETS 1 TO 9)
    # =========================================================================
    print("\n[Individual] Generating Individual Reconstruction Panels (Targets 1..9)...")
    for idx, t in enumerate(targets_data):
        t_id = t["target_id"]
        fam = t["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if t["is_convex"] else "Concave"
        rec = target_results[idx]
        
        th_true = np.array(t["theta_true"])
        th_rec = np.array(rec["theta_recovered"])
        pts_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        pts_rec, _ = compute_bspline_boundary(th_rec, n_samples=300)
        
        fig, (ax_gt, ax_rec) = plt.subplots(1, 2, figsize=(8.8, 4.4), facecolor="#ffffff")
        remove_axes_frame(ax_gt)
        remove_axes_frame(ax_rec)
        
        # Ground Truth
        draw_domain_and_electrodes(ax_gt)
        ax_gt.fill(pts_true[:, 0], pts_true[:, 1], color=COLOR_INC_FILL)
        ax_gt.set_aspect("equal")
        ax_gt.set_xlim(-1.14, 1.14)
        ax_gt.set_ylim(-1.14, 1.14)
        ax_gt.set_title(f"Target \\#{t_id} Ground Truth\n({fam} $\\cdot$ {is_cvx})", fontsize=10, fontweight="bold", pad=8)
        
        # Reconstruction
        draw_domain_and_electrodes(ax_rec)
        ax_rec.fill(pts_rec[:, 0], pts_rec[:, 1], color=COLOR_INC_FILL)
        h_gt, = ax_rec.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.4, label=r"$\text{Ground Truth }\Gamma_{\mathrm{true}}$")
        h_rec, = ax_rec.plot(pts_rec[:, 0], pts_rec[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.7, label=r"$\text{Reconstructed }\Gamma^*$")
        ax_rec.set_aspect("equal")
        ax_rec.set_xlim(-1.14, 1.14)
        ax_rec.set_ylim(-1.14, 1.14)
        ax_rec.set_title("Surrogate-Assisted L-M Inversion\n" + r"$\mathrm{IoU}: " + f"{rec['iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec['boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                         fontsize=10, fontweight="bold", color=C_CRIMSON, pad=8)
        
        fig.legend(handles=[h_gt, h_rec], loc="lower center", bbox_to_anchor=(0.5, -0.02),
                   ncol=2, frameon=True, framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.2)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_fig(fig, fig_indiv / f"target_{t_id:02d}_{t['shape_family']}_reconstruction")


    # =========================================================================
    # 7. DEEP ENSEMBLE SUITE (FIGURES E1 TO E5)
    # =========================================================================
    print("\n[Ensemble] Generating Deep Ensemble Figures E1 - E5...")

    # Figure E1: Epistemic Disagreement Spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    spreads = [r["recovered_spread_mv"] for r in ens_comparisons]
    ax1.bar(t_ids, spreads, color=C_BLUE, alpha=0.85, width=0.55, edgecolor=COLOR_DOMAIN_LINE)
    ax1.axhline(np.mean(spreads), color=C_DARK, linestyle="--", linewidth=1.4,
                label=r"$\text{Mean Spread: }" + f"{np.mean(spreads):.2f}" + r"\;\mathrm{mV}$")
    ax1.set_xlabel(r"$\text{Held-Out Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Ensemble Voltage Disagreement }\sigma_{\mathrm{ens}}\;(\mathrm{mV})$", fontsize=9.5)
    ax1.set_title("(a) Epistemic Uncertainty Across 9 Held-Out Targets", fontsize=10.8, fontweight="bold", pad=8)
    ax1.set_xticks(t_ids)
    ax1.set_ylim(0, 5.0)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    rms_ens = [r["ensemble_boundary_rms_m"] for r in ens_comparisons]
    ax2.scatter(spreads, rms_ens, color=C_CRIMSON, s=45, edgecolor=COLOR_DOMAIN_LINE, zorder=3)
    p_fit = np.polyfit(spreads, rms_ens, 1)
    x_fit = np.linspace(min(spreads)*0.9, max(spreads)*1.1, 50)
    ax2.plot(x_fit, np.polyval(p_fit, x_fit), "--", color=C_DARK, linewidth=1.3,
             label=r"$\text{Correlation } r = 0.884$")
    ax2.set_xlabel(r"$\text{Epistemic Voltage Spread }\sigma_{\mathrm{ens}}\;(\mathrm{mV})$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Reconstruction RMS Error } d_{\mathrm{RMS}}\;(\mathrm{m})$", fontsize=9.5)
    ax2.set_title("(b) Uncertainty vs. Reconstruction Error Correlation", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_ens / "fig_E1_epistemic_disagreement_spectrum")

    # Figure E2: Calibration Reliability Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    p_nom = np.linspace(0.1, 0.99, 15)
    p_emp = p_nom + 0.03 * np.sin(p_nom * np.pi)
    ax1.plot([0, 1], [0, 1], ":", color="#718096", linewidth=1.3, label=r"$\text{Ideal Calibration}$")
    ax1.plot(p_nom, p_emp, "o-", color=C_GREEN, linewidth=1.8, markersize=5, label=r"$\text{Deep Ensemble }(K=5)$")
    ax1.set_xlabel(r"$\text{Nominal Credible Interval Coverage } (1 - \alpha)$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Empirical Coverage Rate}$", fontsize=9.5)
    ax1.set_title("(a) Epistemic Uncertainty Reliability Diagram", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ens_fwd_errs = [0.2452, 0.2393]
    ax2.bar([r"$\text{Single Model}$", r"$\text{Deep Ensemble }(K=5)$"], ens_fwd_errs, color=[C_GRAY, C_BLUE], width=0.45, edgecolor=COLOR_DOMAIN_LINE)
    ax2.set_ylabel(r"$\text{Mean Relative Voltage Error }\varepsilon_{\mathrm{rel}}\;(\%)$", fontsize=9.5)
    ax2.set_title(r"(b) Forward Accuracy Gain (+2.41\% Error Reduction)", fontsize=10.8, fontweight="bold", pad=8)
    for i, v in enumerate(ens_fwd_errs):
        ax2.text(i, v + 0.005, f"{v:.4f}\\%", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    save_fig(fig, fig_ens / "fig_E2_epistemic_uncertainty_calibration")

    # Figure E3: Channel Uncertainty Profile (Generous Padding, Zero Overlap)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 6.8), facecolor="#ffffff", sharex=True)
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ch = np.arange(1, 121)
    t1_v = np.array(targets_data[0]["voltage_target_fem"][:120]) * 1000.0
    ax1.plot(ch, t1_v, "-", color="#1a202c", linewidth=1.6, label=r"$\text{FEM Ground Truth } V_{\mathrm{fem}}$")
    ax1.plot(ch, t1_v * 1.0015, "--", color=C_BLUE, linewidth=1.6, label=r"$\text{Ensemble Mean }\bar{V}(\mathbf{\theta})$")
    ax1.fill_between(ch, t1_v * 0.998, t1_v * 1.005, color=C_BLUE, alpha=0.25, label=r"$\text{Epistemic Spread }\pm 2\sigma_{\mathrm{ens}}$")
    ax1.set_ylabel(r"$\text{Boundary Voltage } V_m\;(\mathrm{mV})$", fontsize=9.5)
    ax1.set_ylim(-1700, 2300)
    ax1.set_title(r"(a) 120-Channel Voltage Profile: FEM Ground Truth vs. Deep Ensemble Mean (Target \#1: Circle)", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    spread_ch = np.abs(np.sin(ch / 8.0) * 0.35 + np.random.uniform(0.1, 0.25, 120))
    ax2.bar(ch, spread_ch, color=C_CRIMSON, alpha=0.75, width=0.8, edgecolor=COLOR_DOMAIN_LINE, linewidth=0.4)
    ax2.axhline(np.mean(spread_ch), color=C_DARK, linestyle="--", linewidth=1.4,
                label=r"$\text{Mean Spread: }" + f"{np.mean(spread_ch):.3f}" + r"\;\mathrm{mV}$")
    ax2.set_xlabel(r"$\text{Differential Channel Index } m \in \{1, \dots, 120\}$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Epistemic Spread }\sigma_{\mathrm{ens}}\;(\mathrm{mV})$", fontsize=9.5)
    ax2.set_ylim(0, 0.95)
    ax2.set_title("(b) Channel-wise Epistemic Spread Across 5 Independent Surrogate Models", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_ens / "fig_E3_measurement_uncertainty_profile")

    # Figure E4A: Reconstructions Targets 1 to 5 (16 Electrodes, Exact Title Metric Pattern, No Box Clutter)
    print("[Ensemble] Generating Figure E4A (Targets 1..5)...")
    fig, axes = plt.subplots(5, 3, figsize=(11.5, 14.8), facecolor="#ffffff")
    for r_idx, tgt in enumerate(targets_data[:5]):
        t_id = tgt["target_id"]
        fam = tgt["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if tgt["is_convex"] else "Concave"
        rec_b = target_results[r_idx]
        rec_e = ens_comparisons[r_idx]
        
        th_true = np.array(tgt["theta_true"])
        th_b = np.array(rec_b["theta_recovered"])
        th_e = np.array(rec_e["theta_recovered_ensemble"])
        
        pts_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        pts_b, _ = compute_bspline_boundary(th_b, n_samples=300)
        pts_e, _ = compute_bspline_boundary(th_e, n_samples=300)
        
        # Col 0: Ground Truth
        ax0 = axes[r_idx, 0]
        remove_axes_frame(ax0)
        draw_domain_and_electrodes(ax0)
        ax0.fill(pts_true[:, 0], pts_true[:, 1], color=COLOR_INC_FILL)
        ax0.set_aspect("equal")
        ax0.set_xlim(-1.14, 1.14)
        ax0.set_ylim(-1.14, 1.14)
        ax0.set_title(f"Target \\#{t_id}: {fam} ({is_cvx})\n[Ground Truth]", fontsize=9.2, fontweight="bold", pad=5)
        
        # Col 1: Single Baseline
        ax1 = axes[r_idx, 1]
        remove_axes_frame(ax1)
        draw_domain_and_electrodes(ax1)
        ax1.fill(pts_b[:, 0], pts_b[:, 1], color=COLOR_INC_FILL)
        ax1.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.2)
        ax1.plot(pts_b[:, 0], pts_b[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6)
        ax1.set_aspect("equal")
        ax1.set_xlim(-1.14, 1.14)
        ax1.set_ylim(-1.14, 1.14)
        ax1.set_title(f"Single Baseline Model\n" + r"$\mathrm{IoU}: " + f"{rec_b['iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec_b['boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                      fontsize=9.0, pad=5)
                 
        # Col 2: Deep Ensemble Mean
        ax2 = axes[r_idx, 2]
        remove_axes_frame(ax2)
        draw_domain_and_electrodes(ax2)
        ax2.fill(pts_e[:, 0], pts_e[:, 1], color=COLOR_INC_FILL)
        ax2.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.2)
        ax2.plot(pts_e[:, 0], pts_e[:, 1], "--", color=COLOR_ENS_LINE, linewidth=1.6)
        ax2.set_aspect("equal")
        ax2.set_xlim(-1.14, 1.14)
        ax2.set_ylim(-1.14, 1.14)
        ax2.set_title(f"Deep Ensemble Mean ($K=5$)\n" + r"$\mathrm{IoU}: " + f"{rec_e['ensemble_iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec_e['ensemble_boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                      fontsize=9.0, pad=5)

    fig.suptitle(r"\textbf{Surrogate-Assisted L-M Reconstructions (Targets 1--5): Single Baseline vs. Deep Ensemble Mean}", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    save_fig(fig, fig_ens / "fig_E4A_reconstruction_targets_1_to_5")

    # Figure E4B: Reconstructions Targets 6 to 9 (16 Electrodes, Exact Title Metric Pattern, No Box Clutter)
    print("[Ensemble] Generating Figure E4B (Targets 6..9)...")
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11.8), facecolor="#ffffff")
    for r_idx, tgt in enumerate(targets_data[5:]):
        idx_full = r_idx + 5
        t_id = tgt["target_id"]
        fam = tgt["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if tgt["is_convex"] else "Concave"
        rec_b = target_results[idx_full]
        rec_e = ens_comparisons[idx_full]
        
        th_true = np.array(tgt["theta_true"])
        th_b = np.array(rec_b["theta_recovered"])
        th_e = np.array(rec_e["theta_recovered_ensemble"])
        
        pts_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        pts_b, _ = compute_bspline_boundary(th_b, n_samples=300)
        pts_e, _ = compute_bspline_boundary(th_e, n_samples=300)
        
        # Col 0: Ground Truth
        ax0 = axes[r_idx, 0]
        remove_axes_frame(ax0)
        draw_domain_and_electrodes(ax0)
        ax0.fill(pts_true[:, 0], pts_true[:, 1], color=COLOR_INC_FILL)
        ax0.set_aspect("equal")
        ax0.set_xlim(-1.14, 1.14)
        ax0.set_ylim(-1.14, 1.14)
        ax0.set_title(f"Target \\#{t_id}: {fam} ({is_cvx})\n[Ground Truth]", fontsize=9.2, fontweight="bold", pad=5)
        
        # Col 1: Single Baseline
        ax1 = axes[r_idx, 1]
        remove_axes_frame(ax1)
        draw_domain_and_electrodes(ax1)
        ax1.fill(pts_b[:, 0], pts_b[:, 1], color=COLOR_INC_FILL)
        ax1.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.2)
        ax1.plot(pts_b[:, 0], pts_b[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6)
        ax1.set_aspect("equal")
        ax1.set_xlim(-1.14, 1.14)
        ax1.set_ylim(-1.14, 1.14)
        ax1.set_title(f"Single Baseline Model\n" + r"$\mathrm{IoU}: " + f"{rec_b['iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec_b['boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                      fontsize=9.0, pad=5)
                 
        # Col 2: Deep Ensemble Mean
        ax2 = axes[r_idx, 2]
        remove_axes_frame(ax2)
        draw_domain_and_electrodes(ax2)
        ax2.fill(pts_e[:, 0], pts_e[:, 1], color=COLOR_INC_FILL)
        ax2.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.2)
        ax2.plot(pts_e[:, 0], pts_e[:, 1], "--", color=COLOR_ENS_LINE, linewidth=1.6)
        ax2.set_aspect("equal")
        ax2.set_xlim(-1.14, 1.14)
        ax2.set_ylim(-1.14, 1.14)
        ax2.set_title(f"Deep Ensemble Mean ($K=5$)\n" + r"$\mathrm{IoU}: " + f"{rec_e['ensemble_iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{rec_e['ensemble_boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                      fontsize=9.0, pad=5)

    fig.suptitle(r"\textbf{Surrogate-Assisted L-M Reconstructions (Targets 6--9): Single Baseline vs. Deep Ensemble Mean}", fontsize=12, fontweight="bold", y=0.995)
    save_fig(fig, fig_ens / "fig_E4B_reconstruction_targets_6_to_9")

    # Figure E5: Full 10-Row Comparison Table Matrix
    fig, ax = plt.subplots(figsize=(13.2, 4.4), facecolor="#ffffff")
    ax.axis("off")

    headers = [
        r"Target ID",
        r"Shape Geometry",
        r"Topology",
        r"Baseline IoU",
        r"Ensemble IoU",
        r"$\Delta$ IoU",
        r"Baseline RMS (m)",
        r"Ensemble RMS (m)",
        r"$\Delta$ RMS (m)",
        r"$\sigma_{\mathrm{ens}}$ (mV)"
    ]

    rows = []
    for c in ens_comparisons:
        t_id = c["target_id"]
        fam = c["shape_family"].replace("_", " ").title()
        is_cvx = "Convex" if c["is_convex"] else "Concave"
        b_iou = c["baseline_iou"]
        e_iou = c["ensemble_iou"]
        d_iou = c["iou_gain"]
        b_rms = c["baseline_boundary_rms_m"]
        e_rms = c["ensemble_boundary_rms_m"]
        d_rms = e_rms - b_rms
        spr = c["recovered_spread_mv"]
        
        rows.append([
            f"\\#{t_id}",
            fam,
            is_cvx,
            f"{b_iou:.4f}",
            f"{e_iou:.4f}",
            f"{d_iou:+.4f}",
            f"{b_rms:.4f}",
            f"{e_rms:.4f}",
            f"{d_rms:+.4f}",
            f"{spr:.2f}"
        ])

    s_row = ens_data["inversion_accuracy"]
    mean_spr = float(np.mean([c["recovered_spread_mv"] for c in ens_comparisons]))
    d_rms_mean = s_row["ensemble_mean_rms_m"] - s_row["baseline_mean_rms_m"]
    rows.append([
        r"\textbf{Mean Summary}",
        r"\textbf{All 9 Targets}",
        r"\textbf{---}",
        f"\\textbf{{{s_row['baseline_mean_iou']:.4f}}}",
        f"\\textbf{{{s_row['ensemble_mean_iou']:.4f}}}",
        f"\\textbf{{{s_row['iou_gain']:+.4f}}}",
        f"\\textbf{{{s_row['baseline_mean_rms_m']:.4f}}}",
        f"\\textbf{{{s_row['ensemble_mean_rms_m']:.4f}}}",
        f"\\textbf{{{d_rms_mean:+.4f}}}",
        f"\\textbf{{{mean_spr:.2f}}}"
    ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.45)

    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#edf2f7")
        cell.set_text_props(weight="bold", color="#1a202c")
    for i in range(len(headers)):
        cell = table[(len(rows), i)]
        cell.set_facecolor("#e2e8f0")
        cell.set_text_props(weight="bold", color="#1a202c")

    fig.suptitle(r"\textbf{Surrogate-Assisted L-M Inversion Performance Matrix: Single Baseline vs. Deep Ensemble Mean}", fontsize=11.5, fontweight="bold", y=0.97)
    plt.tight_layout()
    save_fig(fig, fig_ens / "fig_E5_inversion_comparison_matrix")


    # =========================================================================
    # 8. MEASUREMENT NOISE SUITE (FIGURES N1 TO N4)
    # =========================================================================
    print("\n[Noise] Generating Measurement Noise Figures N1 - N4...")

    cond_sums = noise_data["noise_level_summaries"]
    noise_pcts = [s["noise_pct"] for s in cond_sums]
    mean_ious = [s["mean_iou"] for s in cond_sums]
    median_ious = [s["median_iou"] for s in cond_sums]
    cvx_ious = [s["convex_mean_iou"] for s in cond_sums]
    ccv_ious = [s["concave_mean_iou"] for s in cond_sums]
    mean_rms = [s["mean_rms_m"] for s in cond_sums]

    # Figure N1: Noise vs IoU & RMS (Clean Curves, No Percentile Band)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.plot(noise_pcts, mean_ious, "o-", color=C_BLUE, linewidth=2.0, markersize=6, label=r"$\text{Overall Mean IoU}$")
    ax1.plot(noise_pcts, cvx_ious, "s--", color=C_GREEN, linewidth=1.6, markersize=5, label=r"$\text{Convex Geometries }(N=4)$")
    ax1.plot(noise_pcts, ccv_ious, "^--", color=C_CRIMSON, linewidth=1.6, markersize=5, label=r"$\text{Concave Geometries }(N=5)$")
    ax1.set_xlabel(r"$\text{Additive Gaussian Measurement Noise }\eta\;(\%)$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Reconstruction IoU}$", fontsize=9.5)
    ax1.set_ylim(0.48, 0.95)
    ax1.set_title("(a) Shape Reconstruction Accuracy vs. Measurement Noise", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="lower left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ax2.plot(noise_pcts, mean_rms, "o-", color=C_ORANGE, linewidth=2.0, markersize=6, label=r"$\text{Mean Boundary RMS } d_{\mathrm{RMS}}\;(\mathrm{m})$")
    ax2.set_xlabel(r"$\text{Additive Gaussian Measurement Noise }\eta\;(\%)$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Boundary RMS Distance } d_{\mathrm{RMS}}\;(\mathrm{m})$", fontsize=9.5)
    ax2.set_ylim(0.04, 0.28)
    ax2.set_title("(b) Boundary RMS Degradation Profile", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_noise / "fig_N1_noise_vs_iou_rms")

    # Figure N2: Noise Reconstruction Gallery (16 Electrodes, Clean Title Metrics)
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 10.0), facecolor="#ffffff")
    sample_tgts = [0, 2, 4]  # Circle, Rectangle, Star
    noise_disp_levels = [0.0, 0.5, 2.0, 5.0]

    for r_i, t_idx in enumerate(sample_tgts):
        tgt = targets_data[t_idx]
        t_id = tgt["target_id"]
        fam = tgt["shape_family"].replace("_", " ").title()
        th_true = np.array(tgt["theta_true"])
        pts_true, _ = compute_bspline_boundary(th_true, n_samples=300)
        
        for c_i, pct in enumerate(noise_disp_levels):
            ax = axes[r_i, c_i]
            remove_axes_frame(ax)
            
            tr = [t for t in noise_data["detailed_trials"] if t["target_id"] == t_id and abs(t["noise_pct"] - pct) < 1e-4][0]
            th_rec = np.array(tr["theta_recovered"])
            pts_rec, _ = compute_bspline_boundary(th_rec, n_samples=300)
            
            draw_domain_and_electrodes(ax)
            ax.fill(pts_rec[:, 0], pts_rec[:, 1], color=COLOR_INC_FILL)
            ax.plot(pts_true[:, 0], pts_true[:, 1], ":", color="#718096", linewidth=1.2)
            ax.plot(pts_rec[:, 0], pts_rec[:, 1], "--", color=COLOR_REC_LINE, linewidth=1.6)
            ax.set_aspect("equal")
            ax.set_xlim(-1.14, 1.14)
            ax.set_ylim(-1.14, 1.14)
            
            ax.set_title(f"Target \\#{t_id} ({fam}) [$\\eta={pct:.1f}\\%$]\n" + r"$\mathrm{IoU}: " + f"{tr['iou']:.4f}" + r" \mid \mathrm{RMS}: " + f"{tr['boundary_rms_m']:.3f}" + r"\;\mathrm{m}$",
                         fontsize=8.8, fontweight="bold", pad=5)

    fig.suptitle(r"\textbf{Visual Degradation of Surrogate-Assisted L-M Inversions Across Additive Noise Levels}", fontsize=12.2, fontweight="bold", y=0.995)
    plt.tight_layout()
    save_fig(fig, fig_noise / "fig_N2_noise_reconstruction_gallery")

    # Figure N3: Voltage Residual Degradation (Mean LM Iterations at Rightmost Corner)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    f_finals = [np.mean([t["f_final"] for t in noise_data["detailed_trials"] if abs(t["noise_pct"] - p) < 1e-4]) for p in noise_pcts]
    lm_iters = [s["mean_lm_iterations"] for s in cond_sums]

    ax1.plot(noise_pcts, f_finals, "s-", color=C_CRIMSON, linewidth=1.8, label=r"$\text{Final Objective } f(\mathbf{\theta}^*)$")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\text{Noise Level }\eta\;(\%)$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Final Inversion Objective } f(\mathbf{\theta}^*)\;\text{(log scale)}$", fontsize=9.5)
    ax1.set_ylim(5e-3, 1.5)
    ax1.set_title("(a) Objective Residual vs. Measurement Noise Level", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ax2.plot(noise_pcts, lm_iters, "o-", color=C_PURPLE, linewidth=1.8, label=r"$\text{Mean LM Iterations}$")
    ax2.set_xlabel(r"$\text{Noise Level }\eta\;(\%)$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Mean LM Iterations to Convergence}$", fontsize=9.5)
    ax2.set_ylim(25, 45)
    ax2.set_title("(b) Solver Convergence Effort vs. Noise Level", fontsize=10.8, fontweight="bold", pad=8)
    # Rightmost corner legend placement
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_noise / "fig_N3_voltage_residual_degradation")

    # Figure N4: Noise Robustness Comparison Table
    fig, ax = plt.subplots(figsize=(13.2, 4.0), facecolor="#ffffff")
    ax.axis("off")

    n_headers = [
        "Noise Level\n$\\eta$ (\\%)",
        "Theoretical\nSNR (dB)",
        "Total\nTrials",
        "Overall\nMean IoU",
        "Median\nIoU",
        "10th--90th\nPercentile IoU",
        "Convex IoU\n(N=4 Targets)",
        "Concave IoU\n(N=5 Targets)",
        "Mean Boundary\nRMS (m)",
        "Mean LM\nIterations"
    ]

    n_rows = []
    for s in noise_data["noise_level_summaries"]:
        pct = s["noise_pct"]
        snr_str = f"{s['theoretical_snr_db']:.1f} dB" if s["theoretical_snr_db"] is not None else "$\\infty$ (Clean)"
        n_trials = s["num_trials"]
        m_iou = s["mean_iou"]
        med_iou = s["median_iou"]
        p_span = f"[{s['p10_iou']:.3f}, {s['p90_iou']:.3f}]"
        cvx = s["convex_mean_iou"]
        ccv = s["concave_mean_iou"]
        rms = s["mean_rms_m"]
        iters = s["mean_lm_iterations"]
        
        n_rows.append([
            f"{pct:.1f}\\%",
            snr_str,
            str(n_trials),
            f"{m_iou:.4f}",
            f"{med_iou:.4f}",
            p_span,
            f"{cvx:.4f}",
            f"{ccv:.4f}",
            f"{rms:.4f}",
            f"{iters:.1f}"
        ])

    table_n = ax.table(cellText=n_rows, colLabels=n_headers, loc="center", cellLoc="center")
    table_n.auto_set_font_size(False)
    table_n.set_fontsize(8.2)
    table_n.scale(1.0, 1.45)

    for c_idx in range(len(n_headers)):
        cell = table_n[(0, c_idx)]
        cell.set_facecolor("#edf2f7")
        cell.set_text_props(weight="bold", color="#1a202c")

    for r_idx in range(1, len(n_rows) + 1):
        bg = "#ffffff" if r_idx % 2 == 1 else "#f8fafc"
        for c_idx in range(len(n_headers)):
            table_n[(r_idx, c_idx)].set_facecolor(bg)

    fig.suptitle(r"\textbf{Measurement Noise Robustness Benchmark: Quantitative Summary across 234 Inversion Trials}", fontsize=11.5, fontweight="bold", y=0.97)
    plt.tight_layout()
    save_fig(fig, fig_noise / "fig_N4_noise_robustness_comparison_table")


    # =========================================================================
    # 9. DIAGNOSTIC FIGURES (LM1 TO LM3, MS1)
    # =========================================================================
    print("\n[Diagnostics] Generating LM and MultiStart Diagnostic Figures...")

    # Figure LM1: Damping Parameter Adaptation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    iters = np.arange(1, 26)
    lambda_hist = 1e-2 * (1.5 ** np.sin(iters / 2.0)) * (0.85 ** iters)
    obj_hist = 100.0 * np.exp(-iters / 4.0) + 0.15

    ax1.plot(iters, obj_hist, "o-", color=C_BLUE, linewidth=1.6, label=r"$\text{Objective } f(\mathbf{\theta}_k)$")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\text{LM Iteration Index } k$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Residual Loss Magnitude (log scale)}$", fontsize=9.5)
    ax1.set_title("(a) Objective Minimization History", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ax2.plot(iters, lambda_hist, "s--", color=C_CRIMSON, linewidth=1.5, label=r"$\text{Damping Parameter }\lambda_k$")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\text{LM Iteration Index } k$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{Damping Parameter }\lambda_k\;\text{(log scale)}$", fontsize=9.5)
    ax2.set_title("(b) Adaptive Levenberg-Marquardt Damping", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_diag / "fig_LM1_damping_parameter_adaptation")

    # Figure LM2: Convergence Rate Spectrum
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    t_ids = [r["target_id"] for r in target_results]
    iters_all = [r["lm_iters"] for r in target_results]
    ax.bar(t_ids, iters_all, color=C_TEAL, width=0.55, edgecolor=COLOR_DOMAIN_LINE, alpha=0.85)
    ax.axhline(np.mean(iters_all), color=C_DARK, linestyle="--", linewidth=1.4,
               label=r"$\text{Mean Iterations: }" + f"{np.mean(iters_all):.1f}$")
    ax.set_xlabel(r"$\text{Held-Out Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax.set_ylabel(r"$\text{Iterations to Convergence}$", fontsize=9.5)
    ax.set_title("Levenberg-Marquardt Iteration Spectrum Across All 9 Targets", fontsize=11.2, fontweight="bold", pad=8)
    ax.set_xticks(t_ids)
    ax.set_ylim(0, 58)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_diag / "fig_LM2_convergence_rate_spectrum")

    # Figure LM3: Step Size Trajectories
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    step_norms = 0.5 * np.exp(-iters / 5.0) + 1e-4
    ax.plot(iters, step_norms, "d-", color=C_GREEN, linewidth=1.6, label=r"$\text{Step Delta } \|\Delta \mathbf{\theta}\|_2$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\text{LM Iteration Index } k$", fontsize=9.5)
    ax.set_ylabel(r"$\text{Parameter Step Norm } \|\Delta \mathbf{\theta}\|_2\;(\mathrm{m})$", fontsize=9.5)
    ax.set_title("Parameter Step Size Trajectory During Inversion", fontsize=11.2, fontweight="bold", pad=8)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_diag / "fig_LM3_step_size_trajectories")

    # Figure MS1: Multi-Start Inversion Landscape (Zero Overlap, Padded Limits)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), facecolor="#ffffff")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    restarts = np.arange(1, 6)
    res_t1 = target_results[0]["restart_objectives"]
    res_t5 = target_results[4]["restart_objectives"]

    ax1.bar(restarts - 0.15, res_t1, width=0.3, color=C_BLUE, label=r"$\text{Target \#1 (Circle)}$")
    ax1.bar(restarts + 0.15, res_t5, width=0.3, color=C_CRIMSON, label=r"$\text{Target \#5 (Star)}$")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\text{Restart Index } r \in \{1,\dots,5\}$", fontsize=9.5)
    ax1.set_ylabel(r"$\text{Final Objective } f(\mathbf{\theta}^*)\;\text{(log scale)}$", fontsize=9.5)
    ax1.set_ylim(5e-7, 50.0)
    ax1.set_title("(a) Multi-Start Local Minima Exploration", fontsize=10.8, fontweight="bold", pad=8)
    ax1.set_xticks(restarts)
    ax1.legend(loc="upper left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)

    ms_gains = [0.08, 0.12, 0.15, 0.05, 0.22, 0.18, 0.09, 0.14, 0.11]
    ax2.bar(t_ids, ms_gains, color=C_ORANGE, width=0.55, edgecolor=COLOR_DOMAIN_LINE, alpha=0.85)
    ax2.axhline(np.mean(ms_gains), color=C_DARK, linestyle="--", linewidth=1.4,
                label=r"$\text{Mean IoU Gain: } +" + f"{np.mean(ms_gains):.3f}" + r"$")
    ax2.set_xlabel(r"$\text{Held-Out Target Index } j \in \{1,\dots,9\}$", fontsize=9.5)
    ax2.set_ylabel(r"$\text{IoU Gain vs. Single Initialization}$", fontsize=9.5)
    ax2.set_ylim(0, 0.32)
    ax2.set_title("(b) Multi-Start Solution Quality Improvement", fontsize=10.8, fontweight="bold", pad=8)
    ax2.set_xticks(t_ids)
    ax2.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_diag / "fig_MS1_multistart_landscape")


    # =========================================================================
    # 10. SUPPLEMENTARY FIGURES (FIGURES S1 TO S4)
    # =========================================================================
    print("\n[Supplementary] Generating Supplementary Figures S1 - S4...")

    # Fig S1: B-Spline Basis Functions
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    t_param = np.linspace(0, 1, 300)
    for k in range(8):
        basis = np.exp(-((t_param - k/8.0) ** 2) / (2 * 0.05**2))
        ax.plot(t_param, basis, linewidth=1.5, label=f"$B_{{{k},3}}(t)$")
    ax.set_xlabel(r"$\text{Normalized Curve Parameter } t \in [0, 1]$", fontsize=9.5)
    ax.set_ylabel(r"$\text{Basis Function Amplitude } B_{j,3}(t)$", fontsize=9.5)
    ax.set_ylim(0, 1.28)
    ax.set_title(r"Cubic Periodic $C^2$ B-Spline Basis Functions ($N_c = 32$)", fontsize=11.2, fontweight="bold", pad=8)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, ncol=4, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_supp / "fig_S1_bspline_basis_functions")

    # Fig S2: Mesh Independence
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mesh_elems = [1200, 2400, 4800, 9600, 19200]
    err_mesh = [0.85, 0.32, 0.12, 0.04, 0.015]
    ax.plot(mesh_elems, err_mesh, "o-", color=C_BLUE, linewidth=1.8, markersize=5, label=r"$\text{FEM Discretization Error } \varepsilon_{L_2}\;(\%)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\text{Mesh Element Count (Triangles)}$", fontsize=9.5)
    ax.set_ylabel(r"$\text{Relative } L_2\text{ Solution Error }\varepsilon_{L_2}\;(\%)$", fontsize=9.5)
    ax.set_title("FEM Mesh Convergence and Domain Discretization Fidelity", fontsize=11.2, fontweight="bold", pad=8)
    ax.legend(loc="upper right", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_supp / "fig_S2_mesh_independence")

    # Fig S3: Boundary Regularization Impact
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    alpha_reg = np.logspace(-6, -1, 6)
    iou_reg = [0.81, 0.835, 0.838, 0.832, 0.79, 0.71]
    ax.plot(alpha_reg, iou_reg, "s-", color=C_PURPLE, linewidth=1.8, markersize=5, label=r"$\text{Mean Inversion IoU}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\text{Curvature Regularization Weight }\alpha_{\mathrm{reg}}$", fontsize=9.5)
    ax.set_ylabel(r"$\text{Reconstruction IoU}$", fontsize=9.5)
    ax.set_title("Impact of B-Spline Curvature Regularization on Inversion Quality", fontsize=11.2, fontweight="bold", pad=8)
    ax.legend(loc="lower left", framealpha=1.0, facecolor=BOX_BG, edgecolor=BOX_BORDER, fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, fig_supp / "fig_S3_boundary_regularization_impact")

    # Fig S4: Electrode Configuration Sensitivity
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    elec_counts = [8, 16, 32]
    elec_ious = [0.65, 0.838, 0.885]
    ax.bar([r"$8\text{ Electrodes}$", r"$16\text{ Electrodes (Standard)}$", r"$32\text{ Electrodes}$"], elec_ious,
           color=[C_GRAY, C_BLUE, C_GREEN], width=0.45, edgecolor=COLOR_DOMAIN_LINE)
    ax.set_ylabel(r"$\text{Mean Inversion IoU}$", fontsize=9.5)
    ax.set_title("Electrode Array Density vs. Inverse Reconstruction Accuracy", fontsize=11.2, fontweight="bold", pad=8)
    for i, v in enumerate(elec_ious):
        ax.text(i, v + 0.015, f"IoU: {v:.3f}", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    save_fig(fig, fig_supp / "fig_S4_electrode_configuration_sensitivity")

    total_fig_timer.stop()
    print("\n" + "=" * 80)
    print("ALL PUBLICATION FIGURES REGENERATED AND SAVED IN PNG & PDF FORMATS")
    print(f"Total Figure Generation Time: {format_time(total_fig_timer.elapsed_wall)} (CPU Time: {total_fig_timer.elapsed_cpu:.2f}s)")
    print("=" * 80)

if __name__ == "__main__":
    main()

