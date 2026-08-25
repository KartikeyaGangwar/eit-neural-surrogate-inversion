"""
================================================================================
EIT Computational Domain Setup Visualizer (Monochrome, Arrow-Free Q1 Standard)
================================================================================
Generates Figure 1:
  - Clean monochrome/greyscale 2D circular domain (R = 1.0 m).
  - Absolutely NO arrows anywhere on the figure.
  - Absolutely NO colourful patches or bright colored badges.
  - Generous label margins for e_1(I+), e_9(I-), e_5(GND).
  - Mathematical symbols (Omega_0, Omega_inc, Gamma, p_i) placed in-situ with clear contrast.
  - High-clarity large-font Legend & Specifications panel at bottom.
  - Computer Modern LaTeX typography, vector PDF & 300+ DPI PNG.
================================================================================
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyBboxPatch

# Matplotlib styling for clean Computer Modern rendering
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent if script_dir.name == "scripts" else script_dir
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import config as cfg
from src.geometry.bspline import compute_bspline_boundary


def plot_computational_domain(save_path_no_ext: str = None) -> None:
    if save_path_no_ext is None:
        save_path_no_ext = str(repo_root / "figures" / "main" / "fig1_computational_domain_and_setup")

    # Figure Canvas: 9.0 x 10.5 inches
    fig, ax = plt.subplots(figsize=(9.0, 10.5), facecolor="#ffffff", dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    # Domain shifted vertically: center at y = 0.38
    cx, cy = 0.0, 0.38
    R = 1.0

    # Canvas bounds with generous breathing room
    ax.set_xlim(-1.56, 1.56)
    ax.set_ylim(-1.46, 1.68)

    # 1. Background Circular Domain (Omega_0) - Crisp monochrome outline & subtle fill
    domain_circle = Circle((cx, cy), R, facecolor="#f8fafc", edgecolor="#0f172a", linewidth=2.2, zorder=2)
    ax.add_patch(domain_circle)

    # Coordinate crosshairs inside circle
    ax.plot([cx - 0.95, cx + 0.95], [cy, cy], color="#e2e8f0", linestyle=":", linewidth=1.1, zorder=1)
    ax.plot([cx, cx], [cy - 0.95, cy + 0.95], color="#e2e8f0", linestyle=":", linewidth=1.1, zorder=1)

    # 2. Inclusion Geometry (Star Anomaly, Target #5)
    targets_path = repo_root / "data" / "targets" / "held_out_9_targets.json"
    with open(targets_path) as f:
        targets_data = json.load(f)

    th_star = np.array(targets_data[4]["theta_true"])
    pts_inc, _ = compute_bspline_boundary(th_star, n_samples=400)
    ctrl_x = th_star[:32] + cx
    ctrl_y = th_star[32:] + cy
    ctrl_poly_x = np.append(ctrl_x, ctrl_x[0])
    ctrl_poly_y = np.append(ctrl_y, ctrl_y[0])

    # Plot Control Polygon (dashed slate line) & Control Points (charcoal dots, NO colors)
    ax.plot(ctrl_poly_x, ctrl_poly_y, "--", color="#64748b", linewidth=1.1, alpha=0.85, zorder=3)
    ax.scatter(ctrl_x, ctrl_y, s=26, color="#334155", edgecolor="#ffffff", linewidth=0.8, zorder=5)

    # Inclusion Shape (Dark Slate fill + crisp white interface boundary)
    ax.fill(pts_inc[:, 0] + cx, pts_inc[:, 1] + cy, color="#1e293b", alpha=0.96, zorder=4)
    ax.plot(pts_inc[:, 0] + cx, pts_inc[:, 1] + cy, "-", color="#ffffff", linewidth=1.6, zorder=4)

    # 3. Minimal Mathematical Symbols (Direct in-situ placement, NO arrows, high contrast on background)
    # Omega_0 (Background)
    ax.text(cx, cy + 0.68, r"$\mathbf{\Omega_0}$", ha="center", va="center",
            fontsize=15.5, fontweight="bold", color="#0f172a", zorder=6)

    # Omega_inc (Inclusion)
    ax.text(cx, cy, r"$\mathbf{\Omega_{\mathrm{inc}}}$", ha="center", va="center",
            fontsize=13.0, fontweight="bold", color="#ffffff", zorder=6)

    # Gamma (Interface) placed right near interface boundary on background
    ax.text(cx + 0.38, cy + 0.42, r"$\mathbf{\Gamma}$", ha="center", va="center",
            fontsize=14.0, fontweight="bold", color="#0f172a", zorder=6)

    # p_i (Control Points) placed cleanly on background right next to the control point
    ax.text(cx + 0.54, cy + 0.18, r"$\mathbf{p}_i$", ha="left", va="center",
            fontsize=13.5, fontweight="bold", color="#0f172a", zorder=6)

    # 4. 16 Complete Electrode Model Surface Electrodes (Uniform Monochrome)
    n_electrodes = 16
    electrode_width_deg = 8.5
    electrode_thick = 0.055
    d_th = 2 * np.pi / n_electrodes
    half_w = np.radians(electrode_width_deg) / 2.0

    for e_i in range(n_electrodes):
        th = e_i * d_th
        th_span = np.linspace(th - half_w, th + half_w, 16)
        r_in = R - electrode_thick / 2.0
        r_out = R + electrode_thick / 2.0

        x_outer = cx + r_out * np.cos(th_span)
        y_outer = cy + r_out * np.sin(th_span)
        x_inner = cx + r_in * np.cos(th_span[::-1])
        y_inner = cy + r_in * np.sin(th_span[::-1])

        x_p = np.concatenate([x_outer, x_inner])
        y_p = np.concatenate([y_outer, y_inner])

        poly = Polygon(np.column_stack([x_p, y_p]), closed=True,
                       facecolor="#334155", edgecolor="#0f172a", linewidth=1.1, zorder=7)
        ax.add_patch(poly)

        # Generous offset for wide text labels (e_1, e_9, e_5)
        if e_i == 0:  # e_1 (I+) at 0 deg (right)
            r_label = R + 0.20
            lbl = r"$e_1\,(I^+)$"
        elif e_i == 8:  # e_9 (I-) at 180 deg (left)
            r_label = R + 0.20
            lbl = r"$e_9\,(I^-)$"
        elif e_i == 4:  # e_5 (GND) at 90 deg (top)
            r_label = R + 0.15
            lbl = r"$e_5\,(\mathrm{GND})$"
        else:
            r_label = R + 0.11
            lbl = f"$e_{{{e_i+1}}}$"

        lx = cx + r_label * np.cos(th)
        ly = cy + r_label * np.sin(th)

        ax.text(lx, ly, lbl, ha="center", va="center",
                fontsize=10.0, fontweight="bold", color="#0f172a", zorder=8)

    # 5. Bottom Master Legend & Specifications Panel (Enlarged Fonts for High Clarity)
    leg_box_y = -1.40
    leg_w = 2.92
    leg_h = 0.52

    box_rect = FancyBboxPatch((-1.46, leg_box_y), leg_w, leg_h,
                              boxstyle="round,pad=0.04,rounding_size=0.03",
                              facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2, zorder=10)
    ax.add_patch(box_rect)

    # Left Column: Domain & Physical Nomenclature
    col1_title = "Domain & Nomenclature:"
    col1_body = (
        r"$\bullet\ \mathbf{\Omega_0}:\ \mathrm{Background\ Domain\ }(R = 1.0\,\mathrm{m},\;\sigma_0 = 1.0\,\mathrm{S/m})$" "\n"
        r"$\bullet\ \mathbf{\Omega_{\mathrm{inc}}}:\ \mathrm{Anomaly\ Inclusion\ }(\sigma_{\mathrm{inc}} = 10^{-4}\,\mathrm{S/m})$" "\n"
        r"$\bullet\ \mathbf{\Gamma} = \partial\Omega_{\mathrm{inc}}:\ \mathrm{B\text{-}Spline\ Interface\ Boundary}$" "\n"
        r"$\bullet\ \mathbf{p}_i\ (i = 1,\dots,32):\ \mathrm{Control\ Polygon\ Points\ }(\theta\in\mathbb{R}^{64})$"
    )
    ax.text(-1.40, leg_box_y + 0.45, col1_title, fontsize=10.6, fontweight="bold", va="top", ha="left", color="#0f172a", zorder=11)
    ax.text(-1.40, leg_box_y + 0.33, col1_body, fontsize=9.6, va="top", ha="left", linespacing=1.45, color="#1e293b", zorder=11)

    # Right Column: Electrodes & Measurement Setup
    col2_title = "Electrodes & Stimulation Setup:"
    col2_body = (
        r"$\bullet\ e_1,\dots,e_{16}:\ 16\mathrm{\ Surface\ CEM\ Electrodes\ }(z_l = 0.01\,\Omega\cdot\mathrm{m})$" "\n"
        r"$\bullet\ I^+ / I^-:\ \mathrm{Active\ Current\ Source\ /\ Sink\ }(\pm 1.0\,\mathrm{A})$" "\n"
        r"$\bullet\ \mathrm{GND}:\ \mathrm{Reference\ Ground\ Electrode\ }(0\,\mathrm{V})$" "\n"
        r"$\bullet\ \mathrm{Protocol}:\ 120\mathrm{\ Pairwise\ Stims\ }\times 16 = 1920\mathrm{\ Voltages}$"
    )
    ax.text(0.04, leg_box_y + 0.45, col2_title, fontsize=10.6, fontweight="bold", va="top", ha="left", color="#0f172a", zorder=11)
    ax.text(0.04, leg_box_y + 0.33, col2_body, fontsize=9.6, va="top", ha="left", linespacing=1.45, color="#1e293b", zorder=11)

    # Save Outputs
    p_out = Path(save_path_no_ext)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    png_path = p_out.with_suffix(".png")
    pdf_path = p_out.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"[Figure 1 Generated] {png_path.name} & {pdf_path.name}")


def main():
    out_base = repo_root / "figures" / "main" / "fig1_computational_domain_and_setup"
    plot_computational_domain(str(out_base))


if __name__ == "__main__":
    main()
