"""
Plot Computational Domain Schematic (Fig. 1)
==============================================
Generates publication-quality schematic of the 2D EIT computational setup:
  - 2D Circular domain Omega (R0 = 1.0 m, sigma_bg = 1.0 S/m)
  - 16 Boundary Electrodes (Complete Electrode Model, w = 0.20 rad, z_l = 0.01 Ohm*m^2)
  - Pairwise Current Excitation (+I0 at E1, -I0 at E2)
  - L-shaped Reference Ground (GND at E9)
  - Outward unit normal vector nu on insulating boundary
  - Closed cubic B-spline inclusion Omega_inc (sigma_inc = 10^-4 S/m) with boundary Gamma_theta and control points P_m
  - Clean layout with zero label collisions and minimal uncluttered typography.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Wedge
from scipy.interpolate import splprep, splev

# Configure matplotlib for publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'mathtext.fontset': 'cm',
    'figure.autolayout': True,
})

def create_computational_domain_schematic(output_png: str, output_pdf: str) -> None:
    """Generate the domain schematic figure and save to PNG and PDF."""
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=300)
    
    # 1. Circular Domain Constants
    R0 = 1.0
    N_elec = 16
    elec_w = 0.20  # angular width in radians (~11.46 deg)
    
    # Draw Background circular domain Omega
    tank_circle = Circle((0, 0), R0, facecolor='#F5F8FC', edgecolor='#1D3557', linewidth=2.2, zorder=1)
    ax.add_patch(tank_circle)
    
    # Faint orthogonal center grid lines
    ax.plot([-0.95, 0.95], [0, 0], color='#DCE4EE', linestyle=':', linewidth=1.0, zorder=2)
    ax.plot([0, 0], [-0.95, 0.95], color='#DCE4EE', linestyle=':', linewidth=1.0, zorder=2)
    
    # 2. Draw B-spline Inclusion Curve and Control Polygon
    phi_cp = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    # Smooth anomaly radii (strictly within [0.20, 0.45])
    r_cp = np.array([0.36, 0.30, 0.40, 0.33, 0.26, 0.35, 0.38, 0.29, 0.24, 0.33, 0.36, 0.30])
    x_offset, y_offset = 0.02, -0.02
    x_cp = x_offset + r_cp * np.cos(phi_cp)
    y_cp = y_offset + r_cp * np.sin(phi_cp)
    
    # Smooth closed curve interpolation
    pts_cp = np.vstack((x_cp, y_cp))
    pts_cp = np.hstack((pts_cp, pts_cp[:, :1]))  # close loop
    tck, u = splprep(pts_cp, s=0, per=True, k=3)
    unew = np.linspace(0, 1, 400)
    out = splev(unew, tck)
    curve_x, curve_y = out[0], out[1]
    
    # Fill inclusion anomaly
    inclusion_poly = Polygon(np.column_stack([curve_x, curve_y]), 
                             facecolor='#FFE8D6', edgecolor='#E05A1B', 
                             linewidth=2.4, linestyle='-', zorder=4)
    ax.add_patch(inclusion_poly)
    
    # Draw Control Polygon (subtle dashed)
    ax.plot(np.append(x_cp, x_cp[0]), np.append(y_cp, y_cp[0]), 
            linestyle='--', color='#8D99AE', linewidth=1.2, zorder=5)
    
    # Draw Control Points
    ax.scatter(x_cp, y_cp, s=36, color='#2B2D42', edgecolor='white', linewidth=1.2, zorder=6)
    
    # 3. Draw 16 Electrodes along boundary
    angles = np.linspace(0, 2 * np.pi, N_elec, endpoint=False)
    elec_thickness = 0.04
    
    # Electrode colors
    # E1 (+I0) in Crimson, E2 (-I0) in Navy, E9 (GND) in Teal/Slate, others in Steel Blue
    for i, theta in enumerate(angles):
        deg_start = np.rad2deg(theta - elec_w / 2.0)
        deg_end = np.rad2deg(theta + elec_w / 2.0)
        
        if i == 0:
            e_color = '#E63946'  # +I0
            edge_c = '#9D0208'
        elif i == 1:
            e_color = '#1D3557'  # -I0
            edge_c = '#03045E'
        elif i == 8:  # E9 is Ground
            e_color = '#2A9D8F'  # GND
            edge_c = '#1B4332'
        else:
            e_color = '#457B9D'  # Passive
            edge_c = '#1D3557'
            
        wedge = Wedge(
            (0, 0), R0 + elec_thickness, deg_start, deg_end, 
            width=elec_thickness, facecolor=e_color, edgecolor=edge_c, linewidth=1.2, zorder=7
        )
        ax.add_patch(wedge)
        
        # Electrode labels (E_1 to E_16) positioned cleanly radially outside
        r_lbl = 1.15
        x_lbl = r_lbl * np.cos(theta)
        y_lbl = r_lbl * np.sin(theta)
        
        # For E9 (ground), offset label slightly above horizontal wire for zero overlap
        if i == 8:
            x_lbl = -1.14
            y_lbl = 0.09
            lbl_color = '#1B4332'
            font_wt = 'bold'
        elif i == 0:
            lbl_color = '#E63946'
            font_wt = 'bold'
        elif i == 1:
            lbl_color = '#1D3557'
            font_wt = 'bold'
        else:
            lbl_color = '#2B2D42'
            font_wt = 'normal'
        
        ax.text(x_lbl, y_lbl, f"$E_{{{i+1}}}$", fontsize=11, weight=font_wt,
                color=lbl_color, ha='center', va='center', zorder=8)
        
    # 4. Clean Injection Current Arrows at E1 and E2 (Positioned with ZERO overlap)
    # +I0 arrow pointing towards E1
    arr_in = FancyArrowPatch(
        (1.58, 0.0), (1.27, 0.0),
        arrowstyle='simple,head_width=6,head_length=7',
        color='#E63946', linewidth=1.2, zorder=9
    )
    ax.add_patch(arr_in)
    ax.text(1.64, 0.0, r"$+I_0$", fontsize=13, weight='bold', color='#E63946',
            ha='left', va='center', zorder=9)
    
    # -I0 arrow pointing away from E2
    th1 = angles[1]
    arr_out = FancyArrowPatch(
        (1.27 * np.cos(th1), 1.27 * np.sin(th1)),
        (1.58 * np.cos(th1), 1.58 * np.sin(th1)),
        arrowstyle='simple,head_width=6,head_length=7',
        color='#1D3557', linewidth=1.2, zorder=9
    )
    ax.add_patch(arr_out)
    ax.text(1.64 * np.cos(th1), 1.64 * np.sin(th1), r"$-I_0$", 
            fontsize=13, weight='bold', color='#1D3557', ha='left', va='center', zorder=9)

    # 5. Clean L-SHAPED Ground (GND) Connection at E9 (angle 180 deg)
    # Horizontal segment of wire from E9 going left
    gx0 = -(R0 + elec_thickness)  # -1.04
    gx_corner = -1.24
    gy_corner = 0.0
    gy_end = -0.22
    
    # Draw L-shaped wire (horizontal then vertically down)
    ax.plot([gx0, gx_corner], [gy_corner, gy_corner], color='#2A9D8F', linewidth=1.6, zorder=8)
    ax.plot([gx_corner, gx_corner], [gy_corner, gy_end], color='#2A9D8F', linewidth=1.6, zorder=8)
    
    # Ground standard 3 horizontal bars at the bottom of the vertical wire
    # 1st (top/widest bar)
    ax.plot([gx_corner - 0.065, gx_corner + 0.065], [gy_end, gy_end], color='#2A9D8F', linewidth=2.0, zorder=8)
    # 2nd (middle bar)
    ax.plot([gx_corner - 0.042, gx_corner + 0.042], [gy_end - 0.03, gy_end - 0.03], color='#2A9D8F', linewidth=1.8, zorder=8)
    # 3rd (bottom/smallest bar)
    ax.plot([gx_corner - 0.020, gx_corner + 0.020], [gy_end - 0.06, gy_end - 0.06], color='#2A9D8F', linewidth=1.5, zorder=8)
    
    # GND label placed cleanly to the left of the ground symbol
    ax.text(gx_corner - 0.09, gy_end - 0.03, r"$\mathrm{GND}$", fontsize=11.5, weight='bold', 
            color='#1B4332', ha='right', va='center', zorder=9)

    # 6. Outward Unit Normal vector nu at insulated boundary (at angle 101.25 deg between E5 and E6)
    th_nu = np.deg2rad(101.25)
    p_nu = np.array([R0 * np.cos(th_nu), R0 * np.sin(th_nu)])
    arr_nu = FancyArrowPatch(
        p_nu, p_nu + 0.18 * np.array([np.cos(th_nu), np.sin(th_nu)]),
        arrowstyle='simple,head_width=5.5,head_length=6.5',
        color='#1D3557', linewidth=1.1, zorder=9
    )
    ax.add_patch(arr_nu)
    ax.text(p_nu[0] + 0.25 * np.cos(th_nu), p_nu[1] + 0.25 * np.sin(th_nu), 
            r"$\boldsymbol{\nu}$", fontsize=13.5, color='#1D3557', weight='bold', ha='center', va='center')

    # 7. Uncluttered Domain & Anomaly Symbols (pure math symbols, zero clutter)
    # Background domain symbol
    ax.text(-0.55, 0.58, r"$\Omega$", fontsize=16, color='#1D3557', weight='bold', ha='center', va='center', zorder=5)
    ax.text(-0.55, 0.47, r"$(\sigma_{\mathrm{bg}})$", fontsize=11.5, color='#457B9D', ha='center', va='center', zorder=5)
    
    # Inclusion domain symbol
    ax.text(x_offset, y_offset + 0.04, r"$\Omega_{\mathrm{inc}}$", fontsize=14, color='#B73200', weight='bold', ha='center', va='center', zorder=6)
    ax.text(x_offset, y_offset - 0.06, r"$(\sigma_{\mathrm{inc}})$", fontsize=11.5, color='#B73200', ha='center', va='center', zorder=6)
    
    # Boundary curve label Gamma_theta (clean arrow pointing to bottom-left curve)
    ax.annotate(r"$\Gamma_{\boldsymbol{\theta}}$", 
                xy=(curve_x[260], curve_y[260]), 
                xytext=(curve_x[260] - 0.20, curve_y[260] - 0.16),
                fontsize=13.5, weight='bold', color='#E05A1B',
                arrowprops=dict(arrowstyle="->", color='#E05A1B', lw=1.3),
                zorder=7)
    
    # Control point label P_m (clean arrow pointing to top-right control point)
    idx_cp = 2
    ax.annotate(r"$\mathbf{P}_m$", 
                xy=(x_cp[idx_cp], y_cp[idx_cp]), 
                xytext=(x_cp[idx_cp] + 0.16, y_cp[idx_cp] + 0.12),
                fontsize=13.5, weight='bold', color='#2B2D42',
                arrowprops=dict(arrowstyle="->", color='#2B2D42', lw=1.3),
                zorder=7)

    # Axis limits & aspect ratio (No legend at bottom, symmetrical margins)
    ax.set_xlim(-1.65, 1.85)
    ax.set_ylim(-1.35, 1.40)
    ax.set_aspect('equal')
    ax.axis('off')
    
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Schematic generated successfully:\n  PNG: {output_png}\n  PDF: {output_pdf}")

if __name__ == "__main__":
    # Resolve project root directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir) if os.path.basename(script_dir) in ["visuals", "scripts"] else script_dir
    parent_dir = os.path.dirname(repo_root)
    paper_dir = os.path.join(parent_dir, "paper")
    
    # Output targets
    targets = [
        (os.path.join(repo_root, "figures", "schematic", "fig1_computational_domain_schematic.png"),
         os.path.join(repo_root, "figures", "schematic", "fig1_computational_domain_schematic.pdf")),
    ]
    if os.path.isdir(paper_dir):
        targets.append((
            os.path.join(paper_dir, "figures", "main", "fig1_computational_domain_schematic.png"),
            os.path.join(paper_dir, "figures", "main", "fig1_computational_domain_schematic.pdf")
        ))
        targets.append((
            os.path.join(paper_dir, "figures", "schematic", "fig1_computational_domain_schematic.png"),
            os.path.join(paper_dir, "figures", "schematic", "fig1_computational_domain_schematic.pdf")
        ))

    for png_p, pdf_p in targets:
        create_computational_domain_schematic(png_p, pdf_p)
