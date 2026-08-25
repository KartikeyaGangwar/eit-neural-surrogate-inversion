# Visuals & Domain Schematic Generator

This directory contains standalone high-resolution visualization scripts for the 2D Electrical Impedance Tomography (EIT) computational setup.

---

## 1. Scripts

- `plot_computational_domain_schematic.py`: Generates Figure 1 of the manuscript, visualizing:
  - 2D Circular computational domain $\Omega$ with tank radius $R_0 = 1.0\,\mathrm{m}$ and background conductivity $\sigma_{\mathrm{bg}} = 1.0\,\mathrm{S/m}$.
  - 16 equidistant surface electrodes $E_1, \dots, E_{16}$ governed by the Complete Electrode Model (CEM) with contact impedance $z_l = 0.01\,\Omega\cdot\mathrm{m}^2$ and electrode width $w = 0.20\,\mathrm{rad}$.
  - Pairwise current excitation ($+I_0$ at $E_1$, $-I_0$ at $E_2$) with current amplitude $I_0 = 1.0\,\mathrm{A}$.
  - L-shaped reference ground (GND) at electrode $E_9$.
  - Outward unit normal vector $\boldsymbol{\nu}$ along the insulating boundary $\partial\Omega \setminus \bigcup_{l=1}^{16} e_l$.
  - Closed cubic B-spline anomaly $\Omega_{\mathrm{inc}}$ with conductivity $\sigma_{\mathrm{inc}} = 10^{-4}\,\mathrm{S/m}$, control polygon vertices $\mathbf{P}_m$, and parameterized boundary $\Gamma_{\boldsymbol{\theta}}$.

---

## 2. Usage

```bash
# Generate high-resolution PNG (300 DPI) and vector PDF
python visuals/plot_computational_domain_schematic.py
```
Output files are saved to `figures/schematic/fig1_computational_domain_schematic.png` and `figures/schematic/fig1_computational_domain_schematic.pdf`.
