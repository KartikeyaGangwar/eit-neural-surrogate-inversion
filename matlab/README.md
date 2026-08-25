# MATLAB / GNU Octave Physics Reference & Validation Suite

This directory contains the standalone MATLAB / GNU Octave reference implementations and validation scripts for the **64-D Periodic Cubic B-Spline Electrical Impedance Tomography (EIT)** forward model and geometric sensitivity solver.

---

## 1. Prerequisites

To execute these MATLAB/Octave benchmark scripts:
1. **GNU Octave** (v7.0+ recommended) or **MATLAB** (R2020b+)
2. **EIDORS** (v3.11 or v3.12-ng): [http://eidors3d.sourceforge.net/](http://eidors3d.sourceforge.net/)
   - Ensure `run('path/to/eidors/startup.m')` has been executed before running EIDORS-dependent tests.
3. **Netgen Mesh Generator** (included with standard EIDORS distributions).

---

## 2. File Inventory & Scientific Purpose

| File | Purpose | EIDORS Required? |
| :--- | :--- | :---: |
| [`polygon_conductivity_bspline.m`](polygon_conductivity_bspline.m) | **Core Reference Solver:** Computes element conductivities $\boldsymbol{\Sigma}(\boldsymbol{\theta})$ and analytical sensitivity $\frac{\partial \boldsymbol{\Sigma}}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{N_{\mathrm{elem}} \times 64}$ for a 64-D periodic cubic B-spline using mollified boundary integral winding. | No |
| [`test_bspline_conductivity_derivative.m`](test_bspline_conductivity_derivative.m) | **Stage 1 Unit Test:** Validates analytic derivatives of the mollified winding indicator against central finite differences. | No |
| [`test_polygon_conductivity_bspline.m`](test_polygon_conductivity_bspline.m) | **Stage 2 Validation:** Tests the complete element conductivity Jacobian matrix against finite differences across multiple test geometries. | No |
| [`test_stage3_eidors_conductivity_jacobian.m`](test_stage3_eidors_conductivity_jacobian.m) | **Stage 3 Validation:** Verifies EIDORS Complete Electrode Model (CEM) forward voltage solver and conductivity sensitivity $\mathbf{J}_{\boldsymbol{\Sigma}} = \frac{\partial \mathbf{V}}{\partial \boldsymbol{\Sigma}} \in \mathbb{R}^{1920 \times N_{\mathrm{elem}}}$ with directional perturbation solves. | **Yes** |
| [`test_stage4_mesh_convergence.m`](test_stage4_mesh_convergence.m) | **Stage 4 Mesh Convergence:** Quantifies forward voltage convergence under mesh refinement (varying `mesh_maxsz`). | **Yes** |
| [`test_stage4_mesh_convergence_directional.m`](test_stage4_mesh_convergence_directional.m) | **Stage 4 Sensitivity Convergence:** Validates mesh convergence of the complete chain rule sensitivity $\mathbf{J}_{\boldsymbol{\theta}} = \mathbf{J}_{\boldsymbol{\Sigma}} \frac{\partial \boldsymbol{\Sigma}}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{1920 \times 64}$ along random directional projections. | **Yes** |

---

## 3. How to Run the Validation Suite

### In GNU Octave (or MATLAB CLI)
```matlab
% 1. Start Octave and initialize EIDORS
run('C:/path/to/eidors-v3.12-ng/eidors/startup.m');

% 2. Run Stage 1 & 2 Conductivity Jacobian tests (No EIDORS required)
test_bspline_conductivity_derivative
test_polygon_conductivity_bspline

% 3. Run Stage 3 EIDORS Sensitivity tests
test_stage3_eidors_conductivity_jacobian

% 4. Run Stage 4 Mesh Convergence Studies
test_stage4_mesh_convergence
test_stage4_mesh_convergence_directional
```

---

## 4. Python Equivalence

The pure Python modules in `src/geometry/` and `src/physics/` match these MATLAB reference routines to floating-point machine precision ($< 10^{-12}$ relative discrepancy):
- `polygon_conductivity_bspline.m` $\Longleftrightarrow$ `src/geometry/conductivity.py` (`compute_sigma_and_derivative`)
- `test_stage3_eidors_conductivity_jacobian.m` $\Longleftrightarrow$ `src/validation/val_B_fem_jsigma.py`
- `test_stage4_mesh_convergence_directional.m` $\Longleftrightarrow$ `src/validation/val_C_geometry_jacobian.py`
