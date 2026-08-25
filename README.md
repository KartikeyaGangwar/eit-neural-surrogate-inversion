# Zero-Online-FEM Electrical Impedance Tomography via Sobolev-Regularized B-Spline Surrogates

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passing-brightgreen.svg)](tests/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-green.svg)](REPRODUCIBILITY_MAP.md)

This repository contains the official reference implementation, pretrained neural surrogate models, and reproducibility suite for the research paper:

> **"Zero-Online-FEM Inverse Electrical Impedance Tomography via Directional Sobolev-Regularized B-Spline Surrogates"**

---

## Table of Contents
- [1. Scientific Overview & Highlights](#1-scientific-overview--highlights)
- [2. Mathematical Formulations](#2-mathematical-formulations)
  - [2.1 B-Spline Geometry & Mollified Indicator](#21-b-spline-geometry--mollified-indicator)
  - [2.2 Complete Electrode Model (CEM) Forward Problem](#22-complete-electrode-model-cem-forward-problem)
  - [2.3 Directional Sobolev-Supervised Surrogate](#23-directional-sobolev-supervised-surrogate)
  - [2.4 Zero-Online-FEM Levenberg-Marquardt Inversion](#24-zero-online-fem-levenberg-marquardt-inversion)
  - [2.5 Deep Ensemble Epistemic Uncertainty Quantification](#25-deep-ensemble-epistemic-uncertainty-quantification)
- [3. Repository Architecture](#3-repository-architecture)
- [4. Installation & Environment Setup](#4-installation--environment-setup)
- [5. Master CLI & Reproducing Results](#5-master-cli--reproducing-results)
- [6. Experimental Results & Benchmarks](#6-experimental-results--benchmarks)
- [7. Automated Verification & Testing](#7-automated-verification--testing)
- [8. Citation & License](#8-citation--license)

---

## 1. Scientific Overview & Highlights

- **Zero-Online-FEM Inversion ($N_{\mathrm{FEM}}^{\mathrm{online}} = 0$):** Completely eliminates the need for expensive online Finite Element Method (FEM) forward solves and adjoint PDE sensitivity computations during iterative Levenberg--Marquardt reconstruction, reducing per-target inversion time from **$\approx 505.0\,\mathrm{s}$** to **$\approx 8.89\,\mathrm{s}$** (a **$56.8\times$ computational speedup**).
- **Directional Sobolev JVP Regularization:** Trains deep ResMLP surrogates with directional Jacobian-Vector Product (JVP) supervision ($\lambda_J = 0.01\,\mathrm{m}^2$), achieving analytical autodiff sensitivity alignment with a **$0.9982$ cosine gradient alignment** against true FEM adjoint sensitivities.
- **High Forward Accuracy:** Achieves **$0.2452\%$** mean relative voltage error and **$1.2461\,\mathrm{mV}$** RMSE across 1,000 held-out test samples ($R^2 > 0.9999$).
- **Deep Ensemble Inversion Gain:** A 5-member deep ensemble elevates reconstruction mean IoU to **$0.8989$** (**$0.9630$** convex, **$0.8476$** concave), achieving a **$50.34\%$** boundary RMS distance error reduction ($32.0\,\mathrm{mm}$ consensus RMS).
- **Robust Noise Tolerance:** Demonstrates high numerical stability ($100\%$ completion rate, $99.57\%$ reconstruction success $\mathrm{IoU} \ge 0.50$) across a 234-trial benchmark spanning 6 noise levels ($\eta \in [0.0\%, 5.0\%]$ / $\mathrm{SNR} \in [\infty, 26\,\mathrm{dB}]$).

---

## 2. Mathematical Formulations

### 2.1 B-Spline Geometry & Mollified Indicator
A closed inclusion boundary $\Gamma_{\boldsymbol{\theta}}$ is parameterized by $N_c = 32$ periodic cubic B-spline control points $\mathbf{P}_m = (X_m, Y_m) \in \mathbb{R}^2$ grouped into the 64-D parameter vector $\boldsymbol{\theta} = [X_0, \dots, X_{31}, Y_0, \dots, Y_{31}]^\top \in \mathbb{R}^{64}$ in meters ($R_0 = 1.0\,\mathrm{m}$):
$$\mathbf{q}(s) = \sum_{k=0}^3 B_k(t) \mathbf{P}_{(j+k)\bmod N_c}, \quad s = j + t, \; t \in [0, 1)$$

The internal domain membership is evaluated via the mollified boundary-integral winding indicator $W(x, y)$:
$$W(x, y) = \frac{1}{2\pi} \oint_{\Gamma_{\boldsymbol{\theta}}} \frac{(x_c(s) - x) y_c'(s) - (y_c(s) - y) x_c'(s)}{(x_c(s) - x)^2 + (y_c(s) - y)^2 + \epsilon_{\mathrm{moll}}^2} \, ds$$
where $\epsilon_{\mathrm{moll}} = 0.05\,\mathrm{m}$. The spatial conductivity distribution $\sigma(x, y; \boldsymbol{\theta})$ smoothly transitions from background $\sigma_0 = 1.0\,\mathrm{S/m}$ to inclusion $\sigma_{\mathrm{inc}} = 10^{-4}\,\mathrm{S/m}$:
$$\sigma(x, y; \boldsymbol{\theta}) = \sigma_0 + (\sigma_{\mathrm{inc}} - \sigma_0) \cdot \frac{1}{2}\left[1 + \tanh\left(\alpha \left(W(x, y) - \frac{1}{2}\right)\right)\right], \quad \alpha = 80.0$$

### 2.2 Complete Electrode Model (CEM) Forward Problem
Current injection $I_l$ through 16 boundary electrodes $e_l \subset \partial\Omega$ with contact impedance $z_l = 0.01\,\Omega\cdot\mathrm{m}^2$ generates electric potential $u$ governed by:
$$\nabla \cdot (\sigma \nabla u) = 0 \quad \text{in } \Omega$$
$$u + z_l \sigma \frac{\partial u}{\partial \mathbf{n}} = U_l \quad \text{on } e_l, \quad \int_{e_l} \sigma \frac{\partial u}{\partial \mathbf{n}} \, ds = I_l, \quad \sigma \frac{\partial u}{\partial \mathbf{n}} = 0 \quad \text{on } \partial\Omega \setminus \bigcup_{l=1}^{16} e_l$$
Under $C(16, 2) = 120$ pairwise current excitations, boundary voltages are projected through the mean-zero gauge matrix $\mathbf{M}_{\mathrm{meas}} = \mathbf{I}_{16} - \frac{1}{16}\mathbf{1}_{16}\mathbf{1}_{16}^\top \in \mathbb{R}^{16 \times 16}$, yielding the $1,920$-dimensional measurement vector $\mathbf{V}(\boldsymbol{\theta}) \in \mathbb{R}^{1920}$.

### 2.3 Directional Sobolev-Supervised Surrogate
The forward surrogate $\widehat{\mathbf{V}}(\boldsymbol{\theta}; \mathbf{w}): \mathbb{R}^{64} \to \mathbb{R}^{1920}$ is trained using the directional Sobolev objective:
$$\mathcal{L}_{\mathrm{full}}(\mathbf{w}) = \mathcal{L}_V(\mathbf{w}) + \lambda_J \mathcal{L}_{\mathrm{full}, J}(\mathbf{w})$$
$$\mathcal{L}_V = \frac{1}{B} \sum_{b=1}^B \|\widehat{\mathbf{V}}(\boldsymbol{\theta}_b; \mathbf{w}) - \mathbf{V}_b\|_2^2, \quad \mathcal{L}_{\mathrm{full}, J} = \frac{1}{B} \sum_{b=1}^B \|\widehat{\mathbf{J}}(\boldsymbol{\theta}_b; \mathbf{w}) - \mathbf{J}_b\|_F^2$$
To avoid evaluating full $1920 \times 64$ Jacobians during training, directional Jacobian-Vector Products (JVPs) are sampled on the unit hypersphere $\mathbf{d} \sim \mathrm{Unif}(\mathbb{S}^{63})$ with unbiased expectation $\mathbb{E}_{\mathbf{d}} [\|\widehat{\mathbf{J}}\mathbf{d} - \mathbf{J}\mathbf{d}\|_2^2] = \frac{1}{64} \|\widehat{\mathbf{J}} - \mathbf{J}\|_F^2$.

### 2.4 Zero-Online-FEM Levenberg-Marquardt Inversion
Reconstruction minimizes the nonlinear least-squares residual $\Phi(\boldsymbol{\theta}) = \frac{1}{2} \|\widehat{\mathbf{V}}(\boldsymbol{\theta}) - \mathbf{V}_{\mathrm{target}}\|_2^2$ without any online PDE solves:
$$(\widehat{\mathbf{J}}_k^\top \widehat{\mathbf{J}}_k + \lambda_k \mathbf{I}) \boldsymbol{\delta}_k = -\widehat{\mathbf{J}}_k^\top (\widehat{\mathbf{V}}(\boldsymbol{\theta}_k) - \mathbf{V}_{\mathrm{target}})$$
where $\lambda_k$ (units $(\mathrm{V/m})^2$) adapts dynamically via the Nielsen gain ratio $\rho_k = \frac{\Phi(\boldsymbol{\theta}_k) - \Phi(\boldsymbol{\theta}_k + \boldsymbol{\delta}_k)}{\frac{1}{2}\boldsymbol{\delta}_k^\top (\lambda_k \boldsymbol{\delta}_k - \widehat{\mathbf{J}}_k^\top \mathbf{r}_k)}$.

### 2.5 Deep Ensemble Epistemic Uncertainty Quantification
A $K=5$ Deep Ensemble generates consensus predictions and channelwise epistemic standard deviations:
$$\overline{\mathbf{V}}(\boldsymbol{\theta}) = \frac{1}{K}\sum_{k=1}^K \widehat{\mathbf{V}}^{(k)}(\boldsymbol{\theta}), \quad \boldsymbol{\sigma}_{\mathrm{epi}}(\boldsymbol{\theta}) = \sqrt{\frac{1}{K-1}\sum_{k=1}^K (\widehat{\mathbf{V}}^{(k)}(\boldsymbol{\theta}) - \overline{\mathbf{V}}(\boldsymbol{\theta}))^2}$$
Spatial epistemic uncertainty across the 2D domain is quantified by the local conductivity variance field:
$$s_\sigma^2(\mathbf{x}) = \frac{1}{K-1} \sum_{k=1}^K \left(\sigma(\mathbf{x}; \boldsymbol{\theta}^{(k)}) - \overline{\sigma}(\mathbf{x})\right)^2$$

---

## 3. Repository Architecture

```
Refined Repo/
├── main.py                             # Master CLI entry point for all workflows
├── README.md                           # Master repository documentation & quickstart guide
├── LICENSE                             # MIT License
├── requirements.txt                    # Python runtime dependencies
├── config.py                           # Central authoritative physical & solver configuration
├── REPRODUCIBILITY_MAP.md              # Claim-to-code traceability map
├── REFINED_REPO_MANIFEST.json          # Provenance manifest with SHA-256 hashes
├── NOISE_EXPERIMENT_REPORT.md          # 234-trial noise robustness experiment report
├── DEEP_ENSEMBLE_EXPERIMENT_REPORT.md  # 5-member deep ensemble experiment report
├── EXPERIMENTAL_RESULTS_SUMMARY.md     # Unified summary of all experimental benchmarks
│
├── src/                                # Core scientific library
│   ├── geometry/                       # B-spline parameterization, winding indicator & shapes
│   │   ├── bspline.py                  # Closed cubic C^2 B-spline evaluation
│   │   ├── conductivity.py             # Piecewise-constant conductivity field generator
│   │   ├── parameters.py               # 64-D parameter vector conversion, bounds & concavity
│   │   ├── shape_library.py            # Canonical shape family generators
│   │   └── shape_to_bspline.py         # Geometric shape fitting algorithms
│   ├── physics/                        # High-fidelity FEM forward solver & sensitivity
│   │   ├── fem_model.py                # Complete Electrode Model (CEM) definition
│   │   ├── fem_forward.py              # Forward voltage solver interface
│   │   ├── fem_jacobian.py             # Adjoint sensitivity matrix solver interface
│   │   ├── fem_backend.py              # EIDORS/Octave subprocess interface
│   │   └── fem_backend_optimized.py    # Persistent Octave sessions with in-Octave chain rule
│   ├── surrogate/                      # Neural forward surrogate & Sobolev loss
│   │   ├── model.py                    # VoltageSurrogate ResNet with forward-mode autodiff
│   │   ├── sobolev_loss.py             # Directional Sobolev regularized loss function
│   │   └── training.py                 # Two-stage surrogate training engine
│   ├── ensemble/                       # Deep ensemble container & epistemic spread
│   │   └── model.py                    # 5-member EnsembleModel with forward & Jacobian consensus
│   ├── inversion/                      # Zero-online-FEM inverse reconstruction engine
│   │   ├── problem.py                  # Abstract InversionProblem interface
│   │   ├── lm_solver.py                # Levenberg-Marquardt optimizer with adaptive damping
│   │   ├── multistart.py               # Multi-start solver wrapper
│   │   ├── residuals.py                # SurrogateResidualProblem definitions
│   │   ├── uncertainty.py              # EnsembleInversionProblem wrapper
│   │   └── reconstruction.py           # Geometric IoU & boundary RMS distance metrics
│   ├── validation/                     # Pure Python physical validation modules (Stages A-D)
│   │   ├── val_A_conductivity_fd.py    # Stage 1: pure Python analytic vs FD dSigma/dTheta
│   │   ├── val_B_fem_jsigma.py         # Stage 3: EIDORS J_sigma directional validation
│   │   ├── val_C_geometry_jacobian.py  # Stage 4: J_theta geometry Jacobian validation
│   │   └── val_D_matlab_comparison.py  # Stage D: Python vs MATLAB reference comparison
│   ├── noise/                          # Measurement noise models
│   │   └── noise_models.py             # Additive Gaussian noise injector
│   └── utils/                          # Hardware profiling, logging & reproducibility
│       ├── hardware.py                 # GPU/CPU hardware detection & precision timers
│       ├── io_utils.py                 # Serialization & artifact I/O
│       ├── logger.py                   # Standard scientific logger
│       └── reproducibility.py          # Deterministic seed management
│
├── matlab/                             # Standalone MATLAB / GNU Octave physics reference suite
│   ├── README.md                       # Prerequisites & validation execution guide
│   ├── polygon_conductivity_bspline.m  # Reference 64-D B-spline conductivity & Jacobian solver
│   ├── test_bspline_conductivity_derivative.m # Stage 1: mollified indicator derivative test
│   ├── test_polygon_conductivity_bspline.m    # Stage 2: full conductivity Jacobian test
│   ├── test_stage3_eidors_conductivity_jacobian.m # Stage 3: EIDORS J_sigma validation
│   ├── test_stage4_mesh_convergence.m            # Stage 4: forward solution mesh convergence
│   └── test_stage4_mesh_convergence_directional.m # Stage 4: directional Jacobian mesh convergence
│
├── visuals/                            # Visuals & Domain Schematic Generator
│   ├── README.md                       # Documentation for visual generators
│   └── plot_computational_domain_schematic.py # Fig 1 computational domain schematic
│
├── scripts/                            # Standalone execution entry points
│   ├── generate_dataset.py             # HDF5 dataset generator (requires EIDORS)
│   ├── run_inversion.py                # Zero-online-FEM deterministic inversion runner (9 targets)
│   ├── run_deep_ensemble.py            # Deep ensemble evaluation & inversion runner (9 targets)
│   ├── run_noise_robustness.py         # 6-level noise robustness benchmark runner (234 trials)
│   ├── run_master_validation.py        # Master manuscript claim verification suite
│   ├── train_surrogate.py              # Two-stage surrogate training runner
│   ├── plot_noise_results.py           # Noise robustness visualizer (Fig N1-N4)
│   ├── plot_ensemble_results.py        # Deep ensemble visualizer (Fig E1-E5)
│   ├── generate_all_ensemble_and_lm_figures.py # Master ensemble & LM visualizer
│   └── generate_publication_figures.py # Master publication figure generation engine (all 28 figures)
│
├── models/                             # Pretrained production model checkpoints
│   ├── surrogate_10k_final.pt          # Baseline Sobolev surrogate checkpoint (16.0 MB)
│   ├── deep_ensemble/                  # 5 independently trained ensemble members
│   │   ├── ensemble_member_0_seed42.pt
│   │   ├── ensemble_member_1_seed142.pt
│   │   ├── ensemble_member_2_seed242.pt
│   │   ├── ensemble_member_3_seed342.pt
│   │   └── ensemble_member_4_seed442.pt
│   └── README.md                       # Checkpoint specs & SHA-256 cryptographic hashes
│
├── data/                               # Target definitions and numerical artifacts
│   ├── targets/
│   │   └── held_out_9_targets.json     # Exact ground-truth definitions for 9 test targets
│   └── results/
│       ├── inversion_results.json      # Deterministic inversion records (9 targets)
│       ├── deep_ensemble_results.json  # Deep ensemble inversion records (9 targets)
│       ├── noise_robustness_results.json # Noise benchmark results (234 trials)
│       ├── validation_metrics.json     # Master validation metrics artifact
│       └── training_history.json       # Step-by-step training loss logs
│
├── figures/                            # Publication figures (PNG >=300 DPI and PDF)
│   ├── schematic/                      # Figure 1: 2D EIT Computational Domain Setup
│   ├── main/                           # Main Paper Figures 1 to 5
│   ├── deep_ensemble/                  # Deep Ensemble Figures E1 to E5 & Fig J3
│   ├── inversion_diagnostics/          # LM & Multi-Start Diagnostics (LM1-LM3, Fig J5)
│   └── noise_robustness/               # Noise Robustness Figures N1 to N4
│
└── tests/                              # Automated integrity & regression tests
    ├── test_end_to_end_smoke.py        # End-to-end smoke test suite (6/6 tests passing)
    └── README.md                       # Test suite documentation
```

---

## 4. Installation & Environment Setup

### Prerequisites
- **Python 3.10+**
- **PyTorch 2.0+**
- **CUDA-compatible GPU** (optional; CPU execution is fully supported)

```bash
# Clone the repository
git clone https://github.com/your-username/eit-bspline-surrogate.git
cd eit-bspline-surrogate

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Master CLI & Reproducing Results

Execute complete end-to-end workflows using the master CLI `main.py`:

### 1. Verify All Quantitative Claims & Checksums
```bash
python main.py validate-master
python -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Zero-Online-FEM Inverse Reconstructions (9 Targets)
```bash
python main.py invert
```
*Outputs:* `data/results/inversion_results.json` and diagnostic trajectory figures in `figures/inversion_diagnostics/`.

### 3. Run Deep Ensemble Benchmark & Epistemic Uncertainty
```bash
python main.py benchmark-ensemble
```
*Outputs:* `data/results/deep_ensemble_results.json` and figures in `figures/deep_ensemble/`.

### 4. Run 234-Trial Measurement Noise Robustness Benchmark
```bash
python main.py benchmark-noise
```
*Outputs:* `data/results/noise_robustness_results.json` and figures in `figures/noise_robustness/`.

### 5. Regenerate All 28 Publication Figures
```bash
python main.py generate-figures
```

---

## 6. Experimental Results & Benchmarks

### 6.1 Forward Surrogate Voltage Accuracy (1,000 Held-Out Test Samples)
| Metric | Baseline Single Surrogate | 5-Member Deep Ensemble | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Relative Voltage Error** | $0.2452\%$ | **$0.2393\%$** | $+2.41\%$ error reduction |
| **Median Relative Voltage Error** | $0.2214\%$ | **$0.2180\%$** | $+1.54\%$ error reduction |
| **90th Percentile (P90) Error** | $0.4110\%$ | **$0.3950\%$** | $+3.89\%$ error reduction |
| **Physical Voltage RMSE** | $1.2461\,\mathrm{mV}$ | **$1.2164\,\mathrm{mV}$** | $+2.38\%$ error reduction |
| **Parity Correlation ($R^2$)** | $> 0.9999$ | **$> 0.9999$** | — |

### 6.2 Zero-Online-FEM Inverse Reconstruction (9 Held-Out Targets)
| Target Geometry | Target Category | Single IoU | Ensemble IoU | Ens. RMS (mm) | Single Post-Hoc FEM Err (%) | Single Solve Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Target #1: Circle** | Convex | $0.9329$ | **$0.9782$** | $9.8\,\mathrm{mm}$ | $0.841\%$ | $8.45\,\mathrm{s}$ |
| **Target #2: Ellipse** | Convex | $0.9104$ | **$0.9654$** | $14.2\,\mathrm{mm}$ | $0.923\%$ | $8.62\,\mathrm{s}$ |
| **Target #3: Rectangle** | Convex | $0.8842$ | **$0.9310$** | $28.4\,\mathrm{mm}$ | $1.204\%$ | $8.80\,\mathrm{s}$ |
| **Target #4: Random Convex** | Convex | $0.9341$ | **$0.9774$** | $11.5\,\mathrm{mm}$ | $0.812\%$ | $8.50\,\mathrm{s}$ |
| **Target #5: Star (6-Point)** | Concave | $0.7812$ | **$0.8621$** | $44.1\,\mathrm{mm}$ | $1.650\%$ | $9.10\,\mathrm{s}$ |
| **Target #6: Banana** | Concave | $0.7640$ | **$0.8415$** | $48.2\,\mathrm{mm}$ | $1.720\%$ | $9.15\,\mathrm{s}$ |
| **Target #7: Random Concave** | Concave | $0.7920$ | **$0.8590$** | $42.0\,\mathrm{mm}$ | $1.580\%$ | $8.95\,\mathrm{s}$ |
| **Target #8: Star (5-Point)** | Concave | $0.7780$ | **$0.8492$** | $46.5\,\mathrm{mm}$ | $1.610\%$ | $9.20\,\mathrm{s}$ |
| **Target #9: Crescent** | Concave | $0.7623$ | **$0.8262$** | $51.0\,\mathrm{mm}$ | $1.880\%$ | $9.25\,\mathrm{s}$ |
| **Overall Mean** | — | **$0.8377$** | **$0.8989$** | **$32.0\,\mathrm{mm}$** | **$1.380\%$** | **$8.89\,\mathrm{s}$** |

### 6.3 Measurement Noise Robustness (234 Multi-Start Inversion Trials)
| Noise Fraction $\eta$ | Equivalent SNR | Success Rate ($\mathrm{IoU} \ge 0.50$) | Mean IoU | Median IoU | Convex Mean IoU | Concave Mean IoU | Mean Boundary RMS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.0\%$ (Clean)** | $\infty\,\mathrm{dB}$ | **$100.0\%$** ($9/9$) | $0.8335$ | $0.8262$ | $0.9154$ | $0.7755$ | $80.6\,\mathrm{mm}$ |
| **$0.1\%$** | $60.0\,\mathrm{dB}$ | **$100.0\%$** ($45/45$) | $0.8072$ | $0.8228$ | $0.8714$ | $0.7558$ | $88.3\,\mathrm{mm}$ |
| **$0.5\%$** | $46.0\,\mathrm{dB}$ | **$100.0\%$** ($45/45$) | $0.7701$ | $0.7720$ | $0.8312$ | $0.7212$ | $101.2\,\mathrm{mm}$ |
| **$1.0\%$** | $40.0\,\mathrm{dB}$ | **$100.0\%$** ($45/45$) | $0.7668$ | $0.7810$ | $0.8238$ | $0.7212$ | $108.2\,\mathrm{mm}$ |
| **$2.0\%$** | $34.0\,\mathrm{dB}$ | **$100.0\%$** ($45/45$) | $0.7410$ | $0.7612$ | $0.7950$ | $0.6978$ | $126.4\,\mathrm{mm}$ |
| **$5.0\%$** | $26.0\,\mathrm{dB}$ | **$97.78\%$** ($44/45$) | $0.7252$ | $0.7503$ | $0.7593$ | $0.6980$ | $130.1\,\mathrm{mm}$ |

---

## 7. Automated Verification & Testing

Verify repository integrity, checkpoint checksums, and numerical reproducibility:

```bash
# Run unit test suite
python -m unittest discover -s tests -p "test_*.py" -v
```

All 6 test cases verify:
1. `test_01_checkpoint_hashes`: Cryptographic SHA-256 hash match for all 6 model checkpoints.
2. `test_02_forward_surrogate_and_autodiff`: Forward inference & forward-mode autodiff Jacobian calculation.
3. `test_03_deterministic_inversion_smoke`: Multi-start LM inverse reconstruction loop.
4. `test_04_noise_robustness_smoke`: Additive Gaussian noise perturbation & reconstruction.
5. `test_05_deep_ensemble_smoke`: $K=5$ EnsembleModel evaluation & epistemic variance.
6. `test_06_result_jsons_parse`: Result JSON schema integrity and finite value constraints.

---

## 8. Citation & License

This project is released under the [MIT License](LICENSE).

```bibtex
@article{eit_bspline_surrogate_2026,
  title={Zero-Online-FEM Inverse Electrical Impedance Tomography via Directional Sobolev-Regularized B-Spline Surrogates},
  author={Research Authors},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```
