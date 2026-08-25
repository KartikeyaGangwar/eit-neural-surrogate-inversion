# Dataset Specifications & Numerical Artifacts

This directory contains target definitions, quantitative evaluation metrics, and documentation for the large synthetic training dataset.

---

## 1. Directory Structure

```
data/
├── targets/
│   └── held_out_9_targets.json     # Exact ground-truth coordinates & metadata for all 9 test targets
├── results/
│   ├── validation_metrics.json     # Complete numerical validation metrics (held-out errors, AD/FD, IoU)
│   ├── inversion_results.json      # Per-target inversion metrics (IoU, Boundary RMS, solve times)
│   ├── deep_ensemble_results.json  # 5-member deep ensemble forward & inverse evaluation records
│   ├── noise_robustness_results.json # Comprehensive 234-trial measurement noise benchmark
│   ├── training_history.json       # Step-by-step AdamW & L-BFGS loss trajectories
│   └── lbfgs_convergence_log.json  # Detailed L-BFGS outer iteration convergence logs
└── README.md                       # This document
```

---

## 2. Held-Out 9 Reconstruction Targets (`data/targets/held_out_9_targets.json`)

Contains exact 64-D B-spline parameter vectors $\mathbf{\theta}_{\text{true}} \in \mathbb{R}^{64}$ and ground-truth forward boundary voltages $\mathbf{V}_{\text{fem}} \in \mathbb{R}^{1920}$ across all canonical shape families:

| Target ID | Shape Family | Geometry Type | Ground-Truth Boundary | Test Channel Count |
| :---: | :--- | :---: | :---: | :---: |
| **#1** | Circle | Convex | 32 control points | 1,920 channels |
| **#2** | Ellipse | Convex | 32 control points | 1,920 channels |
| **#3** | Rectangle | Convex | 32 control points | 1,920 channels |
| **#4** | Random Convex | Convex | 32 control points | 1,920 channels |
| **#5** | Star (6-point) | Concave | 32 control points | 1,920 channels |
| **#6** | Banana | Concave | 32 control points | 1,920 channels |
| **#7** | Random Concave | Concave | 32 control points | 1,920 channels |
| **#8** | Star (5-point) | Concave | 32 control points | 1,920 channels |
| **#9** | Crescent | Concave | 32 control points | 1,920 channels |

---

## 3. Experimental Result Artifacts

### 3.1 Deterministic Inversion Results (`data/results/inversion_results.json`)
- **Mean IoU:** $0.8377$ (Convex: $0.9154$, Concave: $0.7755$)
- **Median IoU:** $0.8262$
- **Mean Boundary RMS:** $0.0842\,\text{m}$ ($84.2\,\text{mm}$)
- **Online FEM Solves:** $0$ ($N_{\text{FEM}}^{\text{online}} = 0$)
- **Mean Solve Time:** $8.89\,\text{s}$ per multi-start target

### 3.2 Deep Ensemble Results (`data/results/deep_ensemble_results.json`)
- **Forward Voltage Error:** $0.2393\%$ mean relative error (vs $0.2452\%$ baseline)
- **Epistemic Spread Correlation:** Pearson $r = 0.5341$ ($p < 10^{-74}$), Spearman $\rho = 0.5003$ ($p < 10^{-63}$)
- **Ensemble Inversion IoU:** $0.8989$ (Convex: $0.9630$, Concave: $0.8476$)
- **Ensemble Boundary RMS Error:** $0.0418\,\text{m}$ ($41.8\,\text{mm}$ raw / $32.0\,\text{mm}$ consensus, $-50.34\%$ reduction)

### 3.3 Measurement Noise Robustness Results (`data/results/noise_robustness_results.json`)
- Evaluates 6 noise conditions ($0.0\%, 0.1\%, 0.5\%, 1.0\%, 2.0\%, 5.0\%$) across 234 inversion trials with 100% numerical solver completion.
- **Clean ($0.0\%$ noise, $\infty\,\text{dB}$):** Mean $\text{IoU} = 0.8335$, Median $\text{IoU} = 0.8262$, $\text{RMS} = 0.0806\,\text{m}$ ($80.6\,\text{mm}$)
- **$0.1\%$ Noise ($60\,\text{dB}$ SNR):** Mean $\text{IoU} = 0.8072$, Median $\text{IoU} = 0.8228$, $\text{RMS} = 0.0883\,\text{m}$
- **$0.5\%$ Noise ($46\,\text{dB}$ SNR):** Mean $\text{IoU} = 0.7701$, Median $\text{IoU} = 0.7720$, $\text{RMS} = 0.1012\,\text{m}$
- **$1.0\%$ Noise ($40\,\text{dB}$ SNR):** Mean $\text{IoU} = 0.7668$, Median $\text{IoU} = 0.7810$, $\text{RMS} = 0.1082\,\text{m}$
- **$2.0\%$ Noise ($34\,\text{dB}$ SNR):** Mean $\text{IoU} = 0.7410$, Median $\text{IoU} = 0.7612$, $\text{RMS} = 0.1264\,\text{m}$
- **$5.0\%$ Noise ($26\,\text{dB}$ SNR):** Mean $\text{IoU} = 0.7252$, Median $\text{IoU} = 0.7503$, $\text{RMS} = 0.1301\,\text{m}$

---

## 4. Full Training Dataset Specifications (`dataset_10k.h5`)

The full high-fidelity synthetic forward dataset contains 10,000 Complete Electrode Model (CEM) FEM forward solves. Due to repository size recommendations, the raw HDF5 dataset can be regenerated locally or downloaded from research archives.

### Dataset Schema
- `theta`: Shape `(10000, 64)`, `float32` — Control point coordinates $(x_i, y_i)_{i=1}^{32}$.
- `voltage`: Shape `(10000, 1920)`, `float32` — Boundary differential voltages across 16 adjacent injections.
- `jacobian`: Shape `(10000, 1920, 64)`, `float32` — Adjoint FEM sensitivity matrices.
- `completed_mask`: Shape `(10000,)`, `uint8` — 100% complete samples ($N=10,000$).
- `train_indices`: Samples `0` to `7999` ($N_{\text{train}} = 8,000$).
- `val_indices`: Samples `8000` to `8999` ($N_{\text{val}} = 1,000$).
- `test_indices`: Samples `9000` to `9999` ($N_{\text{test}} = 1,000$).
