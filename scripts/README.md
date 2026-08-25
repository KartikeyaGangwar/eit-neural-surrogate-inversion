# Execution Scripts & Figure Generators

This directory contains standalone execution runners, dataset generators, and publication figure generators.

---

## 1. Script Inventory

| Script | Purpose | Key Outputs | EIDORS Required? |
| :--- | :--- | :--- | :---: |
| `generate_dataset.py` | Generates chunked HDF5 datasets of $(\boldsymbol{\theta}, \mathbf{V}, \mathbf{J}_{\boldsymbol{\theta}})$ via EIDORS Complete Electrode Model. | `data/dataset.h5` | **Yes** |
| `run_inversion.py` | Multi-start Levenberg-Marquardt shape reconstruction on 9 held-out targets ($N_{\text{FEM}}^{\text{online}} = 0$). | Console metrics & IoU breakdown | No |
| `run_deep_ensemble.py` | Evaluates $K=5$ deep ensemble forward models & inverse reconstructions with epistemic spread. | `data/results/deep_ensemble_results.json` | No |
| `run_noise_robustness.py` | Executes 234-trial measurement noise benchmark across 6 noise levels ($\delta \in [0.0\%, 5.0\%]$). | `data/results/noise_robustness_results.json` | No |
| `run_master_validation.py` | Validates all quantitative claims reported in the manuscript against pre-computed artifacts. | `data/results/validation_metrics.json` | No |
| `train_surrogate.py` | Two-stage training pipeline (AdamW + L-BFGS) with directional Sobolev JVP regularization. | `models/surrogate_10k_final.pt` | No |
| `plot_noise_results.py` | Generates noise robustness publication figures (`fig_N1`–`fig_N3`). | `figures/noise_robustness/` | No |
| `generate_all_ensemble_and_lm_figures.py` | Generates Deep Ensemble (`fig_E1`–`fig_E5`) and LM diagnostic figures (`fig_LM1`–`fig_MS1`). | `figures/deep_ensemble/`, `figures/inversion_diagnostics/` | No |
| `generate_publication_figures.py` | Master figure generation suite for all 28 figures in PNG ($\ge 300\text{ DPI}$) and vector PDF. | `figures/` (all subdirectories) | No |

---

## 2. Usage Examples

### Generate Synthetic FEM Dataset (Requires local Octave + EIDORS)
```bash
# Standard single-process generation
python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5

# Fast persistent session generation
python scripts/generate_dataset.py --n-samples 10000 --output data/dataset.h5 --optimized --workers 4
```

### Run Deterministic Inversion ($N_{\text{FEM}}^{\text{online}} = 0$)
```bash
python scripts/run_inversion.py
```

### Run Measurement Noise Benchmark
```bash
python scripts/run_noise_robustness.py
```

### Run Deep Ensemble Benchmark
```bash
python scripts/run_deep_ensemble.py
```

### Regenerate All Publication Figures
```bash
python scripts/generate_publication_figures.py
```
