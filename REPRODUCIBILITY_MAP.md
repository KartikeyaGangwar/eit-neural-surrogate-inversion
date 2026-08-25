# Scientific Reproducibility & Claim-to-Code Traceability Map

This document establishes the bidirectional traceability matrix connecting every mathematical equation, physical formulation, and empirical claim in the research manuscript to its exact code implementation and quantitative artifact in this repository.

---

## 1. Physical & Mathematical Formulations

| Manuscript Component | Theoretical Formulation | Implementation Module | Authoritative Artifact |
| :--- | :--- | :--- | :--- |
| **Complete Electrode Model (CEM)** | $\nabla \cdot (\sigma \nabla u) = 0 \text{ in } \Omega$, $u + z_l \sigma \frac{\partial u}{\partial n} = U_l \text{ on } e_l$ | `src/physics/fem_model.py`, `src/physics/fem_forward.py` | [fig1_pipeline_overview.png](figures/main/fig1_pipeline_overview.png) |
| **B-Spline Boundary Curve** | $\Gamma(t) = \sum_{j=1}^{32} \mathbf{p}_j B_{j,3}(t)$, $\mathbf{\theta} \in \mathbb{R}^{64}$ | `src/geometry/bspline.py` | `data/targets/held_out_9_targets.json` |
| **Mollified Conductivity Field** | $\sigma(\mathbf{x}; \mathbf{\theta}) = \sigma_{\text{bg}} + (\sigma_{\text{inc}} - \sigma_{\text{bg}})\chi_\epsilon(\mathbf{x}; \mathbf{\theta})$ | `src/geometry/conductivity.py` | [val_A_conductivity_fd.py](src/validation/val_A_conductivity_fd.py) |
| **Sobolev Surrogate Loss** | $\mathcal{L} = \text{MSE}(V_{\text{norm}}) + \lambda_J \text{MSE}(J_{\text{norm}} \mathbf{v})$ | `src/surrogate/sobolev_loss.py` | [fig2_training_convergence.png](figures/main/fig2_training_convergence.png) |
| **Forward-Mode Autodiff Jacobian** | $J(\mathbf{\theta}) = \frac{\partial \mathcal{S}_\phi}{\partial \mathbf{\theta}}$ via `torch.func.jacfwd` | `src/surrogate/model.py` | [fig3_forward_surrogate_accuracy.png](figures/main/fig3_forward_surrogate_accuracy.png) |
| **Zero-Online-FEM Inverse Solver** | Multi-Start Levenberg-Marquardt ($N_{\text{FEM}}^{\text{online}} = 0$) | `src/inversion/lm_solver.py`, `src/inversion/multistart.py` | `data/results/inversion_results.json` |
| **Deep Ensemble Mean & Spread** | $\bar{V}(\mathbf{\theta}) = \frac{1}{K}\sum \mathcal{S}_{\phi_k}(\mathbf{\theta})$, $\sigma_{\text{ens}}^2(\mathbf{\theta})$ | `src/ensemble/model.py` | `data/results/deep_ensemble_results.json` |
| **Measurement Noise Robustness** | $V_{\text{noisy}} = V + \mathcal{N}(0, \delta^2 \text{diag}(\text{std}(V))^2)$ | `src/noise/noise_models.py` | `data/results/noise_robustness_results.json` |

---

## 2. Quantitative Claims & Verification Evidence

| Claim # | Manuscript Claim Description | Empirical Value | Target Metric / Bound | Verification Status | Artifact Location |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Forward Surrogate Relative Voltage Error | **$0.2452\%$** | $< 0.30\%$ | **VERIFIED** | `data/results/validation_metrics.json` |
| **2** | Autodiff vs Central FD Jacobian Discrepancy | **$8.54 \times 10^{-4}$** | $< 10^{-3}$ | **VERIFIED** | `data/results/validation_metrics.json` |
| **3** | Exact Physical Scaling Identity Difference | **$6.71 \times 10^{-8}$** | $< 10^{-6}$ | **VERIFIED** | `data/results/validation_metrics.json` |
| **4** | Online FEM Solves During Inverse Optimization | **$0$** | $= 0$ | **VERIFIED** | `data/results/inversion_results.json` |
| **5** | Deterministic Baseline Inversion Mean IoU | **$0.8377$** | $> 0.80$ | **VERIFIED** | `data/results/inversion_results.json` |
| **6** | Deep Ensemble Inversion Mean IoU | **$0.8989$** | $> 0.85$ | **VERIFIED** | `data/results/deep_ensemble_results.json` |
| **7** | Deep Ensemble Boundary RMS Reduction | **$50.34\%$** | $> 20.0\%$ | **VERIFIED** | `data/results/deep_ensemble_results.json` |
| **8** | Epistemic Uncertainty vs Error Pearson $r$ | **$0.5341$** | $p < 10^{-70}$ | **VERIFIED** | `data/results/deep_ensemble_results.json` |
| **9** | Noise Robustness at $1.0\%$ Noise ($40\text{ dB}$ SNR) | **$0.7668$** IoU | $> 0.75$ | **VERIFIED** | `data/results/noise_robustness_results.json` |
| **10**| Severe Noise Tolerance at $5.0\%$ Noise ($26\text{ dB}$ SNR) | **$0.7252$** IoU | $> 0.70$ | **VERIFIED** | `data/results/noise_robustness_results.json` |

---

## 3. Cryptographic Provenance

All repository assets and model weights are tracked in [`REFINED_REPO_MANIFEST.json`](REFINED_REPO_MANIFEST.json) with SHA-256 hashes.
