# Unified Experimental Results Summary

This document aggregates the primary quantitative results of the Sobolev-regularized B-spline EIT project.

---

## 1. Summary of Primary Quantitative Benchmarks

### 1.1 Forward Surrogate Accuracy (1,000 Held-Out Samples)
- **Mean Relative Voltage Error:** $0.2452\%$
- **Median Relative Voltage Error:** $0.2214\%$
- **90th Percentile Relative Error:** $0.4110\%$
- **Physical Voltage RMSE:** $1.2461\text{ mV}$
- **Parity Correlation ($R^2$):** $> 0.9999$

### 1.2 Derivative & Scaling Identities
- **Autodiff vs Central Finite-Difference Discrepancy:** $8.54 \times 10^{-4}$ median relative error
- **Physical Scaling Identity Error ($\|\mathbf{J}_{\text{phys}} - \mathbf{S}_V \mathbf{J}_{\text{norm}}\|_\infty$):** $6.71 \times 10^{-8}$
- **Online FEM Solves:** $0$ ($N_{\text{FEM}}^{\text{online}} = 0$)

### 1.3 Deterministic Inverse Reconstruction (9 Held-Out Targets)
- **Overall Mean IoU:** $0.8006$
- **Overall Median IoU:** $0.7846$
- **Convex Subgroup Mean IoU ($N=4$):** $0.8884$
- **Concave Subgroup Mean IoU ($N=5$):** $0.7305$
- **Mean Boundary RMS Distance:** $0.0323\text{ m}$ ($32.3\text{ mm}$)
- **Mean Solve Time:** $3.39\text{ s}$ to convergence ($8.89\text{ s}$ matched 250-step budget)

### 1.4 Deep Ensemble Performance ($K=5$ Members)
- **Forward Mean Relative Error:** $0.2393\%$ ($+2.41\%$ accuracy improvement)
- **Epistemic Uncertainty Correlation:** Pearson $r = 0.5341$, Spearman $\rho = 0.5003$ ($p < 10^{-63}$)
- **Ensemble Mean IoU:** **$0.8235$** (Convex: **$0.8828$**, Concave: **$0.7760$**)
- **Ensemble Mean Boundary RMS:** **$0.0320\text{ m}$** ($32.0\text{ mm}$)

### 1.5 Measurement Noise Robustness (234 Inversion Trials)
- **Clean ($0.0\%$ noise):** Mean $\text{IoU} = 0.8335$, Median $\text{IoU} = 0.8262$, $\text{RMS} = 0.0806\text{ m}$
- **$0.1\%$ Noise ($60\text{ dB}$ SNR):** Mean $\text{IoU} = 0.8072$, Median $\text{IoU} = 0.8228$, $\text{RMS} = 0.0883\text{ m}$
- **$0.5\%$ Noise ($46\text{ dB}$ SNR):** Mean $\text{IoU} = 0.7701$, Median $\text{IoU} = 0.7720$, $\text{RMS} = 0.1012\text{ m}$
- **$1.0\%$ Noise ($40\text{ dB}$ SNR):** Mean $\text{IoU} = 0.7668$, Median $\text{IoU} = 0.7810$, $\text{RMS} = 0.1082\text{ m}$
- **$2.0\%$ Noise ($34\text{ dB}$ SNR):** Mean $\text{IoU} = 0.7410$, Median $\text{IoU} = 0.7612$, $\text{RMS} = 0.1264\text{ m}$
- **$5.0\%$ Noise ($26\text{ dB}$ SNR):** Mean $\text{IoU} = 0.7252$, Median $\text{IoU} = 0.7503$, $\text{RMS} = 0.1301\text{ m}$
