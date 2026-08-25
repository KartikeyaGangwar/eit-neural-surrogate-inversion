# Measurement Noise Robustness Experiment Report

This report presents the complete empirical evaluation of the Zero-Online-FEM inverse reconstruction pipeline under additive Gaussian measurement noise across 234 independent inversion trials.

---

## 1. Experimental Protocol

- **Inversion Method:** Multi-Start Levenberg-Marquardt optimizer with Sobolev surrogate analytical Jacobians.
- **Online FEM Solves:** $0$ ($N_{\text{FEM}}^{\text{online}} = 0$).
- **Held-Out Targets:** 9 distinct geometries spanning convex ($N=4$) and concave ($N=5$) shape families.
- **Noise Levels Evaluated:** $\delta \in \{0.0\%, 0.1\%, 0.5\%, 1.0\%, 2.0\%, 5.0\%\}$, corresponding to SNR $\in \{\infty, 60.0, 46.0, 40.0, 34.0, 26.0\}\text{ dB}$.
- **Noise Realizations:** 5 independent pseudo-random noise draws per target at each non-zero noise level (1 clean realization at $\delta = 0.0\%$).
- **Total Inversion Solves:** $9 + (5 \times 9 \times 5) = 234\text{ complete multi-start inverse reconstructions}$.

---

## 2. Quantitative Summary Across Noise Levels

| Noise Level $\eta$ (%) | Theoretical SNR (dB) | Total Solves | Mean IoU | Median IoU | 10th–90th %ile IoU | Convex IoU ($N=4$) | Concave IoU ($N=5$) | Mean Boundary RMS (m) | Mean LM Iters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** | $\infty$ (Clean) | 9 | **0.8335** | **0.8262** | [0.6614, 0.9767] | **0.9154** | **0.7681** | **0.0806** | 18.0 |
| **0.1%** | 60.0 dB | 45 | **0.8072** | **0.8228** | [0.6385, 0.9744] | **0.8917** | **0.7396** | **0.0883** | 18.8 |
| **0.5%** | 46.0 dB | 45 | **0.7701** | **0.7720** | [0.6213, 0.9482] | **0.8277** | **0.7240** | **0.1012** | 20.3 |
| **1.0%** | 40.0 dB | 45 | **0.7668** | **0.7810** | [0.6409, 0.9238] | **0.8238** | **0.7212** | **0.1082** | 21.0 |
| **2.0%** | 34.0 dB | 45 | **0.7410** | **0.7612** | [0.6133, 0.9022] | **0.7819** | **0.7083** | **0.1264** | 21.7 |
| **5.0%** | 26.0 dB | 45 | **0.7252** | **0.7503** | [0.5739, 0.8841] | **0.7593** | **0.6980** | **0.1301** | 22.8 |

---

## 3. Key Scientific Findings

1. **Graceful Performance Degradation:** Under typical experimental measurement noise ($0.5\%\text{--}1.0\%$, $46\text{--}40\text{ dB}$ SNR), the mean reconstruction IoU remains above **$0.7668$**, with convex shapes retaining **$0.8238$** mean IoU.
2. **Implicit Regularization Property:** Under severe $5.0\%$ noise ($26\text{ dB}$ SNR), the regularized B-spline parameterization prevents high-frequency noise overfitting, achieving **$0.7252$** mean IoU without numerical instability.
3. **Solver Stability:** The average number of Levenberg-Marquardt iterations increases modestly from $18.0$ (clean) to $22.8$ ($5.0\%$ noise), demonstrating solver stability.

---

## 4. Associated Figure Artifacts

- [fig_N1_noise_vs_iou_rms.png](figures/noise_robustness/fig_N1_noise_vs_iou_rms.png) / [PDF](figures/noise_robustness/fig_N1_noise_vs_iou_rms.pdf)
- [fig_N2_noise_reconstruction_gallery.png](figures/noise_robustness/fig_N2_noise_reconstruction_gallery.png) / [PDF](figures/noise_robustness/fig_N2_noise_reconstruction_gallery.pdf)
- [fig_N3_voltage_residual_degradation.png](figures/noise_robustness/fig_N3_voltage_residual_degradation.png) / [PDF](figures/noise_robustness/fig_N3_voltage_residual_degradation.pdf)
- [fig_N4_noise_robustness_comparison_table.png](figures/noise_robustness/fig_N4_noise_robustness_comparison_table.png) / [PDF](figures/noise_robustness/fig_N4_noise_robustness_comparison_table.pdf)
