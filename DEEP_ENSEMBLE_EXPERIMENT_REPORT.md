# Deep Ensemble & Epistemic Uncertainty Quantification Report

This report documents the performance of the $K=5$ Deep Ensemble surrogate architecture across forward voltage prediction accuracy, epistemic uncertainty calibration, and Zero-Online-FEM inverse shape reconstruction across all 9 held-out targets.

---

## 1. Ensemble Architecture & Members

Five independent VoltageSurrogate models were trained with identical architectures ($6$ ResNet blocks, hidden dimension $512$, $\text{SiLU}$ activations) from different random initialization seeds on the 10,000-sample Sobolev dataset:

| Member Index | Seed | Checkpoint File | Size | SHA-256 Checksum |
| :---: | :---: | :--- | :---: | :--- |
| **Member #1** | 42 | `models/deep_ensemble/ensemble_member_0_seed42.pt` | 16.03 MB | `8301aa04adf710fcfd8ad12ae12e67dc4c623a576b7d6061bc0e4cd6042753a5` |
| **Member #2** | 142 | `models/deep_ensemble/ensemble_member_1_seed142.pt` | 16.01 MB | `bbbf3d996ce8f624388809e76e6a87438508ef2daf46adf7b3d88d0d11969e5e` |
| **Member #3** | 242 | `models/deep_ensemble/ensemble_member_2_seed242.pt` | 16.01 MB | `64606da26052705094b01ced47664831088ead244d371ac7c5a3832be4ece4aa` |
| **Member #4** | 342 | `models/deep_ensemble/ensemble_member_3_seed342.pt` | 16.01 MB | `d48187001bfcd0a83a782027e6ec50be7e80c57d4230a5584c433a8f39060dd0` |
| **Member #5** | 442 | `models/deep_ensemble/ensemble_member_4_seed442.pt` | 16.01 MB | `8829e4dcacd18b817513720d1659e25850f0f0c1697966539ea392333d10a8c8` |

---

## 2. Forward Evaluation & Epistemic Uncertainty Calibration

- **Forward Voltage Relative Error:** Reduced from **$0.2452\%$** (single baseline) to **$0.2393\%$** (ensemble mean), yielding a **$+2.41\%$** accuracy gain.
- **Physical Voltage RMSE:** Reduced to **$1.2164\text{ mV}$**.
- **Epistemic Spread vs. Ground-Truth Error:** Pearson correlation $r = \mathbf{0.5341}$ ($p = 7.66 \times 10^{-75}$), Spearman $\rho = \mathbf{0.5003}$ ($p = 1.82 \times 10^{-64}$).

---

## 3. Inverse Reconstruction Benchmark (9 Held-Out Targets)

| Target | Shape Family | Geometry Type | Baseline IoU | Ensemble IoU | IoU Gain | Baseline RMS (m) | Ensemble RMS (m) | RMS Reduction | Ensemble Spread (mV) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | Circle | Convex | 0.9767 | **0.9841** | +0.0074 | 0.0163 | **0.0084** | 48.5% | 0.12 |
| **#2** | Ellipse | Convex | 0.9279 | **0.9515** | +0.0236 | 0.0331 | **0.0172** | 48.0% | 0.30 |
| **#3** | Rectangle | Convex | 0.8262 | **0.9570** | +0.1308 | 0.1250 | **0.0312** | 75.0% | 0.26 |
| **#4** | Random Convex | Convex | 0.9307 | **0.9594** | +0.0287 | 0.0274 | **0.0151** | 44.9% | 0.24 |
| **#5** | Star (6-point) | Concave | 0.6614 | **0.9286** | +0.2672 | 0.1082 | **0.0171** | 84.2% | 0.90 |
| **#6** | Banana | Concave | 0.7503 | **0.7554** | +0.0051 | 0.1912 | **0.0940** | 50.8% | 4.29 |
| **#7** | Random Concave | Concave | 0.7839 | **0.7731** | -0.0108 | 0.0814 | **0.0712** | 12.5% | 0.26 |
| **#8** | Star (5-point) | Concave | 0.8063 | **0.8840** | +0.0777 | 0.0498 | **0.0301** | 39.6% | 0.38 |
| **#9** | Crescent | Concave | 0.8757 | **0.8966** | +0.0209 | 0.1268 | **0.0922** | 27.3% | 1.83 |
| **MEAN**| **All 9 Targets**| **Overall** | **0.8377** | **0.8989** | **+0.0612** | **0.0842** | **0.0418** | **50.34%** | **0.95** |

---

## 4. Associated Figure Artifacts

- [fig_E1_epistemic_disagreement_spectrum.png](figures/deep_ensemble/fig_E1_epistemic_disagreement_spectrum.png) / [PDF](figures/deep_ensemble/fig_E1_epistemic_disagreement_spectrum.pdf)
- [fig_E2_epistemic_uncertainty_calibration.png](figures/deep_ensemble/fig_E2_epistemic_uncertainty_calibration.png) / [PDF](figures/deep_ensemble/fig_E2_epistemic_uncertainty_calibration.pdf)
- [fig_E3_measurement_uncertainty_profile.png](figures/deep_ensemble/fig_E3_measurement_uncertainty_profile.png) / [PDF](figures/deep_ensemble/fig_E3_measurement_uncertainty_profile.pdf)
- [fig_E4A_reconstruction_targets_1_to_5.png](figures/deep_ensemble/fig_E4A_reconstruction_targets_1_to_5.png) / [PDF](figures/deep_ensemble/fig_E4A_reconstruction_targets_1_to_5.pdf)
- [fig_E4B_reconstruction_targets_6_to_9.png](figures/deep_ensemble/fig_E4B_reconstruction_targets_6_to_9.png) / [PDF](figures/deep_ensemble/fig_E4B_reconstruction_targets_6_to_9.pdf)
- [fig_E5_inversion_comparison_matrix.png](figures/deep_ensemble/fig_E5_inversion_comparison_matrix.png) / [PDF](figures/deep_ensemble/fig_E5_inversion_comparison_matrix.pdf)
