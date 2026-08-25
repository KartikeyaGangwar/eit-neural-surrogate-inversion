# Publication Figures Directory

This directory contains the full suite of 28 high-resolution publication figures (>=300 DPI PNG and vector PDF) produced by the benchmark pipelines.

---

## Directory Organization

```
figures/
├── schematic/                  # Computational domain & electrode setup (Fig 1)
│   ├── fig1_computational_domain_schematic.png
│   └── fig1_computational_domain_schematic.pdf
├── main/                       # Main Manuscript Figures
│   ├── fig3_forward_surrogate_accuracy.pdf
│   ├── fig_J1_sobolev_regularization_ablation.pdf
│   ├── fig4A_zero_fem_inversion_targets1_3.pdf
│   ├── fig4B_zero_fem_inversion_targets4_6.pdf
│   └── fig4C_zero_fem_inversion_targets7_9.pdf
├── deep_ensemble/              # Deep Ensemble & Epistemic Uncertainty (Fig E1-E5, J3)
│   ├── fig_J3_2d_spatial_epistemic_uncertainty.png
│   ├── fig_J3_2d_spatial_epistemic_uncertainty.pdf
│   ├── fig_E1_ensemble_forward_voltage_parity.png
│   ├── fig_E2_ensemble_error_reduction_distributions.png
│   ├── fig_E3_epistemic_uncertainty_correlation.png
│   ├── fig_E4A_ensemble_inversion_convex_targets.png
│   ├── fig_E4B_ensemble_inversion_concave_targets.png
│   └── fig_E5_ensemble_rms_error_reduction.png
├── noise_robustness/           # Measurement Noise Robustness Benchmark (Fig N1-N4)
│   ├── fig_N1_noise_vs_iou_rms.png
│   ├── fig_N1_noise_vs_iou_rms.pdf
│   ├── fig_N2_noise_boxplots_by_target.png
│   ├── fig_N3_reconstruction_degradation_overlays.png
│   └── fig_N4_snr_sensitivity_curve.png
└── inversion_diagnostics/      # Levenberg-Marquardt & Multi-Start Diagnostics (Fig LM1-LM3, J5)
    ├── fig_J5_multistart_optimization_landscape.png
    ├── fig_J5_multistart_optimization_landscape.pdf
    ├── fig_LM1_convergence_trajectories.png
    ├── fig_LM2_damping_parameter_evolution.png
    └── fig_MS1_multistart_energy_barplots.png
```

---

## Figure Regeneration

To regenerate all publication figures programmatically:
```bash
python main.py generate-figures
```
