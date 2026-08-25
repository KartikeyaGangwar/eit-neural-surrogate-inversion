# Automated Smoke & Integrity Test Suite

This directory contains automated unit tests and end-to-end regression tests verifying repository integrity, model checkpoint checksums, forward inference accuracy, analytical Jacobian computation, and inverse reconstruction routines.

---

## 1. Test Suite Overview

`tests/test_end_to_end_smoke.py`:
- `test_01_checkpoint_hashes`: Cryptographically verifies SHA-256 checksums of all 6 production model checkpoints.
- `test_02_forward_surrogate_and_autodiff`: Validates forward voltage prediction and analytical autodiff Jacobian computation against domain bounds.
- `test_03_deterministic_inversion_smoke`: Validates zero-online-FEM Levenberg-Marquardt and MultiStart inversion pipelines on held-out test data.
- `test_04_noise_robustness_smoke`: Validates additive Gaussian noise injection and noisy inversion reconstruction.
- `test_05_deep_ensemble_smoke`: Validates $K=5$ EnsembleModel loading, epistemic predictive spread calculation, and ensemble inversion.
- `test_06_result_jsons_parse`: Validates all JSON result artifacts for schema compliance, valid finite numbers, and complete metadata.

---

## 2. Running Tests

```bash
# Run full smoke test suite
python -m unittest discover -s tests -p "test_*.py" -v
```
