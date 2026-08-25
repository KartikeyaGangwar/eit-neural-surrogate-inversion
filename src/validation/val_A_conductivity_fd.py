"""
Stage 1: Isolated conductivity derivative validation.
======================================================

Exact Python replication of test_bspline_conductivity_derivative.m.

PURPOSE
-------
Validate the analytic dSigma/dTheta against central finite differences.
This test is PURE PYTHON — no EIDORS or Octave required.
Run locally or on Colab.

EXPECTED OUTPUT
---------------
For each tested parameter k and step h:
    Relative error decreases as h shrinks from 1e-2 to ~1e-5,
    then increases again as floating-point noise dominates.

PASS CRITERION
--------------
For each tested parameter:
    min(relative_error over all h) < 1e-2

Run::

    python validation/val_A_conductivity_fd.py
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from geometry.conductivity import compute_sigma, compute_sigma_and_derivative
from geometry.parameters import sample_reference_geometry, control_points_to_theta


def main() -> None:
    """
    Run Stage 1 conductivity derivative validation against central finite differences.
    """
    print("=" * 60)
    print("STAGE 1: CONDUCTIVITY DERIVATIVE VALIDATION (Python FD)")
    print("=" * 60)
    print("Reference: test_bspline_conductivity_derivative.m")
    print()

    # ----------------------------------------------------------------
    # Setup: same as MATLAB (N=2000, rng(42))
    # ----------------------------------------------------------------
    rng = np.random.default_rng(42)
    N = 2000
    centres = (rng.random((N, 2)) - 0.5) * 2.0  # uniform in [-1,1]^2

    # Reference geometry (exact MATLAB control points)
    control_points = np.array([
        [ 0.45,  0.00], [ 0.43,  0.10], [ 0.36,  0.19], [ 0.25,  0.27],
        [ 0.12,  0.31], [-0.02,  0.30], [-0.15,  0.25], [-0.28,  0.18],
        [-0.38,  0.07], [-0.42, -0.06], [-0.38, -0.18], [-0.29, -0.27],
        [-0.17, -0.31], [-0.04, -0.30], [ 0.06, -0.24], [ 0.03, -0.13],
        [-0.05, -0.06], [-0.10,  0.03], [-0.07,  0.12], [ 0.02,  0.17],
        [ 0.13,  0.16], [ 0.22,  0.10], [ 0.28,  0.00], [ 0.25, -0.10],
        [ 0.18, -0.17], [ 0.27, -0.22], [ 0.38, -0.19], [ 0.46, -0.12],
        [ 0.49, -0.04], [ 0.48,  0.02], [ 0.47,  0.05], [ 0.45,  0.00],
    ], dtype=float)
    # Authoritative packing: theta = [X1..X32, Y1..Y32]
    theta = control_points_to_theta(control_points)

    # MATLAB step sizes and test parameters (converted to 0-indexed)
    h_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    test_params_0indexed = [0, 1, 16, 31, 47, 63]  # MATLAB [1,2,17,32,48,64]

    print("Computing analytic dSigma/dTheta...")
    _, dSigma_an = compute_sigma_and_derivative(theta, centres)
    print(f"dSigma_an shape: {dSigma_an.shape}  (expected ({N}, 64))")
    print()

    print("Conductivity Central Finite Difference Errors:")
    print()

    all_pass = True
    for k in test_params_0indexed:
        print(f"--- Parameter {k} (MATLAB index {k+1}) ---")
        min_err = np.inf
        for h in h_list:
            theta_p = theta.copy()
            theta_p[k] += h
            sigma_p = compute_sigma(theta_p, centres)

            theta_m = theta.copy()
            theta_m[k] -= h
            sigma_m = compute_sigma(theta_m, centres)

            fd_col = (sigma_p - sigma_m) / (2.0 * h)
            an_col = dSigma_an[:, k]

            denom = np.linalg.norm(fd_col)
            if denom < 1e-14:
                rel_err = np.linalg.norm(fd_col - an_col)
            else:
                rel_err = np.linalg.norm(fd_col - an_col) / denom

            min_err = min(min_err, rel_err)
            print(f"  h = {h:.0e} : Relative Error = {rel_err:.12e}")

        param_pass = min_err < 1e-2
        if not param_pass:
            all_pass = False
        status = "PASS" if param_pass else "FAIL"
        print(f"  Min error = {min_err:.3e}  [{status}]")
        print()

    print("=" * 60)
    print(f"STAGE 1 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    if all_pass:
        print("All parameters passed: min relative FD error < 1e-2.")
    else:
        print("One or more parameters FAILED: check conductivity.py implementation.")


if __name__ == "__main__":
    main()
