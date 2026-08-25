"""
Stage 4: End-to-end geometry Jacobian validation.
==================================================

PURPOSE
-------
Validate J_theta = dV/dtheta using directional finite differences in
theta-space (not sigma-space).

    For normalized direction d (in theta-space):
        analytic:  J_theta @ d
        FD:        [V(theta+h*d) - V(theta-h*d)] / (2h)

This is the end-to-end validation of the full chain:
    theta -> sigma -> EIDORS -> voltage
    J_theta = J_sigma @ dSigma_dTheta

PREREQUISITES
-------------
  - Octave installed and on PATH
  - EIDORS accessible at config.EIDORS_PATH
  - Run LOCALLY (not on Colab)

PASS CRITERION
--------------
For each of 5 test directions in theta-space:
    min(relative_error over all h) < 0.01

Run::

    python validation/val_C_geometry_jacobian.py
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from physics.fem_backend import FEMBackend
    EIDORS_AVAILABLE = True
except ImportError as e:
    EIDORS_AVAILABLE = False
    _import_error = str(e)

from geometry.parameters import sample_reference_geometry
import config as cfg


def main() -> None:
    """
    Run Stage 4 geometry sensitivity Jacobian chain-rule directional validation.
    """
    if not EIDORS_AVAILABLE:
        print(f"[SKIP] EIDORS not available: {_import_error}")
        print("Run this script LOCALLY with Octave + EIDORS installed.")
        return

    print("=" * 60)
    print("STAGE 4: GEOMETRY JACOBIAN DIRECTIONAL VALIDATION")
    print("=" * 60)
    print("Reference: test_stage4_mesh_convergence_directional.m")
    print()

    backend = FEMBackend(verbose=True)
    print("Building FEM model...")
    backend.build_model()

    theta = sample_reference_geometry()

    print("Computing J_theta via solve_with_jacobian...")
    V0, J_theta = backend.solve_with_jacobian(theta)
    print(f"V0 shape:     {V0.shape}      (expected ({cfg.N_MEAS},))")
    print(f"J_theta shape: {J_theta.shape}  (expected ({cfg.N_MEAS}, {cfg.N_THETA}))")

    # 5 random normalized directions in theta-space
    rng = np.random.default_rng(99)
    directions = []
    for i in range(5):
        d = rng.standard_normal(cfg.N_THETA)
        d /= np.linalg.norm(d)
        directions.append(d)

    h_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    all_pass = True
    for i, d in enumerate(directions):
        print(f"\n=== Direction {i+1} / {len(directions)} ===")
        analytic = J_theta @ d  # (1920,)
        print(f"||J_theta*d|| = {np.linalg.norm(analytic):.6e}")
        min_err = np.inf
        for h in h_list:
            V_plus  = backend.solve(theta + h * d)
            V_minus = backend.solve(theta - h * d)
            fd = (V_plus - V_minus) / (2.0 * h)
            denom = np.linalg.norm(analytic)
            rel_err = np.linalg.norm(fd - analytic) / max(denom, 1e-14)
            min_err = min(min_err, rel_err)
            print(f"  h = {h:.0e} : relative error = {rel_err:.12e}")
        passed = min_err < 0.01
        if not passed:
            all_pass = False
        print(f"  Min error = {min_err:.3e}  [{'PASS' if passed else 'FAIL'}]")

    print("\n" + "=" * 60)
    print(f"STAGE 4 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    if all_pass:
        print("All directions passed: min relative FD error < 1%.")
    else:
        print("One or more directions FAILED.")
        print("Check: geometry/conductivity.py (eps2 bug?), fem_backend.py.")


if __name__ == "__main__":
    main()
