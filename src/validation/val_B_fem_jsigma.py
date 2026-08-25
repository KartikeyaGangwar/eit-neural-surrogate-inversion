"""
Stage 3: Direct validation of EIDORS conductivity Jacobian J_sigma.
=====================================================================

Replication of test_stage3_eidors_conductivity_jacobian.m in Python.

PURPOSE
-------
Validate J_sigma = dV/dSigma using directional finite differences.

    For normalized direction d (in sigma-space):
        analytic:  J_sigma @ d
        FD:        [V(sigma + h*d) - V(sigma - h*d)] / (2h)

PREREQUISITES
-------------
  - Octave installed and on PATH
  - EIDORS accessible at config.EIDORS_PATH
  - Run LOCALLY (not on Colab)

PASS CRITERION
--------------
For each of 6 test directions:
    min(relative_error over all h) < 1e-3

Run::

    python validation/val_B_fem_jsigma.py
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from physics.fem_backend import FEMBackend
    from physics.fem_model import FEMModelConfig
    EIDORS_AVAILABLE = True
except ImportError as e:
    EIDORS_AVAILABLE = False
    _import_error = str(e)

from geometry.parameters import sample_reference_geometry
from geometry.conductivity import compute_sigma
import config as cfg


def _fem_voltage_from_sigma(
    sigma: np.ndarray, backend: "FEMBackend"
) -> np.ndarray:
    """Solve forward problem directly from sigma vector (not theta)."""
    import scipy.io as sio
    import subprocess
    import textwrap

    sigma_path = backend.work_dir / "sigma_dir.mat"
    sio.savemat(str(sigma_path), {"sigma": sigma.reshape(-1, 1)})

    result_path = backend.work_dir / "result_dir.mat"
    ep = backend.eidors_path
    model_mat = backend._model_mat.as_posix()
    sig_mat = sigma_path.as_posix()
    res_mat = result_path.as_posix()

    script = textwrap.dedent(f"""\
        run('{ep}/startup.m');
        load('{model_mat}', 'mdl');
        d = load('{sig_mat}');
        sigma = d.sigma(:);
        img = eidors_obj('image', 'dir_test', 'elem_data', sigma, 'fwd_model', mdl);
        data = fwd_solve(img);
        voltage = data.meas(:);
        save('-v6', '{res_mat}', 'voltage');
    """)
    script_path = backend.work_dir / "run_dir.m"
    script_path.write_text(script, encoding="utf-8")
    backend._run_octave(script_path, label="dir_test")

    data = sio.loadmat(str(result_path))
    return np.array(data["voltage"]).flatten()


def _fem_jsigma(sigma: np.ndarray, backend: "FEMBackend") -> np.ndarray:
    """Compute J_sigma = calc_jacobian for given sigma vector."""
    import scipy.io as sio
    import textwrap

    sigma_path = backend.work_dir / "sigma_jsig.mat"
    sio.savemat(str(sigma_path), {"sigma": sigma.reshape(-1, 1)})

    result_path = backend.work_dir / "result_jsig.mat"
    ep = backend.eidors_path
    model_mat = backend._model_mat.as_posix()

    script = textwrap.dedent(f"""\
        run('{ep}/startup.m');
        load('{model_mat}', 'mdl');
        d = load('{sigma_path.as_posix()}');
        sigma = d.sigma(:);
        img = eidors_obj('image', 'jsig', 'elem_data', sigma, 'fwd_model', mdl);
        J_sigma = calc_jacobian(img);
        save('-v6', '{result_path.as_posix()}', 'J_sigma');
        fprintf('J_sigma size: [%d %d]\\n', size(J_sigma,1), size(J_sigma,2));
    """)
    script_path = backend.work_dir / "run_jsig.m"
    script_path.write_text(script, encoding="utf-8")
    backend._run_octave(script_path, label="jsigma")

    data = sio.loadmat(str(result_path))
    return np.array(data["J_sigma"])


def main() -> None:
    """
    Run Stage 3 FEM conductivity sensitivity Jacobian directional validation.
    """
    if not EIDORS_AVAILABLE:
        print(f"[SKIP] EIDORS not available: {_import_error}")
        print("Run this script LOCALLY with Octave + EIDORS installed.")
        return

    print("=" * 60)
    print("STAGE 3: EIDORS CONDUCTIVITY JACOBIAN DIRECTIONAL VALIDATION")
    print("=" * 60)
    print("Reference: test_stage3_eidors_conductivity_jacobian.m")
    print()

    backend = FEMBackend(verbose=True)
    print("Building FEM model...")
    backend.build_model()

    centres = backend.get_element_centres()
    N_elem = centres.shape[0]
    print(f"N_elem = {N_elem}")

    theta = sample_reference_geometry()
    sigma0 = compute_sigma(theta, centres)

    print("Computing J_sigma via EIDORS...")
    J_sigma = _fem_jsigma(sigma0, backend)
    print(f"J_sigma shape: {J_sigma.shape}  (expected ({cfg.N_MEAS}, {N_elem}))")
    assert J_sigma.shape[0] == cfg.N_MEAS and J_sigma.shape[1] == N_elem

    # --- Build 6 test directions (matching MATLAB rng(12345)) ---
    rng = np.random.default_rng(12345)
    directions = []

    d = np.ones(N_elem);           d /= np.linalg.norm(d); directions.append(("uniform", d))
    for i in range(3):
        d = rng.standard_normal(N_elem); d /= np.linalg.norm(d); directions.append((f"random_{i}", d))
    center_t = np.array([0.15, 0.05])
    dist = np.sqrt(np.sum((centres - center_t) ** 2, axis=1))
    d = np.exp(-(dist / 0.15) ** 2); d /= np.linalg.norm(d); directions.append(("localized", d))
    d = np.sin(4.0 * centres[:, 0]) * np.cos(3.0 * centres[:, 1])
    d /= np.linalg.norm(d); directions.append(("sinusoidal", d))

    h_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    all_pass = True
    for name, d in directions:
        print(f"\n=== Direction: {name} ===")
        analytic = J_sigma @ d
        print(f"||J_sigma*d|| = {np.linalg.norm(analytic):.6e}")
        min_err = np.inf
        for h in h_list:
            V_plus  = _fem_voltage_from_sigma(sigma0 + h * d, backend)
            V_minus = _fem_voltage_from_sigma(sigma0 - h * d, backend)
            fd = (V_plus - V_minus) / (2.0 * h)
            denom = np.linalg.norm(analytic)
            rel_err = np.linalg.norm(fd - analytic) / max(denom, 1e-14)
            min_err = min(min_err, rel_err)
            print(f"  h = {h:.0e} : relative error = {rel_err:.12e}")
        passed = min_err < 1e-3
        if not passed:
            all_pass = False
        print(f"  Min error = {min_err:.3e}  [{'PASS' if passed else 'FAIL'}]")

    print("\n" + "=" * 60)
    print(f"STAGE 3 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
