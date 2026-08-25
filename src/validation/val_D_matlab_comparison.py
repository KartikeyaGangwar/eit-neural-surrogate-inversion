"""
Stage: Python vs MATLAB reference comparison.
=============================================

PURPOSE
-------
For identical theta and FEM configuration, compare:
    - sigma (element conductivity)
    - dSigma_dTheta (analytic conductivity derivative)
    - voltage V (FEM forward solve)
    - J_theta (geometry Jacobian)

between the validated MATLAB/EIDORS implementation and this Python implementation.

PREREQUISITES
-------------
1. Run the MATLAB reference script to generate reference_output.mat.
2. Run LOCALLY with Octave + EIDORS installed.

PASS CRITERION
--------------
    sigma relative error        < 1e-10  (pure math)
    dSigma_dTheta relative error < 1e-10  (pure math)
    voltage relative error      < 1e-6   (FEM numerical tolerance)
    J_theta relative error      < 1e-4   (FEM Jacobian composition tolerance)

Run::

    python validation/val_D_matlab_comparison.py
"""

from __future__ import annotations
import sys
import os
import scipy.io as sio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from physics.fem_backend import FEMBackend
    from physics.fem_forward import fem_voltage
    from physics.fem_jacobian import fem_jacobian
    EIDORS_AVAILABLE = True
except ImportError as e:
    EIDORS_AVAILABLE = False
    _import_error = str(e)

from geometry.parameters import control_points_to_theta
from geometry.conductivity import compute_sigma_and_derivative


def report_error(name: str, py_val: np.ndarray, mat_val: np.ndarray, tol: float) -> bool:
    """
    Compare Python and MATLAB numerical arrays and report error metrics.
    
    Args:
        name: Name of tensor/variable being verified.
        py_val: Array computed in Python.
        mat_val: Reference array computed in MATLAB.
        tol: Maximum allowable relative tolerance.
        
    Returns:
        True if relative error < tol.
    """
    py_val = np.asarray(py_val, dtype=float)
    mat_val = np.asarray(mat_val, dtype=float)

    py_flat = py_val.ravel()
    mat_flat = mat_val.ravel()

    assert py_flat.shape == mat_flat.shape, \
        f"Shape mismatch for {name}: Python {py_val.shape} vs MATLAB {mat_val.shape}"

    diff = py_flat - mat_flat
    abs_err = np.abs(diff)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))

    mat_norm = float(np.linalg.norm(mat_flat))
    diff_norm = float(np.linalg.norm(diff))

    rel_err = diff_norm / max(mat_norm, 1e-14)

    print(f"\n--- {name} ---")
    print(f"Mean Abs Error: {mean_abs_err:.5e}")
    print(f"Max Abs Error:  {max_abs_err:.5e}")
    print(f"Norm Error:     {diff_norm:.5e}")
    print(f"Rel Error:      {rel_err:.5e}")

    if rel_err < tol:
        print(f"-> PASS: Rel Error {rel_err:.5e} < {tol}")
        return True
    else:
        print(f"-> FAIL: Rel Error {rel_err:.5e} >= {tol}")
        return False


def main() -> None:
    """
    Run Stage D reference comparison between Python pipeline and MATLAB reference outputs.
    """
    print("=" * 60)
    print("STAGE D: PYTHON vs MATLAB REFERENCE COMPARISON")
    print("=" * 60)

    mat_file = "reference_output.mat"
    if not os.path.exists(mat_file):
        print(f"[SKIP] Reference file '{mat_file}' not found.")
        print("Run the MATLAB reference script first to generate reference_output.mat.")
        return

    if not EIDORS_AVAILABLE:
        print(f"[SKIP] EIDORS not available: {_import_error}")
        print("Run this script LOCALLY with Octave + EIDORS installed.")
        return

    data = sio.loadmat(mat_file)
    theta_mat = np.array(data["theta"]).ravel()
    sigma_mat = np.array(data["sigma"]).ravel()
    dSigma_dTheta_mat = np.array(data["dSigma_dTheta"])
    voltage_mat = np.array(data["voltage"]).ravel()
    J_theta_mat = np.array(data["J_theta"])

    backend = FEMBackend()
    backend.build_model()
    centres = backend.get_element_centres()

    sigma_py, dSigma_dTheta_py = compute_sigma_and_derivative(theta_mat, centres)
    voltage_py = fem_voltage(theta_mat, backend)
    J_theta_py = fem_jacobian(theta_mat, backend)

    pass_sigma  = report_error("sigma",         sigma_py,          sigma_mat,         1e-10)
    pass_dsigma = report_error("dSigma_dTheta", dSigma_dTheta_py,  dSigma_dTheta_mat, 1e-10)
    pass_volt   = report_error("voltage",       voltage_py,        voltage_mat,       1e-6)
    pass_jtheta = report_error("J_theta",       J_theta_py,        J_theta_mat,       1e-4)

    all_pass = pass_sigma and pass_dsigma and pass_volt and pass_jtheta
    print("\n" + "=" * 60)
    print(f"STAGE D RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
