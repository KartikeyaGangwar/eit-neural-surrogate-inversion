"""
B-spline geometry module.

Provides:
  - theta_to_control_points : unpack 64-D parameter vector
  - compute_bspline_boundary : evaluate closed cubic B-spline
  - evaluate_bspline_point   : single-point evaluation

All implementations are exact Python translations of the MATLAB reference
polygon_conductivity_bspline.m (lines 33-65).

No EIDORS dependency. Runs on Colab.
"""
