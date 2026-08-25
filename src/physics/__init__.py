"""
Physics module for the EIT B-Spline + FEM project.

This module provides the static EIT forward model configuration,
the high-level interfaces for forward and Jacobian solvers, and the
EIDORS-dependent backend for execution.

Submodules:
    fem_model: Defines the FEMModelConfig dataclass.
    fem_backend: Provides the EIDORS/Octave wrapper (EIDORS-dependent).
    fem_forward: High-level forward solve interface.
    fem_jacobian: High-level Jacobian computation interface.
"""
