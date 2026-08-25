# Core Scientific Library (`src/`)

This directory contains the core scientific algorithms, physical forward solvers, neural surrogate models, and inverse reconstruction routines.

---

## 1. Package Architecture

```
src/
├── geometry/            # B-spline parameterization, shape libraries, and validity checkers
│   ├── bspline.py       # Closed periodic cubic C^2 B-spline evaluation & curve derivatives
│   ├── conductivity.py  # Piecewise-constant conductivity field generator with tanh mollifier
│   ├── parameters.py    # 64-D parameter vector packing, validity, and self-intersection tests
│   ├── shape_library.py # Canonical shape family generators (circle, ellipse, rectangle, stars, etc.)
│   └── shape_to_bspline.py # Non-linear least-squares fitting of target shapes to B-spline control points
│
├── physics/             # High-fidelity FEM forward solver & adjoint sensitivity computation
│   ├── fem_model.py     # Complete Electrode Model (CEM) physical configuration
│   ├── fem_forward.py   # Forward boundary voltage computation
│   ├── fem_jacobian.py  # Adjoint sensitivity Jacobian matrix computation
│   ├── fem_backend.py   # EIDORS/Octave subprocess interface
│   └── fem_backend_optimized.py # Persistent Octave sessions with in-Octave chain-rule evaluation
│
├── surrogate/           # Neural surrogate architectures and Sobolev training
│   ├── model.py         # VoltageSurrogate ResNet with forward-mode autodiff (torch.func.jacfwd)
│   ├── sobolev_loss.py  # Directional Sobolev loss function with directional JVPs
│   └── training.py      # Multi-stage training engine (AdamW stage + L-BFGS refinement)
│
├── ensemble/            # Deep ensemble container and epistemic uncertainty
│   └── model.py         # EnsembleModel wrapping K=5 models with epistemic spread monitoring
│
├── inversion/           # Zero-online-FEM inverse reconstruction engine
│   ├── problem.py       # Abstract InversionProblem base class
│   ├── lm_solver.py     # Levenberg-Marquardt optimizer with adaptive damping
│   ├── multistart.py    # MultiStart solver wrapper over multiple initial guesses
│   ├── residuals.py     # SurrogateResidualProblem and FEMResidualProblem implementations
│   ├── uncertainty.py   # EnsembleInversionProblem implementation
│   ├── reconstruction.py # Geometric IoU and boundary RMS distance calculation
│   ├── initialization.py # BSplineInitializer generating deterministic/random restarts
│   ├── normalization.py  # VoltageNormalizer utility
│   ├── objective.py     # Objective function and gradient evaluations
│   └── trust_region_solver.py # Alternative bounded trust-region optimization solver
│
├── noise/               # Measurement noise models
│   └── noise_models.py  # Additive Gaussian noise injector with configurable SNR
│
├── dataset/             # Dataset generation and storage utilities
│   ├── sampler.py       # Parameter space sampler across shape families
│   ├── storage.py       # HDF5 dataset writer and streaming dataloader
│   └── generator.py     # High-throughput batch dataset generator
│
├── utils/               # General utilities
│   ├── logger.py        # Structured logging to console and file
│   ├── io_utils.py      # Robust JSON/array serialization helpers
│   └── reproducibility.py # Global seed setting (NumPy, PyTorch, CUDA)
│
└── validation/          # Python vs MATLAB/Octave reference validation suite
    ├── gen_reference_output.py # Generates reference_output.mat
    ├── val_A_conductivity_fd.py # Stage 1 conductivity derivative vs finite differences
    ├── val_B_fem_jsigma.py     # Stage 3 FEM conductivity Jacobian validation
    ├── val_C_geometry_jacobian.py # Stage 4 geometry Jacobian chain-rule validation
    └── val_D_matlab_comparison.py # Stage D full comparison against MATLAB reference
```

---

## 2. Theoretical Formulation

### Forward Problem (Complete Electrode Model)
$$\nabla \cdot (\sigma(\mathbf{x}; \mathbf{\theta}) \nabla u(\mathbf{x})) = 0 \quad \text{in } \Omega$$
$$u + z_l \sigma \frac{\partial u}{\partial n} = U_l \quad \text{on } e_l, \quad l = 1, \dots, 16$$
$$\int_{e_l} \sigma \frac{\partial u}{\partial n} \, ds = I_l, \quad \sum_{l=1}^{16} I_l = 0$$

### Sobolev-Regularized Surrogate Optimization
$$\min_\phi \frac{1}{N} \sum_{i=1}^N \left( \|\mathcal{S}_\phi(\mathbf{\theta}_i) - V_i\|_2^2 + \lambda_J \|\nabla_\mathbf{\theta} \mathcal{S}_\phi(\mathbf{\theta}_i) \mathbf{v}_i - J_i \mathbf{v}_i\|_2^2 \right)$$

### Zero-Online-FEM Inverse Reconstruction
$$\min_\mathbf{\theta} \frac{1}{2} \|\mathcal{S}_\phi(\mathbf{\theta}) - V_{\text{meas}}\|_2^2 + \alpha_{\text{reg}} \mathcal{R}_{\text{curv}}(\mathbf{\theta}) \quad (N_{\text{FEM}}^{\text{online}} = 0)$$
