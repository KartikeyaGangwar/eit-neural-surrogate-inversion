# Physics-Informed Neural Surrogate Inversion for EIT

Author: Kartikey Singh  
Research guidance: Prof. Debasish Roy

This repository implements a hybrid Physics-Informed Neural Network (PINN)
surrogate framework for solving the Electrical Impedance Tomography (EIT)
inverse problem.

The framework includes:
- finite-difference forward solver
- PINN surrogate model
- polygon parameterization for inverse reconstruction
- hybrid optimization (AdamW + LBFGS)

## Repository structure

src/       – main implementation  
data/      – pre-generated datasets  
models/    – pretrained network weights  
results/   – reconstruction figures
logs/      - training logs 


## Running the code
```bash
python src/eit_pinn_v13.py
```

## Requirements
```
PyTorch  
NumPy  
SciPy  
Matplotlib
```
This repository is currently private while the associated research
manuscript is under preparation.
