# Pretrained Surrogate Models & Checkpoints

This directory contains the frozen production surrogate model checkpoints used for all inverse reconstructions, physical FEM validations, noise robustness benchmarks, epistemic uncertainty quantification, and publication figures.

---

## 1. Production Model Checkpoints

### 1.1 Baseline Deterministic Surrogate Checkpoint
- **Location:** `models/surrogate_10k_final.pt`
- **Architecture:** 6 ResNet blocks, hidden dimension 512, SiLU activations ($1,053,376$ parameters)
- **Training Protocol:** Two-stage optimization (Stage 1: AdamW 20K steps with Cosine Annealing; Stage 2: L-BFGS refinement with strong Wolfe line search)
- **Size:** 16.03 MB (16,806,207 bytes)
- **SHA-256 Checksum:** `8301aa04adf710fcfd8ad12ae12e67dc4c623a576b7d6061bc0e4cd6042753a5`

### 1.2 Deep Ensemble Checkpoints ($K=5$ Independent Models)
Located in `models/deep_ensemble/`:

| Member | Initialization Seed | Size | SHA-256 Checksum |
| :--- | :---: | :---: | :--- |
| **Member #1** | 42 | 16.03 MB | `8301aa04adf710fcfd8ad12ae12e67dc4c623a576b7d6061bc0e4cd6042753a5` |
| **Member #2** | 142 | 16.01 MB | `bbbf3d996ce8f624388809e76e6a87438508ef2daf46adf7b3d88d0d11969e5e` |
| **Member #3** | 242 | 16.01 MB | `64606da26052705094b01ced47664831088ead244d371ac7c5a3832be4ece4aa` |
| **Member #4** | 342 | 16.01 MB | `d48187001bfcd0a83a782027e6ec50be7e80c57d4230a5584c433a8f39060dd0` |
| **Member #5** | 442 | 16.01 MB | `8829e4dcacd18b817513720d1659e25850f0f0c1697966539ea392333d10a8c8` |

---

## 2. Architecture & Normalization Specifications

- **Input Dimension:** 64 ($32 \times (X_m, Y_m)$ closed cubic B-spline control points $\mathbf{\theta} \in \mathbb{R}^{64}$ in meters)
- **Output Dimension:** 1,920 ($120 \text{ pairwise current excitation patterns} \times 16 \text{ electrode potentials}$)
- **Network Depth:** 6 residual blocks with skip connections and LayerNorm
- **Activation:** Continuous $\text{SiLU}(x) = x \cdot \sigma(x)$ (ensuring $C^\infty$ differentiability for analytical forward-mode autodiff Jacobian computation)
- **Sobolev Supervision:** Directional Jacobian-vector product (JVP) loss with $\lambda_J = 0.01\,\text{m}^2$

---

## 3. Loading Models in Python

### Baseline Model
```python
import torch
from src.surrogate.model import VoltageSurrogate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoltageSurrogate().to(device)

checkpoint = torch.load("models/surrogate_10k_final.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
if "v_mean" in checkpoint and "v_std" in checkpoint:
    model.set_normalization_stats(checkpoint["v_mean"], checkpoint["v_std"])
model.eval()
```

### Deep Ensemble Container
```python
import torch
from pathlib import Path
from src.surrogate.model import VoltageSurrogate
from src.ensemble.model import EnsembleModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [42, 142, 242, 342, 442]
members = []

for idx, seed in enumerate(seeds):
    ckpt_path = Path("models/deep_ensemble") / f"ensemble_member_{idx}_seed{seed}.pt"
    m = VoltageSurrogate().to(device)
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c)
    if "v_mean" in c and "v_std" in c:
        m.set_normalization_stats(c["v_mean"], c["v_std"])
    m.eval()
    members.append(m)

ensemble = EnsembleModel(members)
ensemble.eval()
```
