"""
Reproducibility utilities.
==========================

Sets random seeds across NumPy, PyTorch, and Python random module.

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import random
import numpy as np
import torch

import config as cfg


def set_reproducibility(seed: int = cfg.DEFAULT_SEED) -> None:
    """
    Set random seed across random, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
