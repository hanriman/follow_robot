"""
Deterministic random seed configuration.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_torch: bool = False) -> None:
    """
    Set deterministic seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic_torch: Whether to enforce deterministic CUDA/CUDNN algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
