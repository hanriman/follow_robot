"""
Hardware device resolution utility for wm-control.
Supports auto-detection with priority: CUDA -> MPS (Apple Silicon) -> CPU.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(device_str: str | None = None) -> torch.device:
    """
    Resolve PyTorch compute device.

    Args:
        device_str: Explicit device identifier (e.g., 'cuda', 'mps', 'cpu', 'cuda:0').
                    If provided, verifies availability and fails loud if not present.

    Returns:
        torch.device instance.
    """
    if device_str is not None:
        device_str = device_str.strip().lower()
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested device '{device_str}' but CUDA is not available on this system.")
            return torch.device(device_str)
        elif device_str == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Requested device 'mps' but Apple Silicon Metal is not available.")
            return torch.device("mps")
        elif device_str == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device specification '{device_str}'. Expected 'cuda', 'mps', or 'cpu'.")

    # Auto-detection hierarchy: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device
