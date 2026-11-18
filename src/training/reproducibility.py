"""Reproducibility utilities, seed controls, and device resolution."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch algorithms to be deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        # Enable deterministic algorithms where possible
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device based on preference and available hardware.

    Args:
        preference: One of "auto", "cuda", "mps", "cpu".

    Returns:
        torch.device instance.
    """
    pref = preference.lower().strip()
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    elif pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")
    elif pref == "cpu":
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {preference}. Use auto, cuda, mps, or cpu.")


def get_system_info() -> dict[str, Any]:
    """Collect runtime environment metadata for audit and reproducibility."""
    info: dict[str, Any] = {
        "python_version": str(sys.version),
        "platform": str(platform.platform()),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info
