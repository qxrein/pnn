"""Shared utilities: device selection, seeding, dtype helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import PINNConfig


def resolve_device(request: str = "auto") -> torch.device:
    """Select compute device with macOS Metal (MPS) support.

    Priority for ``auto``: CUDA > MPS > CPU.
    """
    request = request.lower()
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if request == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available on this system.")

    if request == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")

    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    """Map dtype string to ``torch.dtype``."""
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def resolve_training_dtype(name: str, device: torch.device) -> torch.dtype:
    """Select training dtype with MPS compatibility.

    Apple Metal (MPS) has limited float64 support; fall back to float32 on MPS
    while preserving float64 on CPU/CUDA for PDE residual accuracy.
    """
    import warnings

    dtype = resolve_dtype(name)
    if device.type == "mps" and dtype == torch.float64:
        warnings.warn(
            "MPS device detected: using float32 instead of float64 for Metal compatibility.",
            stacklevel=2,
        )
        return torch.float32
    return dtype


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: PINNConfig) -> None:
    """Create output directories from configuration."""
    for attr in ("output_dir", "checkpoint_dir", "figure_dir"):
        Path(getattr(config.paths, attr)).mkdir(parents=True, exist_ok=True)


def to_tensor(
    data: np.ndarray | list[float],
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert array-like data to a tensor on *device*."""
    return torch.as_tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)


def detach_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Move tensor to CPU and return NumPy array."""
    return tensor.detach().cpu().numpy()
