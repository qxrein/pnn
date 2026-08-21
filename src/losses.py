"""Composite loss assembly for PINN training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.boundary_conditions import (
    bottom_boundary_loss,
    interface_continuity_loss,
    periodic_boundary_loss,
    top_boundary_loss,
)
from src.physics import physics_loss
from src.sampling import SampleBatch

if TYPE_CHECKING:
    from src.config import LossWeights, PhysicsConfig


def optional_data_loss(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    e_real_ref: torch.Tensor,
    e_imag_ref: torch.Tensor,
) -> torch.Tensor:
    """Supervised data loss against reference field samples."""
    e_pred = model(x, z)
    target = torch.stack([e_real_ref, e_imag_ref], dim=-1)
    return torch.mean((e_pred - target) ** 2)


def compute_losses(
    model: nn.Module,
    samples: SampleBatch,
    physics: PhysicsConfig,
    weights: LossWeights,
    data_points: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all loss terms and weighted total.

    Returns a dictionary with keys:
    ``pde``, ``periodic``, ``top``, ``bottom``, ``interface``, ``data``,
    ``total``, and weighted variants ``weighted_*``.
    """
    losses: dict[str, torch.Tensor] = {}

    losses["pde"] = physics_loss(model, samples.interior_x, samples.interior_z, physics)
    losses["periodic"] = periodic_boundary_loss(
        model,
        samples.periodic_left_x,
        samples.periodic_left_z,
        samples.periodic_right_x,
        samples.periodic_right_z,
    )
    losses["top"] = top_boundary_loss(model, samples.top_x, samples.top_z, physics)
    losses["bottom"] = bottom_boundary_loss(model, samples.bottom_x, samples.bottom_z, physics)

    if samples.interface_x.numel() > 0:
        losses["interface"] = interface_continuity_loss(
            model, samples.interface_x, samples.interface_z
        )
    else:
        losses["interface"] = torch.tensor(0.0, device=losses["pde"].device, dtype=losses["pde"].dtype)

    if data_points is not None and weights.data > 0:
        x_d, z_d, e_r, e_i = data_points
        losses["data"] = optional_data_loss(model, x_d, z_d, e_r, e_i)
    else:
        losses["data"] = torch.tensor(0.0, device=losses["pde"].device, dtype=losses["pde"].dtype)

    total = (
        weights.pde * losses["pde"]
        + weights.periodic * losses["periodic"]
        + weights.top * losses["top"]
        + weights.bottom * losses["bottom"]
        + weights.interface * losses["interface"]
        + weights.data * losses["data"]
    )
    losses["total"] = total

    for key in ("pde", "periodic", "top", "bottom", "interface", "data"):
        w = getattr(weights, key)
        losses[f"weighted_{key}"] = w * losses[key]

    return losses
