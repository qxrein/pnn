"""Geometry and relative permittivity map for the binary grating."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import PhysicsConfig


def normalize_coordinates(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map physical coordinates to [-1, 1] for network input normalization.

    Coordinate convention
    -------------------
    - ``x`` in [0, period]: periodic horizontal direction.
    - ``z`` in [0, domain_height]: vertical direction, **z=0 is top** (incident
      boundary), **z increases downward** toward the substrate.
    """
    x_norm = 2.0 * x / physics.period - 1.0
    z_norm = 2.0 * z / physics.domain_height - 1.0
    return x_norm, z_norm


def epsilon_r(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    """Piecewise-constant relative permittivity εr(x, z).

    Regions (priority: ridge > substrate > air):
    - **Air** (εr = n_air²): above the substrate top surface, outside the ridge.
    - **Ridge** (εr = n_ridge²): rectangular dielectric protrusion.
    - **Substrate** (εr = n_substrate²): z >= ridge_base_z.

    Parameters
    ----------
    x, z:
        Coordinate tensors of matching shape.
    physics:
        Physical configuration.
    """
    eps = torch.full_like(x, physics.eps_air)

    in_substrate = z >= physics.ridge_base_z
    eps = torch.where(in_substrate, torch.full_like(x, physics.eps_substrate), eps)

    in_ridge = (
        (x >= physics.ridge_x_min)
        & (x <= physics.ridge_x_max)
        & (z >= physics.ridge_z_min)
        & (z <= physics.ridge_z_max)
    )
    eps = torch.where(in_ridge, torch.full_like(x, physics.eps_ridge), eps)
    return eps


def epsilon_r_grid(physics: PhysicsConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build visualization grid and permittivity map.

    Returns
    -------
    x_grid, z_grid, eps_grid : 2D meshgrid tensors (nz, nx).
    """
    nx = physics.nx_visualization
    nz = physics.nz_visualization
    x = torch.linspace(0.0, physics.period, nx, device=device, dtype=dtype)
    z = torch.linspace(0.0, physics.domain_height, nz, device=device, dtype=dtype)
    x_grid, z_grid = torch.meshgrid(x, z, indexing="xy")
    eps_grid = epsilon_r(x_grid, z_grid, physics)
    return x_grid, z_grid, eps_grid


def interface_mask(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
    margin: float | None = None,
) -> torch.Tensor:
    """Boolean mask for points near material interfaces.

    Used to optionally sample interface collocation points or exclude
    interior points from PDE residual evaluation.
    """
    if margin is None:
        margin = physics.interface_margin

    ridge_x_min = physics.ridge_x_min
    ridge_x_max = physics.ridge_x_max
    ridge_z_min = physics.ridge_z_min
    ridge_z_max = physics.ridge_z_max
    z_sub = physics.ridge_base_z

    near = torch.zeros_like(x, dtype=torch.bool)
    # Substrate-air horizontal interface
    near |= torch.abs(z - z_sub) <= margin
    # Ridge vertical faces
    near |= (
        (z >= ridge_z_min)
        & (z <= ridge_z_max)
        & (
            (torch.abs(x - ridge_x_min) <= margin)
            | (torch.abs(x - ridge_x_max) <= margin)
        )
    )
    # Ridge top
    near |= (
        (x >= ridge_x_min)
        & (x <= ridge_x_max)
        & (torch.abs(z - ridge_z_max) <= margin)
    )
    return near
