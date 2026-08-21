"""Boundary condition losses for the grating PINN.

Convention
----------
Time dependence exp(+iωt) suppressed.  Wave propagates in **+z** (downward).

    E_inc(z) = exp(-i k0 z)  =>  Re = cos(k0 z),  Im = -sin(k0 z)

Top boundary (z = 0)
    Total-field formulation: E_total = E_inc at z=0 (soft Dirichlet).
    At z=0: E_inc = 1 + 0i.

    Scattered-field formulation: E_scat = 0 at z=0 (soft Dirichlet for
    the scattered component only).

Bottom boundary (z = domain_height)
    Approximate first-order absorbing (Sommerfeld-like):
        ∂E/∂z + i k_sub E ≈ 0  (Robin condition)
    In the current prototype this is enforced as a soft penalty.
    The substrate wavenumber is k_sub = k0 * n_substrate.

    NOT a rigorous ABC — documented limitation.

Periodic side boundaries (x = 0, x = period)
    Enforce E(0, z) = E(period, z) for both real and imaginary parts.

Propagation-sign note
---------------------
The sign ``-i k0 z`` in E_inc is consistent with the exp(+iωt) convention
and ``∂²E/∂z² + k0² E = 0`` being satisfied by exp(∓ i k0 z).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.physics import incident_field

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# Boundary targets
# ---------------------------------------------------------------------------


def top_target(z: torch.Tensor, physics: "PhysicsConfig") -> tuple[torch.Tensor, torch.Tensor]:
    """Incident-field target at the top boundary.

    For total-field formulation: E_total(z=0) = E_inc(z=0).
    Since z ≈ 0 on the top boundary, E_inc ≈ 1 + 0i.
    Returns (target_real, target_imag).
    """
    return incident_field(z, physics)


def bottom_target_robin(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """First-order absorbing (Robin) condition at the bottom boundary.

    Penalty:  ∂E/∂z + i k_sub E  should be ≈ 0 for a downward-propagating wave.

    Splitting into real/imaginary::

        Re: ∂E_real/∂z - k_sub E_imag  = 0
        Im: ∂E_imag/∂z + k_sub E_real  = 0

    This is a mild approximation valid for normal incidence and a locally
    uniform substrate.  It is NOT a rigorous absorbing boundary condition.
    Documented limitation.
    """
    from src.derivatives import first_derivative

    k_sub = physics.k0 * physics.n_substrate
    xr = x.detach().clone().requires_grad_(True)
    zr = z.detach().clone().requires_grad_(True)
    e_real, e_imag = model.field_components(xr, zr)
    dEr_dz = first_derivative(e_real, zr)
    dEi_dz = first_derivative(e_imag, zr)
    res_real = dEr_dz - k_sub * e_imag
    res_imag = dEi_dz + k_sub * e_real
    return torch.mean(res_real**2 + res_imag**2)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def periodic_boundary_loss(
    model: nn.Module,
    x_left: torch.Tensor,
    z_left: torch.Tensor,
    x_right: torch.Tensor,
    z_right: torch.Tensor,
) -> torch.Tensor:
    """Enforce E(x=0, z) = E(x=period, z) for real and imaginary parts."""
    e_l = model(x_left, z_left)
    e_r = model(x_right, z_right)
    return torch.mean((e_l - e_r) ** 2)


def top_boundary_loss(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Soft Dirichlet BC: E_total ≈ E_inc at top boundary (z = 0).

    At z=0 the target is cos(0), -sin(0) = 1, 0.
    The z values passed here should all be zero (or very close to zero).
    """
    e_pred = model(x, z)
    e_real_t, e_imag_t = top_target(z, physics)
    target = torch.stack([e_real_t, e_imag_t], dim=-1)
    return torch.mean((e_pred - target) ** 2)


def bottom_boundary_loss(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Approximate absorbing BC at the bottom boundary.

    Uses the Robin (first-order absorbing) condition::

        ∂E/∂z + i k_sub E = 0

    which is more physically meaningful than pinning to a specific target
    wave, since the transmitted amplitude is unknown a priori.

    Falls back to a soft Dirichlet targeting a downward-propagating wave
    in the substrate only when use_robin=False (for gradient stability).

    Current implementation: Robin penalty.
    """
    return bottom_target_robin(model, x, z, physics)


def interface_continuity_loss(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Optional field magnitude regulariser near material interfaces.

    For a single continuous network representing the total field, explicit
    interface matching is implicit.  This loss acts as a mild regulariser
    when enabled (loss_weights.interface > 0).
    """
    e = model(x, z)
    return torch.mean(e**2)


# ---------------------------------------------------------------------------
# Scattered-field boundary conditions
# ---------------------------------------------------------------------------


def top_boundary_loss_scattered(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Scattered-field formulation: E_scat = 0 at top boundary.

    In the scattered-field formulation the network predicts E_scat, and at
    the top boundary E_scat = 0 (incident wave carries the BC).
    """
    e_pred = model(x, z)
    return torch.mean(e_pred**2)


def bottom_boundary_loss_scattered(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Outgoing-wave Robin condition for the scattered field at the bottom.

    Same Robin penalty as the total-field version but for E_scat.
    """
    return bottom_target_robin(model, x, z, physics)


# ---------------------------------------------------------------------------
# Boundary diagnostic utilities
# ---------------------------------------------------------------------------


def boundary_diagnostic(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
    boundary: str,
) -> dict[str, torch.Tensor]:
    """Return target and prediction tensors for diagnostic plotting.

    Parameters
    ----------
    boundary : ``'top'`` or ``'bottom'``

    Returns
    -------
    dict with keys: x, z, pred_real, pred_imag, target_real, target_imag
    """
    with torch.no_grad():
        e_pred = model(x, z)
        pred_real = e_pred[:, 0]
        pred_imag = e_pred[:, 1]

    if boundary == "top":
        target_real, target_imag = top_target(z, physics)
    elif boundary == "bottom":
        k_sub = physics.k0 * physics.n_substrate
        target_real = torch.cos(k_sub * z)
        target_imag = -torch.sin(k_sub * z)
    else:
        raise ValueError(f"Unknown boundary: {boundary!r}. Use 'top' or 'bottom'.")

    return {
        "x": x,
        "z": z,
        "pred_real": pred_real,
        "pred_imag": pred_imag,
        "target_real": target_real,
        "target_imag": target_imag,
    }
