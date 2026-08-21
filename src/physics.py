"""Helmholtz PDE residual for scalar TE/TM formulation.

Convention
----------
Time dependence: exp(+i ω t) suppressed throughout.
Incident wave propagates in **+z** (downward):  E_inc(z) = exp(-i k0 z).
This gives Re{E_inc} = cos(k0 z), Im{E_inc} = -sin(k0 z).

Nondimensional coordinates
--------------------------
The network receives nondimensional inputs::

    x_tilde = k0 * x
    z_tilde = k0 * z

The Helmholtz equation in nondimensional form is::

    ∂²E/∂x̃² + ∂²E/∂z̃² + εr E = 0

Because d/dx = k0 * d/dx̃, the physical Laplacian becomes::

    ∂²E/∂x² + ∂²E/∂z² = k0² (∂²E/∂x̃² + ∂²E/∂z̃²)

so the nondimensional PDE residual is::

    ∂²E/∂x̃² + ∂²E/∂z̃² + εr E = 0

No additional k0 factor is needed when autograd derivatives are taken
with respect to x̃ and z̃ directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.derivatives import field_laplacian, prepare_coords
from src.geometry import epsilon_r

if TYPE_CHECKING:
    from src.config import PhysicsConfig


def helmholtz_residual(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Helmholtz residuals for real and imaginary parts.

    The model receives nondimensional coordinates x̃ = k0*x, z̃ = k0*z
    (handled inside FieldMLP).  Autograd derivatives are taken with respect
    to these nondimensional inputs, so the residual is::

        ∂²E/∂x̃² + ∂²E/∂z̃² + εr E = 0

    Parameters
    ----------
    x, z :
        Physical coordinates (not nondimensional).  The model internally
        converts to nondimensional before computing the forward pass, but
        we differentiate through the whole graph.
    physics :
        Physical configuration.

    Returns
    -------
    res_real, res_imag : tensors of shape (N,)
    """
    # Convert to nondimensional; differentiate w.r.t. these
    k0 = physics.k0
    x_tilde = x * k0
    z_tilde = z * k0
    x_tilde, z_tilde = prepare_coords(x_tilde, z_tilde)

    # Forward pass through model using nondimensional coords directly
    e_real, e_imag = model.field_components_nd(x_tilde, z_tilde)

    _, _, _, _, lap_real, lap_imag = field_laplacian(e_real, e_imag, x_tilde, z_tilde)

    # epsilon_r still takes physical coords
    # x_phys = x_tilde / k0 — but we need the original physical x for eps lookup
    # We pass x and z (physical) for eps evaluation
    eps = epsilon_r(x, z, physics)

    res_real = lap_real + eps * e_real
    res_imag = lap_imag + eps * e_imag
    return res_real, res_imag


def physics_loss(
    model: nn.Module,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    """Mean squared Helmholtz residual over interior collocation points."""
    res_real, res_imag = helmholtz_residual(model, x, z, physics)
    return torch.mean(res_real**2 + res_imag**2)


def incident_field(z: torch.Tensor, physics: PhysicsConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Incident plane wave E_inc(z) = exp(-i k0 z).

    Convention: exp(+iωt) suppressed; wave propagates in +z.

        Re{E_inc}(z) = cos(k0 z)
        Im{E_inc}(z) = -sin(k0 z)

    At z=0: E_inc = 1 + 0i.  At z=λ: E_inc = 1 + 0i (one full cycle).
    """
    phase = physics.k0 * z
    return torch.cos(phase), -torch.sin(phase)


def scattered_source_term(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source term for scattered-field formulation.

    For E_total = E_inc + E_scat, the scattered field satisfies::

        ΔE_scat + k0² εr E_scat = -k0² (εr - 1) E_inc

    which in nondimensional form (∂/∂x̃) becomes::

        ΔE_scat + εr E_scat = -(εr - 1) E_inc

    Returns the RHS (source_real, source_imag).  Note sign: the source
    drives E_scat; the residual for scattered field is::

        lap_scat + eps * E_scat + (eps - 1) * E_inc = 0
    """
    eps = epsilon_r(x, z, physics)
    e_inc_real, e_inc_imag = incident_field(z, physics)
    delta_eps = eps - 1.0
    src_real = delta_eps * e_inc_real
    src_imag = delta_eps * e_inc_imag
    return src_real, src_imag
