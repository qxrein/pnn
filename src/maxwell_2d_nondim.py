"""First-order 2-D Maxwell PINN in global nondimensional coordinates.

Global nondimensional coordinates
-----------------------------------
    xbar = k0 * x    ∈ [0,  k0*period]
    zbar = k0 * z    ∈ [0,  k0*domain_height]

Maxwell equations in nondimensional form (dEr/dzbar = Hi_x, etc.)
-------------------------------------------------------------------
Physical equations:
    dEr/dz =  k0 * Hi_x                         (Ar)
    dEi/dz = -k0 * Hr_x                         (Ai)
    dEr/dx = -k0 * Hi_z                         (Br)
    dEi/dx =  k0 * Hr_z                         (Bi)
    dHr_x/dz - dHr_z/dx =  k0 * εr * Ei         (Cr)
    dHi_x/dz - dHi_z/dx = -k0 * εr * Er         (Ci)

Since d/dzbar = (1/k0) * d/dz:

    dEr/dzbar =  Hi_x                            (Ar_nd)
    dEi/dzbar = -Hr_x                            (Ai_nd)
    dEr/dxbar = -Hi_z                            (Br_nd)
    dEi/dxbar =  Hr_z                            (Bi_nd)
    dHr_x/dzbar - dHr_z/dxbar =  εr * Ei         (Cr_nd)
    dHi_x/dzbar - dHi_z/dxbar = -εr * Er         (Ci_nd)

No k0 factors appear in the residuals.

Verification against the plane wave E = exp(-izbar), H̃_x = n * E:
    dEr/dzbar = -sin(zbar)
    Hi_x = n * (-sin(zbar))  =>  with n=1: Hi_x = -sin(zbar)
    Residual Ar_nd = -sin - (-sin) = 0  ✓

Network input
--------------
The subnet receives (xbar, zbar) as input.
Features: sin(2^l * xbar), cos(2^l * xbar), sin(2^l * zbar), cos(2^l * zbar).
Level l=0: sin(xbar) and sin(zbar) span the oscillation at k0 exactly.

This is the correct formulation that matches the validated 1-D Maxwell benchmark:
the 1-D benchmark used ξ = k_j*(z-z_lo), which for the air region with k_j=k0
and z_lo=0 reduces to zbar = k0*z.  The 1-D benchmark achieved 0.017% error
with this coordinate choice.

Derivative scaling relation
-----------------------------
Physical derivatives obtained from autograd through xbar, zbar:
    d/dz = k0 * d/dzbar
    d/dx = k0 * d/dxbar

The PDE in physical coords uses k0 explicitly; in nondim coords k0 cancels.
The autograd computes d/dzbar automatically when z_bar is the differentiation
variable. No manual chain-rule correction is needed.

Scattered-field source terms (in nondim)
-----------------------------------------
E_inc(zbar) = exp(-i zbar),   H̃_inc(zbar) = n_air * E_inc
Source in Cr_nd: εr * Ei_scat + (εr-1) * Ei_inc
Source in Ci_nd: -εr * Er_scat - (εr-1) * Er_inc
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# Subdomain network with global nondimensional coordinates
# ---------------------------------------------------------------------------


class Maxwell2DSubdomainMLP_ND(nn.Module):
    """Per-subdomain MLP using global nondimensional coordinates xbar=k0*x, zbar=k0*z.

    Features: sin(2^l * xbar), cos(2^l * xbar), sin(2^l * zbar), cos(2^l * zbar)
    Level 0: sin(xbar), cos(xbar), sin(zbar), cos(zbar) — exactly spans the
    physical oscillation frequency.  This is the key property that makes the
    1-D Maxwell benchmark work.

    Optional grating features: sin(G0*x), cos(G0*x) where G0 = 2π/Λ.
    For Λ=0.8λ, G0/k0=1.25 which falls between l=0 (k0) and l=1 (2k0).
    Adding these features explicitly allows the network to represent ±1
    diffraction orders.

    Outputs: [Er_scat, Ei_scat, Hr_x_scat, Hi_x_scat, Hr_z_scat, Hi_z_scat]
    """

    def __init__(
        self,
        k0: float,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
        period: float | None = None,
        num_grating_levels: int = 0,
    ) -> None:
        super().__init__()
        self.k0 = k0
        self.num_fourier_levels = num_fourier_levels
        self.period = period
        self.num_grating_levels = num_grating_levels

        # Base Fourier features: 4 per level (sin/cos in x and z)
        in_dim = 4 * num_fourier_levels
        # Grating-periodic x features: 2 per level (sin/cos in x only)
        if period is not None and num_grating_levels > 0:
            in_dim += 2 * num_grating_levels

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 6))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _fourier(self, xbar: torch.Tensor, zbar: torch.Tensor) -> torch.Tensor:
        """Fourier features on nondimensional coords xbar=k0*x, zbar=k0*z.

        Base levels: sin/cos(2^l * xbar), sin/cos(2^l * zbar)
        Grating levels: sin/cos(l_g * G0 * x) where G0 = 2*pi/Lambda
          These allow the network to represent m=+-1, +-2, ... diffraction orders
          even when G0 is not a power-of-two multiple of k0.
        """
        parts = []
        for l in range(self.num_fourier_levels):
            s = float(2 ** l)
            parts += [
                torch.sin(s * xbar), torch.cos(s * xbar),
                torch.sin(s * zbar), torch.cos(s * zbar),
            ]
        if self.period is not None and self.num_grating_levels > 0:
            G0_over_k0 = (2.0 * math.pi / self.period) / self.k0
            for l_g in range(1, self.num_grating_levels + 1):
                s = float(l_g) * G0_over_k0
                parts += [
                    torch.sin(s * xbar), torch.cos(s * xbar),
                ]
        return torch.stack(parts, dim=-1)

    def forward_nd(self, xbar: torch.Tensor, zbar: torch.Tensor) -> torch.Tensor:
        """Forward from nondimensional coords. Shape (N, 6). Graph preserved."""
        return self.net(self._fourier(xbar, zbar))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward from physical coords. Shape (N, 6)."""
        return self.forward_nd(x * self.k0, z * self.k0)

    def field_components(self, x, z):
        out = self.forward(x, z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]

    def field_components_nd(self, xbar, zbar):
        """From nondimensional coords, graph preserved for autograd."""
        out = self.forward_nd(xbar, zbar)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


# ---------------------------------------------------------------------------
# Domain-decomposition model with ND subnets
# ---------------------------------------------------------------------------


class Maxwell2DDD_ND(nn.Module):
    """Three-subdomain 2-D Maxwell DD-PINN using nondimensional coordinates."""

    def __init__(
        self,
        physics: "PhysicsConfig",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
        num_grating_levels: int = 0,
    ) -> None:
        super().__init__()
        self.physics = physics
        k0 = physics.k0
        period = physics.period if num_grating_levels > 0 else None
        # All three subnets share the same k0 — same nondimensional coordinates
        self.net_air  = Maxwell2DSubdomainMLP_ND(k0, hidden_layers, hidden_width,
                                                  num_fourier_levels, period, num_grating_levels)
        self.net_grat = Maxwell2DSubdomainMLP_ND(k0, hidden_layers, hidden_width,
                                                  num_fourier_levels, period, num_grating_levels)
        self.net_sub  = Maxwell2DSubdomainMLP_ND(k0, hidden_layers, hidden_width,
                                                  num_fourier_levels, period, num_grating_levels)

    def forward_E(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return [Er_scat, Ei_scat] shape (N, 2), routed by z."""
        out = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
        p = self.physics
        m_air  = z <= p.ridge_z_min
        m_grat = (z > p.ridge_z_min) & (z <= p.ridge_z_max)
        m_sub  = z > p.ridge_z_max
        if m_air.any():
            o = self.net_air.forward(x[m_air], z[m_air]); out[m_air] = o[:, :2]
        if m_grat.any():
            o = self.net_grat.forward(x[m_grat], z[m_grat]); out[m_grat] = o[:, :2]
        if m_sub.any():
            o = self.net_sub.forward(x[m_sub], z[m_sub]); out[m_sub] = o[:, :2]
        return out


# ---------------------------------------------------------------------------
# PDE residuals in nondimensional coordinates
# ---------------------------------------------------------------------------


def _grad_nd(field, wrt, ones):
    g = torch.autograd.grad(field, wrt, grad_outputs=ones,
                             create_graph=True, retain_graph=True, allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(wrt)


def maxwell_2d_nd_pde_residual(
    subnet: Maxwell2DSubdomainMLP_ND,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
    eps_value: float,
    scattered: bool = True,
) -> tuple[torch.Tensor, ...]:
    """First-order Maxwell residuals in nondimensional coordinates.

    Nondimensional equations (xbar=k0*x, zbar=k0*z):
        (Ar)  dEr/dzbar  -  Hi_x = 0
        (Ai)  dEi/dzbar  +  Hr_x = 0
        (Br)  dEr/dxbar  +  Hi_z = 0
        (Bi)  dEi/dxbar  -  Hr_z = 0
        (Cr)  dHr_x/dzbar - dHr_z/dxbar -  eps * Ei = 0  [+ source if scattered]
        (Ci)  dHi_x/dzbar - dHi_z/dxbar +  eps * Er = 0  [+ source if scattered]

    No k0 factors in the residuals.

    Parameters
    ----------
    eps_value :
        Scalar permittivity of this subdomain (constant).
        Use the mean eps within the subdomain or pass the correct value.
    """
    k0 = physics.k0
    xbar = (x * k0).detach().clone().requires_grad_(True)
    zbar = (z * k0).detach().clone().requires_grad_(True)

    Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z = subnet.field_components_nd(xbar, zbar)
    ones = torch.ones_like(Er)

    dEr_dz  = _grad_nd(Er,   zbar, ones); dEi_dz  = _grad_nd(Ei,   zbar, ones)
    dHrx_dz = _grad_nd(Hr_x, zbar, ones); dHix_dz = _grad_nd(Hi_x, zbar, ones)
    dEr_dx  = _grad_nd(Er,   xbar, ones); dEi_dx  = _grad_nd(Ei,   xbar, ones)
    dHrz_dx = _grad_nd(Hr_z, xbar, ones); dHiz_dx = _grad_nd(Hi_z, xbar, ones)

    res = [
        dEr_dz  -  Hi_x,          # Ar
        dEi_dz  +  Hr_x,          # Ai
        dEr_dx  +  Hi_z,          # Br
        dEi_dx  -  Hr_z,          # Bi
        dHrx_dz - dHrz_dx -  eps_value * Ei,   # Cr
        dHix_dz - dHiz_dx +  eps_value * Er,   # Ci
    ]

    if scattered:
        # Source term from permittivity contrast: (εr - 1) * E_inc
        # E_inc in nondim: exp(-i*zbar) => Er_inc=cos(zbar), Ei_inc=-sin(zbar)
        Er_inc = torch.cos(zbar.detach())
        Ei_inc = -torch.sin(zbar.detach())
        delta = eps_value - 1.0
        res[4] = res[4] - delta * Ei_inc   # Cr source: -delta*Ei_inc
        res[5] = res[5] + delta * Er_inc   # Ci source: +delta*Er_inc

    return tuple(res)


def maxwell_2d_nd_pde_loss_subdomain(
    subnet: Maxwell2DSubdomainMLP_ND,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
    eps_value: float,
    scattered: bool = True,
) -> torch.Tensor:
    res = maxwell_2d_nd_pde_residual(subnet, x, z, physics, eps_value, scattered)
    return sum(torch.mean(r**2) for r in res) / len(res)


# ---------------------------------------------------------------------------
# Interface losses (evaluate at physical z_int, autograd through xbar)
# ---------------------------------------------------------------------------


def maxwell_2d_nd_interface_loss(
    net_left: Maxwell2DSubdomainMLP_ND,
    net_right: Maxwell2DSubdomainMLP_ND,
    z_int: float,
    x_pts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E_y and H̃_x continuity at z = z_int."""
    z = torch.full_like(x_pts, z_int)
    Er_l, Ei_l, Hr_l, Hi_l, _, _ = net_left.field_components(x_pts, z)
    Er_r, Ei_r, Hr_r, Hi_r, _, _ = net_right.field_components(x_pts, z)
    L_E = torch.mean((Er_l - Er_r)**2 + (Ei_l - Ei_r)**2)
    L_H = torch.mean((Hr_l - Hr_r)**2 + (Hi_l - Hi_r)**2)
    return L_E, L_H


# ---------------------------------------------------------------------------
# Boundary losses (nondimensional form)
# ---------------------------------------------------------------------------


def maxwell_2d_nd_top_bc(
    net: Maxwell2DSubdomainMLP_ND,
    x_pts: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Scattered-field top BC at z=0 (zbar=0): outgoing upward.

    H̃_x_scat = -n_air * E_scat   (upward-propagating Robin condition)
    """
    z = torch.zeros_like(x_pts)
    Er, Ei, Hr_x, Hi_x, _, _ = net.field_components(x_pts, z)
    n = physics.n_air
    return torch.mean((Hr_x + n * Er)**2 + (Hi_x + n * Ei)**2)


def maxwell_2d_nd_bottom_bc(
    net: Maxwell2DSubdomainMLP_ND,
    x_pts: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Scattered-field bottom BC at z=domain_height: outgoing downward.

    H̃_x_scat = n_sub * E_scat + (n_sub - n_air) * E_inc
    """
    z_bot = physics.domain_height
    z = torch.full_like(x_pts, z_bot)
    Er_s, Ei_s, Hr_x, Hi_x, _, _ = net.field_components(x_pts, z)
    n_sub = physics.n_substrate
    n_air = physics.n_air
    k0    = physics.k0
    Er_inc = torch.cos(k0 * z)
    Ei_inc = -torch.sin(k0 * z)
    Hr_t = n_sub * Er_s + (n_sub - n_air) * Er_inc
    Hi_t = n_sub * Ei_s + (n_sub - n_air) * Ei_inc
    return torch.mean((Hr_x - Hr_t)**2 + (Hi_x - Hi_t)**2)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


def maxwell_2d_nd_total_loss(
    model: Maxwell2DDD_ND,
    pts: dict[str, torch.Tensor],
    physics: "PhysicsConfig",
    w_pde: float = 1.0,
    w_E: float = 500.0,
    w_H: float = 500.0,
    w_top: float = 200.0,
    w_bot: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Compute all losses for the ND DD Maxwell PINN."""
    from src.geometry import epsilon_r

    zero = torch.zeros(1, dtype=torch.float64)
    grat_thick = physics.ridge_z_max - physics.ridge_z_min
    min_thick  = 1e-6

    def _pde(key, subnet, eps_val):
        xk = pts.get(f"x_{key}"); zk = pts.get(f"z_{key}")
        if xk is None or len(xk) == 0:
            return zero
        return maxwell_2d_nd_pde_loss_subdomain(subnet, xk, zk, physics, eps_val)

    L_air  = _pde("air",  model.net_air,  physics.n_air**2)
    L_grat = _pde("grat", model.net_grat, physics.n_ridge**2) if grat_thick > min_thick else zero
    L_sub  = _pde("sub",  model.net_sub,  physics.n_substrate**2)
    L_pde  = L_air + L_grat + L_sub

    if grat_thick > min_thick:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat, physics.ridge_z_min, pts["x_int1"])
        LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,  physics.ridge_z_max, pts["x_int2"])
    else:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air, model.net_sub, physics.ridge_z_min, pts.get("x_int1", pts["x_top"]))
        LE2 = zero; LH2 = zero

    L_top = maxwell_2d_nd_top_bc(model.net_air, pts["x_top"], physics)
    L_bot = maxwell_2d_nd_bottom_bc(model.net_sub, pts["x_bot"], physics)

    total = w_pde*L_pde + w_E*(LE1+LE2) + w_H*(LH1+LH2) + w_top*L_top + w_bot*L_bot
    return {
        "pde": L_pde, "pde_air": L_air, "pde_grat": L_grat, "pde_sub": L_sub,
        "E_int1": LE1, "H_int1": LH1, "E_int2": LE2, "H_int2": LH2,
        "top": L_top, "bottom": L_bot, "total": total,
    }


# ---------------------------------------------------------------------------
# Sampling (reuse from maxwell_2d_dd)
# ---------------------------------------------------------------------------


def sample_nd_points(
    physics: "PhysicsConfig",
    n_per_region: int,
    n_interface: int,
    n_top: int,
    n_bot: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
    margin: float = 5e-3,
) -> dict[str, torch.Tensor]:
    """Sample collocation points for the ND DD model."""
    from src.maxwell_2d_dd import sample_dd_points
    return sample_dd_points(physics, n_per_region, n_interface, n_top, n_bot,
                            device, dtype, seed, margin)
