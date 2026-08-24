"""Feature-frequency ablation for the 2-D Maxwell DD-PINN.

Four feature variants tested under identical conditions:

A. NORMALIZED
   Input: xn = 2*x/period - 1,  zn = 2*(z-zlo)/h - 1  ∈ [-1, 1]
   Level-0 sin(pi*xn), cos(pi*xn), sin(pi*zn), cos(pi*zn)
   Physical frequency: 2*pi/period, 2*pi/h
   Grating slab: 2*pi/0.2 = 31.4 rad/λ  (3.33× above k2=9.42)

B. GLOBAL_K0
   Input: xbar = k0*x,  zbar = k0*z
   Level-0 sin(xbar), cos(xbar), sin(zbar), cos(zbar)
   Physical frequency: k0 = 6.28 rad/λ
   Grating slab: k0 = 6.28  (0.67× below k2=9.42)

C. LOCAL_MATERIAL_K
   Input z:  xi_z = k_j*(z - z_lo_j)   (k_j = k0*n_j per medium)
   Input x:  xbar = k0*x  (global k0 for x)
   Level-0 sin(xbar), cos(xbar), sin(xi_z), cos(xi_z)
   Physical z frequency: k_j  — EXACT match to medium oscillation
   Grating slab: k2=9.42 EXACT

D. LOCAL_MATERIAL_PLUS_GRATING_X
   Same as C for z, but for x uses grating-periodic xi_x = 2*pi*x/Lambda
   Level-0 sin(xi_x)=sin(k_grating*x): encodes grating periodicity directly

PDE derivatives
---------------
All variants compute PDE residuals w.r.t. global xbar and zbar.
The autograd chain rule is applied correctly in all cases via the physical
(x, z) differentiation path in maxwell_2d_nd_pde_residual.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PhysicsConfig


FEATURE_VARIANTS = ("normalized", "global_k0", "local_material_k", "local_material_plus_grating_x")


class SubdomainMLP_Variant(nn.Module):
    """Per-subdomain MLP with configurable feature encoding.

    All variants output [Er_scat, Ei_scat, Hr_x_scat, Hi_x_scat, Hr_z_scat, Hi_z_scat].
    PDE residuals are always computed via physical (x, z) autograd.
    """

    def __init__(
        self,
        z_lo: float,
        z_hi: float,
        k_j: float,      # wavenumber in this medium
        period: float,
        k0: float,
        variant: str = "global_k0",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        if variant not in FEATURE_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {FEATURE_VARIANTS}")
        self.z_lo    = z_lo
        self.z_hi    = z_hi
        self.h       = z_hi - z_lo
        self.k_j     = k_j
        self.k0      = k0
        self.period  = period
        self.variant = variant
        self.num_fourier_levels = num_fourier_levels

        in_dim = 4 * num_fourier_levels  # sin+cos for (feat_x, feat_z)
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

    def _features(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute Fourier features from physical (x, z)."""
        v = self.variant

        if v == "normalized":
            # xn ∈ [-1,1], zn ∈ [-1,1]
            feat_x = 2.0 * x / self.period - 1.0
            feat_z = 2.0 * (z - self.z_lo) / self.h - 1.0
            freq_scale = math.pi   # sin(pi * feat)

        elif v == "global_k0":
            # xbar = k0*x, zbar = k0*z
            feat_x = x * self.k0
            feat_z = z * self.k0
            freq_scale = 1.0       # sin(1 * feat)

        elif v == "local_material_k":
            # xi_x = k0*x (global), xi_z = k_j*(z-z_lo)
            feat_x = x * self.k0
            feat_z = self.k_j * (z - self.z_lo)
            freq_scale = 1.0

        elif v == "local_material_plus_grating_x":
            # xi_x = 2*pi*x/period (grating-periodic), xi_z = k_j*(z-z_lo)
            feat_x = 2.0 * math.pi * x / self.period
            feat_z = self.k_j * (z - self.z_lo)
            freq_scale = 1.0

        parts = []
        for l in range(self.num_fourier_levels):
            s = freq_scale * (2.0 ** l)
            parts += [torch.sin(s * feat_x), torch.cos(s * feat_x),
                      torch.sin(s * feat_z), torch.cos(s * feat_z)]
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(self._features(x, z))

    def field_components(self, x, z):
        out = self.forward(x, z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


class Maxwell2DDD_Variant(nn.Module):
    """Three-subdomain 2-D Maxwell DD-PINN with configurable feature encoding."""

    def __init__(
        self,
        physics: "PhysicsConfig",
        variant: str = "global_k0",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self.physics = physics
        self.variant = variant
        k0 = physics.k0
        k1 = k0 * physics.n_air
        k2 = k0 * physics.n_ridge
        k3 = k0 * physics.n_substrate
        kw = dict(variant=variant, hidden_layers=hidden_layers,
                  hidden_width=hidden_width, num_fourier_levels=num_fourier_levels)
        self.net_air  = SubdomainMLP_Variant(0.0,          physics.ridge_z_min,    k1, physics.period, k0, **kw)
        self.net_grat = SubdomainMLP_Variant(physics.ridge_z_min, physics.ridge_z_max, k2, physics.period, k0, **kw)
        self.net_sub  = SubdomainMLP_Variant(physics.ridge_z_max, physics.domain_height, k3, physics.period, k0, **kw)

    def forward_E(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
