"""Domain-decomposition 2-D Maxwell PINN for the binary diffraction grating.

Architecture
------------
Three subnetworks, one per z-layer:

    SubNet_air  : z ∈ [0,          ridge_z_min]   uniform air  (εr = n_air²)
    SubNet_grat : z ∈ [ridge_z_min, ridge_z_max]  grating slab (εr varies in x)
    SubNet_sub  : z ∈ [ridge_z_max, domain_height] uniform sub (εr = n_sub²)

Each subnetwork predicts [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z] for its region.

Coordinate normalisation per subdomain
---------------------------------------
Each subnet receives (x_n, z_n) where:
    x_n = 2*x/period - 1         ∈ [-1, 1]  (same for all)
    z_n = 2*(z - z_lo)/h_k - 1  ∈ [-1, 1]  (local to subdomain)

Fourier features are applied to (x_n, z_n).

Interface conditions at z = z_int
-----------------------------------
Scalar TE, μr = 1:
    E_y continuous:   E_y,left = E_y,right
    H̃_x continuous:  H̃_x,left = H̃_x,right   (only x-component is tangential)

Note: H̃_z is the z-component; its tangential component across a horizontal
interface is H̃_z itself (it lies in the plane of the interface). For a
perfectly horizontal interface (normal in ẑ) with TE polarisation:
    Tangential E: E_y is continuous  ✓
    Tangential H: H̃_x is continuous (H_x lies in the interface plane)  ✓
    Normal B:     H̃_z can be discontinuous (no μr contrast, actually continuous)
We enforce both E_y and H̃_x continuity.

Scattered-field formulation
-----------------------------
Network predicts [E_scat, H̃_scat].  The incident plane wave

    E_inc(z) = exp(-ik₀z),  H̃_x_inc(z) = n_air * E_inc(z),  H̃_z_inc = 0

is added to the network output to get the total field.

The scattered-field PDE in each region has the source term:
    source_Cr = k₀*(εr - 1)*Ei_inc
    source_Ci = -k₀*(εr - 1)*Er_inc

For the air region (εr = n_air² = 1): source = 0 → pure outgoing wave.
For the grating layer: source ≠ 0 only inside the ridge (x ∈ [x_lo, x_hi]).
For the substrate (εr = n_sub²): source = k₀*(n_sub²-1)*E_inc ≠ 0.

Scattered-field BCs:
    Top (z=0):    outgoing (upward) → H̃_scat_x = -n_air * E_scat
    Bottom:       outgoing (downward) → H̃_scat_x = +n_sub * E_scat
    Periodic:     E_scat, H̃_scat periodic with grating period (inherited from Fourier)
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
# Subdomain network
# ---------------------------------------------------------------------------


class Maxwell2DSubdomainMLP(nn.Module):
    """Per-subdomain MLP with local z normalisation.

    Outputs [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z].
    """

    def __init__(
        self,
        z_lo: float,
        z_hi: float,
        period: float,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self.z_lo   = z_lo
        self.z_hi   = z_hi
        self.period = period
        self.num_fourier_levels = num_fourier_levels

        in_dim = 4 * num_fourier_levels  # (x_n, z_n) each with sin+cos per level
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

    def _norm(self, x: torch.Tensor, z: torch.Tensor):
        xn = 2.0 * x / self.period - 1.0
        zn = 2.0 * (z - self.z_lo) / (self.z_hi - self.z_lo) - 1.0
        return xn, zn

    def _fourier(self, xn, zn):
        parts = []
        for l in range(self.num_fourier_levels):
            freq = (2.0 ** l) * math.pi
            parts += [torch.sin(freq * xn), torch.cos(freq * xn),
                      torch.sin(freq * zn), torch.cos(freq * zn)]
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xn, zn = self._norm(x, z)
        return self.net(self._fourier(xn, zn))

    def field_components(self, x, z):
        out = self.forward(x, z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


# ---------------------------------------------------------------------------
# Domain-decomposition model
# ---------------------------------------------------------------------------


class Maxwell2DDD(nn.Module):
    """Three-subdomain 2-D Maxwell PINN for the grating."""

    def __init__(
        self,
        physics: "PhysicsConfig",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self.physics = physics
        p = physics
        self.net_air  = Maxwell2DSubdomainMLP(0.0,          p.ridge_z_min,    p.period, hidden_layers, hidden_width, num_fourier_levels)
        self.net_grat = Maxwell2DSubdomainMLP(p.ridge_z_min, p.ridge_z_max,   p.period, hidden_layers, hidden_width, num_fourier_levels)
        self.net_sub  = Maxwell2DSubdomainMLP(p.ridge_z_max, p.domain_height, p.period, hidden_layers, hidden_width, num_fourier_levels)

    def forward_E(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return [Er_scat, Ei_scat] shape (N, 2), routed by z."""
        out = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
        m_air  = z <= self.physics.ridge_z_min
        m_grat = (z > self.physics.ridge_z_min) & (z <= self.physics.ridge_z_max)
        m_sub  = z > self.physics.ridge_z_max
        if m_air.any():
            o = self.net_air.forward(x[m_air], z[m_air])
            out[m_air] = o[:, :2]
        if m_grat.any():
            o = self.net_grat.forward(x[m_grat], z[m_grat])
            out[m_grat] = o[:, :2]
        if m_sub.any():
            o = self.net_sub.forward(x[m_sub], z[m_sub])
            out[m_sub] = o[:, :2]
        return out


# ---------------------------------------------------------------------------
# PDE residuals
# ---------------------------------------------------------------------------


def _g(field, wrt, ones):
    g = torch.autograd.grad(field, wrt, grad_outputs=ones,
                             create_graph=True, retain_graph=True, allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(wrt)


def maxwell_2d_dd_pde_residual(
    subnet: Maxwell2DSubdomainMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
    eps_fn,           # callable: (x_detached, z_detached) -> eps tensor
    scattered: bool = True,
) -> tuple[torch.Tensor, ...]:
    """First-order Maxwell PDE residuals for one subdomain.

    Scattered-field source terms are included when scattered=True.
    """
    x_g = x.detach().clone().requires_grad_(True)
    z_g = z.detach().clone().requires_grad_(True)

    Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z = subnet.field_components(x_g, z_g)
    k0  = physics.k0
    eps = eps_fn(x_g.detach(), z_g.detach())
    ones = torch.ones_like(Er)

    dEr_dz  = _g(Er,   z_g, ones);  dEi_dz  = _g(Ei,   z_g, ones)
    dHrx_dz = _g(Hr_x, z_g, ones);  dHix_dz = _g(Hi_x, z_g, ones)
    dEr_dx  = _g(Er,   x_g, ones);  dEi_dx  = _g(Ei,   x_g, ones)
    dHrz_dx = _g(Hr_z, x_g, ones);  dHiz_dx = _g(Hi_z, x_g, ones)

    res = [dEr_dz  -  k0 * Hi_x,
           dEi_dz  +  k0 * Hr_x,
           dEr_dx  +  k0 * Hi_z,
           dEi_dx  -  k0 * Hr_z,
           dHrx_dz - dHrz_dx - k0 * eps * Ei,
           dHix_dz - dHiz_dx + k0 * eps * Er]

    if scattered:
        # Source term from permittivity contrast
        Er_inc = torch.cos(k0 * z_g.detach())
        Ei_inc = -torch.sin(k0 * z_g.detach())
        delta  = eps - 1.0
        res[4] = res[4] - k0 * delta * Ei_inc
        res[5] = res[5] + k0 * delta * Er_inc

    return tuple(res)


# ---------------------------------------------------------------------------
# Interface losses
# ---------------------------------------------------------------------------


def maxwell_2d_dd_interface_loss(
    net_left: Maxwell2DSubdomainMLP,
    net_right: Maxwell2DSubdomainMLP,
    z_int: float,
    x_pts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """E_y and H̃_x continuity at z = z_int.

    Returns (L_E, L_H).
    """
    z = torch.full_like(x_pts, z_int)
    Er_l, Ei_l, Hr_l, Hi_l, _, _ = net_left.field_components(x_pts, z)
    Er_r, Ei_r, Hr_r, Hi_r, _, _ = net_right.field_components(x_pts, z)
    L_E = torch.mean((Er_l - Er_r)**2 + (Ei_l - Ei_r)**2)
    L_H = torch.mean((Hr_l - Hr_r)**2 + (Hi_l - Hi_r)**2)
    return L_E, L_H


# ---------------------------------------------------------------------------
# Boundary losses
# ---------------------------------------------------------------------------


def maxwell_2d_dd_top_bc(
    net: Maxwell2DSubdomainMLP,
    x_pts: torch.Tensor,
    physics: "PhysicsConfig",
    scattered: bool = True,
) -> torch.Tensor:
    """Top BC at z=0.

    Scattered: outgoing (upward) → H̃_x_scat = -n_air * E_scat
    Total:     E_total = E_inc → E = (1,0), H̃_x = (n_air, 0)
    """
    z = torch.zeros_like(x_pts)
    Er, Ei, Hr_x, Hi_x, _, _ = net.field_components(x_pts, z)
    n = physics.n_air
    if scattered:
        return torch.mean((Hr_x + n * Er)**2 + (Hi_x + n * Ei)**2)
    else:
        return torch.mean((Er - 1.0)**2 + Ei**2 + (Hr_x - n)**2 + Hi_x**2)


def maxwell_2d_dd_bottom_bc(
    net: Maxwell2DSubdomainMLP,
    x_pts: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Bottom BC at z=domain_height: outgoing scattered wave in substrate.

    Derivation (scattered-field):
        H̃_total_x = n_sub * E_total          (pure downward wave in substrate)
        H̃_inc_x   = n_air * E_inc            (incident defined with air impedance)
        H̃_scat_x  = H̃_total - H̃_inc
                   = n_sub*(E_scat + E_inc) - n_air*E_inc
                   = n_sub*E_scat + (n_sub - n_air)*E_inc

    The missing term (n_sub - n_air)*E_inc was previously omitted, causing
    a systematic error of order |n_sub - n_air| ≈ 0.45 at the bottom boundary.
    """
    z = torch.full_like(x_pts, physics.domain_height)
    Er_s, Ei_s, Hr_x, Hi_x, _, _ = net.field_components(x_pts, z)
    n_sub = physics.n_substrate
    n_air = physics.n_air
    k0    = physics.k0

    Er_inc = torch.cos(k0 * z)
    Ei_inc = -torch.sin(k0 * z)

    Hr_target = n_sub * Er_s + (n_sub - n_air) * Er_inc
    Hi_target = n_sub * Ei_s + (n_sub - n_air) * Ei_inc

    return torch.mean((Hr_x - Hr_target)**2 + (Hi_x - Hi_target)**2)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


def maxwell_2d_dd_total_loss(
    model: Maxwell2DDD,
    pts: dict[str, torch.Tensor],
    physics: "PhysicsConfig",
    w_pde: float = 1.0,
    w_E: float = 500.0,
    w_H: float = 500.0,
    w_top: float = 200.0,
    w_bot: float = 100.0,
    scattered: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute all losses for the DD 2-D Maxwell PINN.

    Gracefully handles degenerate geometry (zero-thickness grating or substrate)
    by skipping PDE and interface losses for collapsed subdomains.
    """
    from src.geometry import epsilon_r

    zero = torch.zeros(1, dtype=torch.float64)

    def eps_fn(x, z):
        return epsilon_r(x, z, physics)

    # Minimum subdomain thickness to include PDE loss
    min_thickness = 1e-6

    grat_thickness = physics.ridge_z_max - physics.ridge_z_min
    sub_thickness  = physics.domain_height - physics.ridge_z_max

    def _pde(key, subnet):
        xk = pts.get(f"x_{key}")
        zk = pts.get(f"z_{key}")
        if xk is None or len(xk) == 0:
            return zero
        res = maxwell_2d_dd_pde_residual(subnet, xk, zk, physics, eps_fn, scattered)
        return sum(torch.mean(r**2) for r in res) / len(res)

    L_pde_air  = _pde("air", model.net_air)
    L_pde_grat = _pde("grat", model.net_grat) if grat_thickness > min_thickness else zero
    L_pde_sub  = _pde("sub",  model.net_sub)  if sub_thickness  > min_thickness else zero
    L_pde = L_pde_air + L_pde_grat + L_pde_sub

    # Interface losses — skip if grating layer is degenerate
    if grat_thickness > min_thickness:
        x_int1 = pts["x_int1"]; x_int2 = pts["x_int2"]
        LE1, LH1 = maxwell_2d_dd_interface_loss(model.net_air,  model.net_grat, physics.ridge_z_min, x_int1)
        LE2, LH2 = maxwell_2d_dd_interface_loss(model.net_grat, model.net_sub,  physics.ridge_z_max, x_int2)
    else:
        # No meaningful grating layer: enforce air=sub at the single interface
        x_int1 = pts.get("x_int1", pts["x_top"])
        LE1, LH1 = maxwell_2d_dd_interface_loss(model.net_air, model.net_sub, physics.ridge_z_min, x_int1)
        LE2 = zero; LH2 = zero

    L_top = maxwell_2d_dd_top_bc(model.net_air, pts["x_top"], physics, scattered)
    L_bot = maxwell_2d_dd_bottom_bc(model.net_sub, pts["x_bot"], physics)

    total = (w_pde * L_pde
             + w_E  * (LE1 + LE2)
             + w_H  * (LH1 + LH2)
             + w_top * L_top
             + w_bot * L_bot)

    return {
        "pde": L_pde, "pde_air": L_pde_air, "pde_grat": L_pde_grat, "pde_sub": L_pde_sub,
        "E_int1": LE1, "H_int1": LH1, "E_int2": LE2, "H_int2": LH2,
        "top": L_top, "bottom": L_bot, "total": total,
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_dd_points(
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
    """Sample collocation points for the DD model."""
    rng = np.random.default_rng(seed)

    def _t(a): return torch.as_tensor(a, dtype=dtype, device=device)

    # Interior of each subdomain (excluding interface margins)
    x_air  = rng.uniform(0.0, physics.period, n_per_region)
    z_air  = rng.uniform(margin, max(physics.ridge_z_min - margin, margin * 2), n_per_region)

    x_grat = rng.uniform(0.0, physics.period, n_per_region)
    grat_lo = physics.ridge_z_min + margin
    grat_hi = physics.ridge_z_max - margin
    if grat_lo >= grat_hi:
        # Degenerate grating layer (zero thickness): sample at midpoint
        z_grat = np.full(n_per_region, (physics.ridge_z_min + physics.ridge_z_max) / 2.0)
    else:
        z_grat = rng.uniform(grat_lo, grat_hi, n_per_region)

    x_sub  = rng.uniform(0.0, physics.period, n_per_region)
    sub_lo = physics.ridge_z_max + margin
    sub_hi = physics.domain_height - margin
    if sub_lo >= sub_hi:
        # Degenerate substrate (zero thickness): sample near bottom
        z_sub = np.full(n_per_region, max(physics.ridge_z_max, physics.domain_height * 0.95))
    else:
        z_sub  = rng.uniform(sub_lo, sub_hi, n_per_region)

    # Interface x points
    x_i1 = rng.uniform(0.0, physics.period, n_interface)
    x_i2 = rng.uniform(0.0, physics.period, n_interface)

    # Boundary x points
    x_top = rng.uniform(0.0, physics.period, n_top)
    x_bot = rng.uniform(0.0, physics.period, n_bot)

    return {
        "x_air":  _t(x_air),  "z_air":  _t(z_air),
        "x_grat": _t(x_grat), "z_grat": _t(z_grat),
        "x_sub":  _t(x_sub),  "z_sub":  _t(z_sub),
        "x_int1": _t(x_i1),
        "x_int2": _t(x_i2),
        "x_top":  _t(x_top),
        "x_bot":  _t(x_bot),
    }
