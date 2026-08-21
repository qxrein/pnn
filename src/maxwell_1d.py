"""First-order Maxwell PINN for the 1-D layered medium.

First-order Maxwell system (scalar TE, μr = 1, exp(+iωt) suppressed)
----------------------------------------------------------------------
For a z-propagating field in a non-magnetic medium:

    dE_y / dz = -iωμ₀ H_x  = -i(k₀/c)μ₀ (c/μ₀) H_x
              = -i k₀ (μ₀ c) H_x

Using normalised fields and the standard relation
    H_x = -(1/iωμ₀) dE_y/dz  for a plane wave,
the coupled system in each uniform region is:

    dE_y/dz  = -i k₀ μr  H̃_x          ... (1)
    dH̃_x/dz  = -i k₀ εr  E_y           ... (2)

where H̃_x = Z₀ H_x  (Z₀ = 377 Ω, normalised impedance).

Since μr = 1 everywhere:
    dE_y/dz  = -i k₀ H̃_x               ... (1)
    dH̃_x/dz  = -i k₀ εr E_y            ... (2)

Convention verification against the plane-wave solution
--------------------------------------------------------
For a downward plane wave E_y = exp(-ik_j z), H̃_x = n_j exp(-ik_j z):

    dE_y/dz  = -ik_j exp(-ik_j z)
             = -i(k₀ n_j) exp(-ik_j z)
             = -ik₀ · [n_j exp(-ik_j z)]
             = -ik₀ H̃_x                 ✓  (equation 1 satisfied)

    dH̃_x/dz = -ik_j n_j exp(-ik_j z)
             = -i(k₀ n_j) n_j exp(-ik_j z)
             = -ik₀ n_j² E_y
             = -ik₀ εr E_y              ✓  (equation 2 satisfied)

Splitting into real and imaginary parts
----------------------------------------
Write E_y = E_r + iE_i,  H̃_x = H_r + iH_i.

Equation (1):  dE_r/dz + i dE_i/dz = -i k₀ (H_r + iH_i)
               dE_r/dz =  k₀ H_i
               dE_i/dz = -k₀ H_r

Equation (2):  dH_r/dz + i dH_i/dz = -i k₀ εr (E_r + iE_i)
               dH_r/dz =  k₀ εr E_i
               dH_i/dz = -k₀ εr E_r

Interface conditions (μr = 1 everywhere)
-----------------------------------------
At z = z_int:
    E_y continuous:  E_r, E_i match
    H̃_x continuous:  H_r, H_i match  (since μr is equal on both sides)

Boundary conditions
--------------------
Top (z = 0):
    E_y(0) = 1 + r     (total field)
    H̃_x(0) = 1 - r     (incident contributes +n_air, reflected contributes -n_air;
                          with n_air = 1: H̃_x = 1 - r)

Bottom (z = domain_height):
    First-order radiation: only forward wave in substrate.
    Forward plane wave satisfies H̃_x = n_sub * E_y.
    Condition: H_r = n_sub * E_r,  H_i = n_sub * E_i.

Network architecture
---------------------
One MLP per subdomain predicting [E_r, E_i, H_r, H_i].
Same ξ_j = k_j*(z - z_lo) coordinate as the Helmholtz DD-PINN.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.benchmarks import LayeredMediumBenchmark


# ---------------------------------------------------------------------------
# Subdomain network (4 outputs: E_r, E_i, H_r, H_i)
# ---------------------------------------------------------------------------


class MaxwellSubdomainMLP(nn.Module):
    """Single-subdomain MLP predicting (E_real, E_imag, H_real, H_imag).

    Uses ξ_j = k_j*(z - z_lo) as input coordinate.
    Fourier features: sin(2^l * ξ), cos(2^l * ξ) for l = 0..L-1.

    At level 0, sin(ξ) and cos(ξ) span the physical oscillation frequency.
    """

    def __init__(
        self,
        z_lo: float,
        z_hi: float,
        k_j: float,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.k_j  = k_j
        self.num_fourier_levels = num_fourier_levels

        in_dim = 2 * num_fourier_levels
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 4))  # E_r, E_i, H_r, H_i
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _xi(self, z: torch.Tensor) -> torch.Tensor:
        return self.k_j * (z - self.z_lo)

    def _fourier(self, xi: torch.Tensor) -> torch.Tensor:
        parts = []
        for l in range(self.num_fourier_levels):
            s = float(2 ** l)
            parts += [torch.sin(s * xi), torch.cos(s * xi)]
        return torch.stack(parts, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Shape (N, 4): [E_r, E_i, H_r, H_i]."""
        return self.net(self._fourier(self._xi(z)))

    def field_EH(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (E_r, E_i, H_r, H_i)."""
        out = self.forward(z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ---------------------------------------------------------------------------
# Domain-decomposition first-order Maxwell model
# ---------------------------------------------------------------------------


class Maxwell1DLayered(nn.Module):
    """Three-subdomain first-order Maxwell PINN for the layered medium."""

    def __init__(
        self,
        bm: "LayeredMediumBenchmark",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self.bm = bm
        k1 = bm.k0 * bm.n_air
        k2 = bm.k0 * bm.n_slab
        k3 = bm.k0 * bm.n_sub
        self.net_air  = MaxwellSubdomainMLP(0.0,          bm.z_slab_top,    k1, hidden_layers, hidden_width, num_fourier_levels)
        self.net_slab = MaxwellSubdomainMLP(bm.z_slab_top, bm.z_slab_bot,   k2, hidden_layers, hidden_width, num_fourier_levels)
        self.net_sub  = MaxwellSubdomainMLP(bm.z_slab_bot, bm.domain_height, k3, hidden_layers, hidden_width, num_fourier_levels)

    def forward_E(self, z: torch.Tensor) -> torch.Tensor:
        """Return E field only, shape (N, 2): [E_r, E_i]."""
        out = torch.zeros(z.shape[0], 2, dtype=z.dtype, device=z.device)
        m_air  = z <= self.bm.z_slab_top
        m_slab = (z > self.bm.z_slab_top) & (z <= self.bm.z_slab_bot)
        m_sub  = z > self.bm.z_slab_bot
        if m_air.any():  out[m_air]  = self.net_air.forward(z[m_air])[:, :2]
        if m_slab.any(): out[m_slab] = self.net_slab.forward(z[m_slab])[:, :2]
        if m_sub.any():  out[m_sub]  = self.net_sub.forward(z[m_sub])[:, :2]
        return out


# ---------------------------------------------------------------------------
# PDE residuals (first-order)
# ---------------------------------------------------------------------------


def maxwell_pde_residual(
    subnet: MaxwellSubdomainMLP,
    z_interior: torch.Tensor,
    k0: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-order Maxwell residuals in physical z coordinates.

    Equations:
        dE_r/dz -  k0 * H_i = 0
        dE_i/dz +  k0 * H_r = 0
        dH_r/dz -  k0*eps * E_i = 0
        dH_i/dz +  k0*eps * E_r = 0

    Returns res_Er, res_Ei, res_Hr, res_Hi  (shapes: (N,) each).
    """
    from src.derivatives import first_derivative

    z = z_interior.detach().clone().requires_grad_(True)
    er, ei, hr, hi = subnet.field_EH(z)

    dEr_dz = first_derivative(er, z)
    dEi_dz = first_derivative(ei, z)
    dHr_dz = first_derivative(hr, z)
    dHi_dz = first_derivative(hi, z)

    res_Er = dEr_dz - k0 * hi
    res_Ei = dEi_dz + k0 * hr
    res_Hr = dHr_dz - k0 * eps * ei
    res_Hi = dHi_dz + k0 * eps * er
    return res_Er, res_Ei, res_Hr, res_Hi


# ---------------------------------------------------------------------------
# Interface losses
# ---------------------------------------------------------------------------


def maxwell_interface_loss(
    net_left: MaxwellSubdomainMLP,
    net_right: MaxwellSubdomainMLP,
    z_int: float,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuity of E and H at z = z_int.

    Returns L_E (field), L_H (magnetic) separately.
    """
    z = torch.full((n_points,), z_int, dtype=dtype, device=device)
    er_l, ei_l, hr_l, hi_l = net_left.field_EH(z)
    er_r, ei_r, hr_r, hi_r = net_right.field_EH(z)
    L_E = torch.mean((er_l - er_r)**2 + (ei_l - ei_r)**2)
    L_H = torch.mean((hr_l - hr_r)**2 + (hi_l - hi_r)**2)
    return L_E, L_H


# ---------------------------------------------------------------------------
# Boundary losses
# ---------------------------------------------------------------------------


def maxwell_top_bc(
    net: MaxwellSubdomainMLP,
    z_top: float,
    r: complex,
    n_air: float,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Top BC (z=0): E_y = 1+r,  H̃_x = n_air*(1-r)/n_air = 1-r.

    For the total field with n_air = 1:
        E_y(0) = 1 + r
        H̃_x(0) = 1 - r   (incident: H̃=n_air*E, reflected: H̃=-n_air*E, n_air=1)
    """
    z = torch.full((n_points,), z_top, dtype=dtype, device=device)
    er, ei, hr, hi = net.field_EH(z)
    E_tgt_r = float(1.0 + r.real)
    E_tgt_i = float(r.imag)
    H_tgt_r = float(1.0 - r.real)   # H_inc - H_ref = (1) - (-r) = 1-r ... sign:
    # Incident: E=exp(-ikz), H̃=n_air*exp(-ikz). Reflected: E=r*exp(+ikz), H̃=-n_air*r*exp(+ikz)
    # At z=0: H̃ = n_air*(1) + (-n_air*r) = n_air*(1-r). With n_air=1: H̃=1-r.
    H_tgt_i = float(-r.imag)
    loss_E = torch.mean((er - E_tgt_r)**2 + (ei - E_tgt_i)**2)
    loss_H = torch.mean((hr - H_tgt_r)**2 + (hi - H_tgt_i)**2)
    return loss_E + loss_H


def maxwell_bottom_bc(
    net: MaxwellSubdomainMLP,
    z_bot: float,
    n_sub: float,
    n_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Bottom BC: forward wave only => H̃_x = n_sub * E_y.

    Splitting: H_r = n_sub * E_r,  H_i = n_sub * E_i.
    """
    z = torch.full((n_points,), z_bot, dtype=dtype, device=device)
    er, ei, hr, hi = net.field_EH(z)
    loss_r = torch.mean((hr - n_sub * er)**2)
    loss_i = torch.mean((hi - n_sub * ei)**2)
    return loss_r + loss_i


# ---------------------------------------------------------------------------
# Analytical H field for verification
# ---------------------------------------------------------------------------


def analytical_H_np(bm: "LayeredMediumBenchmark", z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytical normalised magnetic field H̃_x = Z₀ H_x.

    H̃_x = n_j * (forward - backward) for TE waves:
        Air:   H̃ = n_air*(exp(-ik1*z) - r*exp(+ik1*z))
        Slab:  H̃ = n_slab*(A*exp(-ik2*z') - B*exp(+ik2*z')) ... actually use k2/k0
        Sub:   H̃ = n_sub * t * exp(-ik3*z'')

    Derivation: dE/dz = -ik₀ H̃  =>  H̃ = (i/k₀) dE/dz
    """
    c = bm._tmm_coefficients()
    r, t, A, B = c["r"], c["t"], c["A"], c["B"]
    k1, k2, k3 = c["k1"], c["k2"], c["k3"]
    k0 = bm.k0

    H = np.zeros_like(z, dtype=complex)
    air  = z <= bm.z_slab_top
    slab = (z > bm.z_slab_top) & (z <= bm.z_slab_bot)
    sub  = z > bm.z_slab_bot

    # dE_air/dz = -ik1*exp(-ik1*z) + ik1*r*exp(+ik1*z)
    # H̃ = (i/k0)*dE/dz = (i/k0)*(-ik1*E_inc + ik1*r*E_ref) = (k1/k0)*(E_inc - r*E_ref)
    E_inc = np.exp(-1j*k1*z[air]); E_ref = np.exp(+1j*k1*z[air])
    H[air] = (k1/k0) * (E_inc - r * E_ref)

    zp = z[slab] - bm.z_slab_top
    E_fwd = A*np.exp(-1j*k2*zp); E_bwd = B*np.exp(+1j*k2*zp)
    H[slab] = (k2/k0) * (E_fwd - E_bwd)

    zpp = z[sub] - bm.z_slab_bot
    H[sub] = (k3/k0) * t * np.exp(-1j*k3*zpp)

    return np.real(H), np.imag(H)
