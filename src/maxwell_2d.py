"""First-order 2-D Maxwell PINN for the binary diffraction grating (scalar TE).

Physical system
---------------
TE polarisation: E_y field, with H_x and H_z components.
Non-magnetic medium: μr = 1 everywhere.
Convention: exp(+iωt) suppressed.  Propagation convention: E_inc = exp(-ik0 z).

Normalised magnetic field
--------------------------
We define the normalised magnetic field:

    H̃_x = +Z₀ * (-H_x)   (note: negative of H_x_physical)
    H̃_z = +Z₀ * (-H_z)

This sign choice makes the equations consistent with the validated 1-D system
(src/maxwell_1d.py).  Verification against the forward plane wave E_y = exp(-ik0 z):

    H̃_x = +n * E_y   (for forward wave in medium with index n)
    H̃_z = 0          (for z-propagating wave)

2-D TE Maxwell equations (in this convention)
----------------------------------------------
Splitting into real and imaginary parts with E_y = Er + iEi,
H̃_x = Hr_x + iHi_x,  H̃_z = Hr_z + iHi_z:

    (Ar)  ∂Er/∂z =  k0 * Hi_x
    (Ai)  ∂Ei/∂z = -k0 * Hr_x
    (Br)  ∂Er/∂x = -k0 * Hi_z
    (Bi)  ∂Ei/∂x =  k0 * Hr_z
    (Cr)  ∂Hr_x/∂z - ∂Hr_z/∂x =  k0 * εr * Ei
    (Ci)  ∂Hi_x/∂z - ∂Hi_z/∂x = -k0 * εr * Er

Network
-------
One MLP for the full 2-D domain, predicting 6 real outputs:
    [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z]

Inputs are normalised (x, z) coordinates mapped to [-1, 1].
Fourier features encode the known oscillation frequencies.

Boundary and interface conditions
----------------------------------
Top (z=0): incident plane wave E_y = exp(-ik0 z).
    Er(x, 0) = cos(0) = 1,   Ei(x, 0) = -sin(0) = 0  (total field = E_inc)
    Hr_x(x, 0) = n_air * Er(x, 0) = n_air
    Hi_x(x, 0) = n_air * Ei(x, 0) = 0

    Note: for a total-field formulation, E_total = E_inc + E_scat.
    At z=0, E_total = E_inc (assume negligible back-reflection from bottom;
    this is a soft constraint).

Bottom (z=domain_height): outgoing wave in substrate.
    Robin condition: H̃_x = n_sub * E_y  (forward wave only)
    Hr_x = n_sub * Er,  Hi_x = n_sub * Ei

Left/right (x=0, x=period): periodic boundary.
    E, H̃ are periodic with period = grating period.

PDE loss weighting
------------------
All six equations are weighted equally in the PDE loss.
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
# Network
# ---------------------------------------------------------------------------


class Maxwell2DMLP(nn.Module):
    """2-D Maxwell field network: (x, z) → [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z].

    Inputs normalised to [-1, 1].
    Fourier features at multiple frequency levels.
    """

    def __init__(
        self,
        physics: "PhysicsConfig",
        hidden_layers: int = 5,
        hidden_width: int = 128,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self._period = physics.period
        self._domain_height = physics.domain_height
        self._k0 = physics.k0
        self.num_fourier_levels = num_fourier_levels

        in_dim = 4 * num_fourier_levels  # sin+cos for each of x_n, z_n
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 6))  # [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _phys_to_norm(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xn = 2.0 * x / self._period - 1.0
        zn = 2.0 * z / self._domain_height - 1.0
        return xn, zn

    def _fourier(self, xn: torch.Tensor, zn: torch.Tensor) -> torch.Tensor:
        parts = []
        for l in range(self.num_fourier_levels):
            freq = (2.0 ** l) * math.pi
            parts += [
                torch.sin(freq * xn),
                torch.cos(freq * xn),
                torch.sin(freq * zn),
                torch.cos(freq * zn),
            ]
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Shape (N, 6): [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z]."""
        xn, zn = self._phys_to_norm(x, z)
        return self.net(self._fourier(xn, zn))

    def field_components(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return (Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z)."""
        out = self.forward(x, z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------


def maxwell_2d_pde_residual(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> tuple[torch.Tensor, ...]:
    """Compute all six 2-D TE Maxwell residuals efficiently.

    Uses a single backward pass to compute all eight first-order partial
    derivatives simultaneously, reducing autograd overhead by ~8×.

    Equations:
        (Ar)  ∂Er/∂z -  k0 * Hi_x = 0
        (Ai)  ∂Ei/∂z +  k0 * Hr_x = 0
        (Br)  ∂Er/∂x +  k0 * Hi_z = 0
        (Bi)  ∂Ei/∂x -  k0 * Hr_z = 0
        (Cr)  ∂Hr_x/∂z - ∂Hr_z/∂x - k0*εr * Ei = 0
        (Ci)  ∂Hi_x/∂z - ∂Hi_z/∂x + k0*εr * Er = 0
    """
    from src.geometry import epsilon_r

    x_g = x.detach().clone().requires_grad_(True)
    z_g = z.detach().clone().requires_grad_(True)

    Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z = model.field_components(x_g, z_g)
    k0  = physics.k0
    eps = epsilon_r(x_g.detach(), z_g.detach(), physics)

    ones = torch.ones_like(Er)

    # Compute all z-derivatives in one grad call
    dz_vals = torch.autograd.grad(
        [Er, Ei, Hr_x, Hi_x],
        z_g,
        grad_outputs=[ones, ones, ones, ones],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    dEr_dz  = dz_vals[0] if dz_vals[0] is not None else torch.zeros_like(Er)
    dEi_dz  = dz_vals[1] if dz_vals[1] is not None else torch.zeros_like(Ei)
    dHrx_dz = dz_vals[2] if dz_vals[2] is not None else torch.zeros_like(Hr_x)
    dHix_dz = dz_vals[3] if dz_vals[3] is not None else torch.zeros_like(Hi_x)

    # Compute all x-derivatives in one grad call
    dx_vals = torch.autograd.grad(
        [Er, Ei, Hr_z, Hi_z],
        x_g,
        grad_outputs=[ones, ones, ones, ones],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    dEr_dx  = dx_vals[0] if dx_vals[0] is not None else torch.zeros_like(Er)
    dEi_dx  = dx_vals[1] if dx_vals[1] is not None else torch.zeros_like(Ei)
    dHrz_dx = dx_vals[2] if dx_vals[2] is not None else torch.zeros_like(Hr_z)
    dHiz_dx = dx_vals[3] if dx_vals[3] is not None else torch.zeros_like(Hi_z)

    res_Ar = dEr_dz  -  k0 * Hi_x
    res_Ai = dEi_dz  +  k0 * Hr_x
    res_Br = dEr_dx  +  k0 * Hi_z
    res_Bi = dEi_dx  -  k0 * Hr_z
    res_Cr = dHrx_dz - dHrz_dx - k0 * eps * Ei
    res_Ci = dHix_dz - dHiz_dx + k0 * eps * Er

    return res_Ar, res_Ai, res_Br, res_Bi, res_Cr, res_Ci


def maxwell_2d_pde_loss(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Mean squared 2-D Maxwell PDE residual."""
    residuals = maxwell_2d_pde_residual(model, x, z, physics)
    return sum(torch.mean(r**2) for r in residuals) / len(residuals)


# ---------------------------------------------------------------------------
# Boundary and interface losses
# ---------------------------------------------------------------------------


def maxwell_2d_top_bc_loss(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Soft Dirichlet at z=0: total field = incident plane wave.

    E_y(x, 0) = exp(-ik0 * 0) = 1  =>  Er=1, Ei=0
    H̃_x(x, 0) = n_air * E_y(x, 0)  =>  Hr_x=n_air, Hi_x=0
    """
    Er, Ei, Hr_x, Hi_x, _, _ = model.field_components(x, z)
    n_air = physics.n_air
    loss_E = torch.mean((Er - 1.0)**2 + Ei**2)
    loss_H = torch.mean((Hr_x - n_air)**2 + Hi_x**2)
    return loss_E + loss_H


def maxwell_2d_bottom_bc_loss(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Robin outgoing condition at z=domain_height: H̃_x = n_sub * E_y."""
    Er, Ei, Hr_x, Hi_x, _, _ = model.field_components(x, z)
    n_sub = physics.n_substrate
    return torch.mean((Hr_x - n_sub * Er)**2 + (Hi_x - n_sub * Ei)**2)


def maxwell_2d_periodic_bc_loss(
    model: Maxwell2DMLP,
    x_left: torch.Tensor,
    z_per: torch.Tensor,
    x_right: torch.Tensor,
) -> torch.Tensor:
    """Periodic BC: all fields match at x=0 and x=period."""
    out_left  = model.forward(x_left,  z_per)
    out_right = model.forward(x_right, z_per)
    return torch.mean((out_left - out_right)**2)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_collocation_points(
    physics: "PhysicsConfig",
    n_interior: int,
    n_top: int,
    n_bottom: int,
    n_periodic: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
    margin: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Sample all collocation point sets.

    Returns a dict with keys:
        x_int, z_int          interior (away from interfaces)
        x_top, z_top          top boundary (z=0)
        x_bot, z_bot          bottom boundary (z=domain_height)
        x_left, z_per         left periodic boundary (x=0)
        x_right               right periodic boundary (x=period)
    """
    from src.geometry import interface_mask

    rng = np.random.default_rng(seed)

    # --- Interior (exclude interface margins) ---
    x_int_list, z_int_list = [], []
    attempts = 0
    while sum(len(a) for a in x_int_list) < n_interior and attempts < 30:
        xc = rng.uniform(0.0, physics.period, n_interior)
        zc = rng.uniform(0.0, physics.domain_height, n_interior)
        xt = torch.as_tensor(xc, dtype=dtype)
        zt = torch.as_tensor(zc, dtype=dtype)
        keep = ~interface_mask(xt, zt, physics, margin).numpy()
        x_int_list.append(xc[keep])
        z_int_list.append(zc[keep])
        attempts += 1
    x_int = np.concatenate(x_int_list)[:n_interior]
    z_int = np.concatenate(z_int_list)[:n_interior]

    # --- Top boundary (z=0) ---
    x_top = rng.uniform(0.0, physics.period, n_top)
    z_top = np.zeros(n_top)

    # --- Bottom boundary (z=domain_height) ---
    x_bot = rng.uniform(0.0, physics.period, n_bottom)
    z_bot = np.full(n_bottom, physics.domain_height)

    # --- Periodic boundaries ---
    z_per  = rng.uniform(0.0, physics.domain_height, n_periodic)
    x_left  = np.zeros(n_periodic)
    x_right = np.full(n_periodic, physics.period)

    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype, device=device)

    return {
        "x_int":   _t(x_int),
        "z_int":   _t(z_int),
        "x_top":   _t(x_top),
        "z_top":   _t(z_top),
        "x_bot":   _t(x_bot),
        "z_bot":   _t(z_bot),
        "x_left":  _t(x_left),
        "x_right": _t(x_right),
        "z_per":   _t(z_per),
    }


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


def maxwell_2d_total_loss(
    model: Maxwell2DMLP,
    pts: dict[str, torch.Tensor],
    physics: "PhysicsConfig",
    w_pde: float = 1.0,
    w_top: float = 100.0,
    w_bot: float = 50.0,
    w_per: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Compute all Maxwell 2-D loss terms.

    Returns dict with 'pde', 'top', 'bottom', 'periodic', 'total'.
    """
    L_pde = maxwell_2d_pde_loss(model, pts["x_int"], pts["z_int"], physics)
    L_top = maxwell_2d_top_bc_loss(model, pts["x_top"], pts["z_top"], physics)
    L_bot = maxwell_2d_bottom_bc_loss(model, pts["x_bot"], pts["z_bot"], physics)
    L_per = maxwell_2d_periodic_bc_loss(model, pts["x_left"], pts["z_per"], pts["x_right"])
    total = w_pde * L_pde + w_top * L_top + w_bot * L_bot + w_per * L_per
    return {
        "pde": L_pde, "top": L_top, "bottom": L_bot,
        "periodic": L_per, "total": total,
    }


# ---------------------------------------------------------------------------
# Incident field reference (for diagnostics)
# ---------------------------------------------------------------------------


def incident_field_2d(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> tuple[torch.Tensor, torch.Tensor]:
    """E_inc(x, z) = exp(-ik0 z): Re = cos(k0 z), Im = -sin(k0 z)."""
    phase = physics.k0 * z
    return torch.cos(phase), -torch.sin(phase)
