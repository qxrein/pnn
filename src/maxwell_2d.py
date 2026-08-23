"""First-order 2-D Maxwell PINN for the binary diffraction grating (scalar TE).

Physical system
---------------
TE polarisation: E_y field, H̃_x and H̃_z components (normalised magnetic fields).
Non-magnetic medium: μr = 1 everywhere.
Convention: exp(+iωt) suppressed.  E_inc(z) = exp(-ik0 z).

Normalised magnetic field convention
--------------------------------------
H̃_x is defined so that the validated 1-D Maxwell equations hold:
    dEr/dz =  k0 * Hi_x
    dEi/dz = -k0 * Hr_x

This is consistent with H̃_x = +n * E_y for a forward wave (verified).

2-D TE Maxwell equations (real/imag split)
-------------------------------------------
    (Ar)  ∂Er/∂z =  k0 * Hi_x
    (Ai)  ∂Ei/∂z = -k0 * Hr_x
    (Br)  ∂Er/∂x = -k0 * Hi_z
    (Bi)  ∂Ei/∂x =  k0 * Hr_z
    (Cr)  ∂Hr_x/∂z - ∂Hr_z/∂x =  k0 * εr * Ei
    (Ci)  ∂Hi_x/∂z - ∂Hi_z/∂x = -k0 * εr * Er

Two formulations
-----------------
TOTAL-FIELD (TF):
    Network predicts the complete field [E_y, H̃_x, H̃_z].
    Top BC: E(z=0) = E_inc(0) = 1 + 0i.
    Problem: this suppresses reflected diffraction orders at the top, which
    is physically wrong for a grating that reflects light.

SCATTERED-FIELD (SF):
    E_total = E_inc + E_scat,  H̃_total = H̃_inc + H̃_scat.
    Network predicts only [E_scat, H̃_scat].
    The incident field satisfies Maxwell exactly (with εr=1 everywhere),
    so the scattered-field PDE has a source term from the permittivity contrast:
        ∂E_scat_r/∂z =  k0 * Hi_scat_x
        ∂E_scat_i/∂z = -k0 * Hr_scat_x
        ...
        ∂Hr_scat_x/∂z - ∂Hr_scat_z/∂x =  k0*εr*Ei_scat + k0*(εr-1)*Ei_inc
        ∂Hi_scat_x/∂z - ∂Hi_scat_z/∂x = -k0*εr*Er_scat - k0*(εr-1)*Er_inc

    Scattered-field BCs (which correctly allow reflection and transmission):
        Top (z=0):    E_scat = 0 + outgoing Robin condition  (scattered wave
                      enters from top and propagates upward — radiation BC)
        Bottom:       H̃_scat_x = n_sub * E_scat  (outgoing into substrate)
        Periodic:     E_scat and H̃_scat periodic with grating period

    The SF formulation is the standard approach for grating problems.
    It naturally admits all reflected and transmitted orders.

Sampling strategy
-----------------
Region-wise sampling (in addition to uniform) for the grating problem:
    - Air region (above grating)
    - Ridge region (inside grating ridge)
    - Substrate (below grating)
    - Near-interface band (within 2× interface_margin)
    - Top boundary
    - Bottom boundary
    - Periodic side boundaries
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
    """2-D Maxwell field network: (x, z) → 6 real outputs.

    Total-field mode:     [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z]
    Scattered-field mode: [Er_scat, Ei_scat, Hr_x_scat, Hi_x_scat, Hr_z_scat, Hi_z_scat]

    The mode is determined by how the losses are computed; the network
    architecture is identical.
    """

    def __init__(
        self,
        physics: "PhysicsConfig",
        hidden_layers: int = 4,
        hidden_width: int = 64,
        num_fourier_levels: int = 4,
    ) -> None:
        super().__init__()
        self._period = physics.period
        self._domain_height = physics.domain_height
        self._k0 = physics.k0
        self.num_fourier_levels = num_fourier_levels

        in_dim = 4 * num_fourier_levels  # sin+cos for (x_n, z_n)
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

    def _phys_to_norm(self, x, z):
        return 2.0 * x / self._period - 1.0, 2.0 * z / self._domain_height - 1.0

    def _fourier(self, xn, zn):
        parts = []
        for l in range(self.num_fourier_levels):
            freq = (2.0 ** l) * math.pi
            parts += [torch.sin(freq * xn), torch.cos(freq * xn),
                      torch.sin(freq * zn), torch.cos(freq * zn)]
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Shape (N, 6)."""
        xn, zn = self._phys_to_norm(x, z)
        return self.net(self._fourier(xn, zn))

    def field_components(self, x, z):
        out = self.forward(x, z)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5]


# ---------------------------------------------------------------------------
# Incident field (needed for scattered-field formulation)
# ---------------------------------------------------------------------------


def incident_E_H(z: torch.Tensor, physics: "PhysicsConfig") -> tuple[torch.Tensor, ...]:
    """Incident plane wave E_inc = exp(-ik0 z) and H̃_inc = n_air * E_inc.

    Returns (Er_inc, Ei_inc, Hr_x_inc, Hi_x_inc) — all shape (N,).
    Hr_z_inc = Hi_z_inc = 0 for a z-propagating wave.
    """
    k0, n = physics.k0, physics.n_air
    Er = torch.cos(k0 * z)
    Ei = -torch.sin(k0 * z)
    return Er, Ei, n * Er, n * Ei


# ---------------------------------------------------------------------------
# PDE residuals
# ---------------------------------------------------------------------------


def _grad(field, wrt, ones):
    g = torch.autograd.grad(field, wrt, grad_outputs=ones,
                             create_graph=True, retain_graph=True,
                             allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(wrt)


def maxwell_2d_pde_residual(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> tuple[torch.Tensor, ...]:
    """Total-field 2-D TE Maxwell residuals (6 equations).

    Returns res_Ar, res_Ai, res_Br, res_Bi, res_Cr, res_Ci each shape (N,).
    """
    from src.geometry import epsilon_r

    x_g = x.detach().clone().requires_grad_(True)
    z_g = z.detach().clone().requires_grad_(True)
    Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z = model.field_components(x_g, z_g)
    k0  = physics.k0
    eps = epsilon_r(x_g.detach(), z_g.detach(), physics)
    ones = torch.ones_like(Er)

    dEr_dz  = _grad(Er,   z_g, ones);  dEi_dz  = _grad(Ei,   z_g, ones)
    dHrx_dz = _grad(Hr_x, z_g, ones);  dHix_dz = _grad(Hi_x, z_g, ones)
    dEr_dx  = _grad(Er,   x_g, ones);  dEi_dx  = _grad(Ei,   x_g, ones)
    dHrz_dx = _grad(Hr_z, x_g, ones);  dHiz_dx = _grad(Hi_z, x_g, ones)

    return (dEr_dz  -  k0 * Hi_x,
            dEi_dz  +  k0 * Hr_x,
            dEr_dx  +  k0 * Hi_z,
            dEi_dx  -  k0 * Hr_z,
            dHrx_dz - dHrz_dx - k0 * eps * Ei,
            dHix_dz - dHiz_dx + k0 * eps * Er)


def maxwell_2d_scattered_pde_residual(
    model: Maxwell2DMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> tuple[torch.Tensor, ...]:
    """Scattered-field 2-D TE Maxwell residuals.

    The network predicts [E_scat, H̃_scat].
    The incident field satisfies Maxwell with εr=1; the contrast
    (εr - 1) drives the scattered field:

        ∂Er_s/∂z  -  k0 * Hi_s_x = 0
        ∂Ei_s/∂z  +  k0 * Hr_s_x = 0
        ∂Er_s/∂x  +  k0 * Hi_s_z = 0
        ∂Ei_s/∂x  -  k0 * Hr_s_z = 0
        ∂Hr_s_x/∂z - ∂Hr_s_z/∂x - k0*εr*Ei_s = k0*(εr-1)*Ei_inc
        ∂Hi_s_x/∂z - ∂Hi_s_z/∂x + k0*εr*Er_s = -k0*(εr-1)*Er_inc

    Moving the source to the left:
        res_Cr = ∂Hr_s_x/∂z - ∂Hr_s_z/∂x - k0*εr*Ei_s - k0*(εr-1)*Ei_inc
        res_Ci = ∂Hi_s_x/∂z - ∂Hi_s_z/∂x + k0*εr*Er_s + k0*(εr-1)*Er_inc
    """
    from src.geometry import epsilon_r

    x_g = x.detach().clone().requires_grad_(True)
    z_g = z.detach().clone().requires_grad_(True)
    Er_s, Ei_s, Hr_sx, Hi_sx, Hr_sz, Hi_sz = model.field_components(x_g, z_g)
    k0  = physics.k0
    eps = epsilon_r(x_g.detach(), z_g.detach(), physics)
    ones = torch.ones_like(Er_s)

    # Incident field at these points (no graph needed)
    Er_inc, Ei_inc, _, _ = incident_E_H(z_g.detach(), physics)

    dEr_dz  = _grad(Er_s, z_g, ones);  dEi_dz  = _grad(Ei_s, z_g, ones)
    dHrx_dz = _grad(Hr_sx, z_g, ones); dHix_dz = _grad(Hi_sx, z_g, ones)
    dEr_dx  = _grad(Er_s, x_g, ones);  dEi_dx  = _grad(Ei_s, x_g, ones)
    dHrz_dx = _grad(Hr_sz, x_g, ones); dHiz_dx = _grad(Hi_sz, x_g, ones)

    delta_eps = eps - 1.0   # = 0 in air, nonzero inside dielectric

    return (dEr_dz  -  k0 * Hi_sx,
            dEi_dz  +  k0 * Hr_sx,
            dEr_dx  +  k0 * Hi_sz,
            dEi_dx  -  k0 * Hr_sz,
            dHrx_dz - dHrz_dx - k0 * eps * Ei_s - k0 * delta_eps * Ei_inc,
            dHix_dz - dHiz_dx + k0 * eps * Er_s + k0 * delta_eps * Er_inc)


def maxwell_2d_pde_loss(model, x, z, physics, scattered=False):
    """Mean squared PDE residual (average over 6 equations)."""
    fn = maxwell_2d_scattered_pde_residual if scattered else maxwell_2d_pde_residual
    residuals = fn(model, x, z, physics)
    return sum(torch.mean(r**2) for r in residuals) / len(residuals)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def maxwell_2d_top_bc_loss(model, x, z, physics, scattered=False):
    """Top BC at z=0.

    Total-field:   E(x,0) = E_inc(0) = (1, 0),  H̃_x = (n_air, 0).
                   WRONG for grating: suppresses reflected orders.

    Scattered-field (correct for grating):
                   E_scat(x,0): Robin radiation BC — outgoing scattered wave
                   propagates upward.  For a plane wave decomposition, this is
                   H̃_scat_x = -n_air * E_scat  (upward-propagating convention).
                   The minus sign: upward (reflected) wave has H̃ = -n*E.
    """
    Er, Ei, Hr_x, Hi_x, _, _ = model.field_components(x, z)
    if scattered:
        # Robin outgoing (upward): H̃_scat_x = -n_air * E_scat
        n = physics.n_air
        return torch.mean((Hr_x + n * Er)**2 + (Hi_x + n * Ei)**2)
    else:
        # Soft Dirichlet: E_total = E_inc at z=0 (suppresses reflection — incorrect for gratings)
        n = physics.n_air
        loss_E = torch.mean((Er - 1.0)**2 + Ei**2)
        loss_H = torch.mean((Hr_x - n)**2 + Hi_x**2)
        return loss_E + loss_H


def maxwell_2d_bottom_bc_loss(model, x, z, physics, scattered=False):
    """Bottom BC at z=domain_height: outgoing (downward) wave.

    Downward-propagating: H̃_x = +n_sub * E_y.
    """
    Er, Ei, Hr_x, Hi_x, _, _ = model.field_components(x, z)
    n_sub = physics.n_substrate
    return torch.mean((Hr_x - n_sub * Er)**2 + (Hi_x - n_sub * Ei)**2)


def maxwell_2d_periodic_bc_loss(model, x_left, z_per, x_right):
    """Periodic BC: field at x=0 equals field at x=period."""
    out_left  = model.forward(x_left,  z_per)
    out_right = model.forward(x_right, z_per)
    return torch.mean((out_left - out_right)**2)


# ---------------------------------------------------------------------------
# Region-wise sampling
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
    n_interface: int = 0,
) -> dict[str, torch.Tensor]:
    """Sample collocation points with optional interface-region oversampling.

    Interior sampling uses region-wise allocation so the small grating ridge
    (≈4% of domain area) gets proportional coverage:
        50% uniform across full domain (interface-margin excluded)
        25% from ridge region
        25% from near-interface band

    Parameters
    ----------
    n_interface :
        Extra points near the grating ridge boundary.  0 = disabled.
    """
    from src.geometry import interface_mask

    rng = np.random.default_rng(seed)
    physics_margin = margin

    def _t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    # ---- Interior ----
    n_uniform   = n_interior // 2
    n_ridge     = n_interior // 4
    n_near_intf = n_interior - n_uniform - n_ridge

    # Uniform (excluding margins)
    xu_list, zu_list = [], []
    attempts = 0
    while sum(len(a) for a in xu_list) < n_uniform and attempts < 30:
        xc = rng.uniform(0.0, physics.period, n_uniform)
        zc = rng.uniform(0.0, physics.domain_height, n_uniform)
        xt = torch.as_tensor(xc, dtype=dtype)
        zt = torch.as_tensor(zc, dtype=dtype)
        keep = ~interface_mask(xt, zt, physics, physics_margin).numpy()
        xu_list.append(xc[keep]); zu_list.append(zc[keep])
        attempts += 1
    x_uni = np.concatenate(xu_list)[:n_uniform]
    z_uni = np.concatenate(zu_list)[:n_uniform]

    # Ridge region — only sample if ridge has non-zero area
    ridge_area = (physics.ridge_x_max - physics.ridge_x_min) * (physics.ridge_z_max - physics.ridge_z_min)
    if ridge_area > (2 * physics_margin) ** 2:
        x_ridge = rng.uniform(physics.ridge_x_min + physics_margin,
                              physics.ridge_x_max - physics_margin, n_ridge)
        z_ridge = rng.uniform(physics.ridge_z_min + physics_margin,
                              physics.ridge_z_max - physics_margin, n_ridge)
    else:
        # No ridge (homogeneous/layered case): use uniform sampling for this quota
        x_ridge = rng.uniform(0.0, physics.period, n_ridge)
        z_ridge = rng.uniform(0.0, physics.domain_height, n_ridge)

    # Near-interface band
    xi_list, zi_list = [], []
    if n_near_intf > 0:
        attempts = 0
        while sum(len(a) for a in xi_list) < n_near_intf and attempts < 30:
            xc = rng.uniform(0.0, physics.period, n_near_intf * 4)
            zc = rng.uniform(0.0, physics.domain_height, n_near_intf * 4)
            xt = torch.as_tensor(xc, dtype=dtype)
            zt = torch.as_tensor(zc, dtype=dtype)
            near = interface_mask(xt, zt, physics, physics_margin * 5).numpy()
            not_too_close = ~interface_mask(xt, zt, physics, physics_margin).numpy()
            keep = near & not_too_close
            xi_list.append(xc[keep]); zi_list.append(zc[keep])
            attempts += 1
        x_ni = np.concatenate(xi_list)[:n_near_intf]
        z_ni = np.concatenate(zi_list)[:n_near_intf]
    else:
        x_ni, z_ni = np.empty(0), np.empty(0)

    x_int = np.concatenate([x_uni, x_ridge, x_ni])[:n_interior]
    z_int = np.concatenate([z_uni, z_ridge, z_ni])[:n_interior]

    # ---- Boundaries ----
    x_top = rng.uniform(0.0, physics.period, n_top)
    z_top = np.zeros(n_top)
    x_bot = rng.uniform(0.0, physics.period, n_bottom)
    z_bot = np.full(n_bottom, physics.domain_height)
    z_per  = rng.uniform(0.0, physics.domain_height, n_periodic)
    x_left  = np.zeros(n_periodic)
    x_right = np.full(n_periodic, physics.period)

    result = {
        "x_int": _t(x_int), "z_int": _t(z_int),
        "x_top": _t(x_top), "z_top": _t(z_top),
        "x_bot": _t(x_bot), "z_bot": _t(z_bot),
        "x_left": _t(x_left), "x_right": _t(x_right), "z_per": _t(z_per),
    }

    # ---- Optional extra interface points ----
    if n_interface > 0:
        xi2, zi2 = [], []
        attempts = 0
        while sum(len(a) for a in xi2) < n_interface and attempts < 30:
            xc = rng.uniform(0.0, physics.period, n_interface * 4)
            zc = rng.uniform(0.0, physics.domain_height, n_interface * 4)
            xt = torch.as_tensor(xc, dtype=dtype)
            zt = torch.as_tensor(zc, dtype=dtype)
            near = interface_mask(xt, zt, physics, physics_margin * 2).numpy()
            xi2.append(xc[near]); zi2.append(zc[near])
            attempts += 1
        result["x_intf"] = _t(np.concatenate(xi2)[:n_interface])
        result["z_intf"] = _t(np.concatenate(zi2)[:n_interface])

    return result


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
    scattered: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute all Maxwell 2-D loss terms.

    Returns dict with 'pde', 'top', 'bottom', 'periodic', 'total'.
    """
    L_pde = maxwell_2d_pde_loss(model, pts["x_int"], pts["z_int"], physics, scattered)
    L_top = maxwell_2d_top_bc_loss(model, pts["x_top"], pts["z_top"], physics, scattered)
    L_bot = maxwell_2d_bottom_bc_loss(model, pts["x_bot"], pts["z_bot"], physics, scattered)
    L_per = maxwell_2d_periodic_bc_loss(model, pts["x_left"], pts["z_per"], pts["x_right"])
    total = w_pde * L_pde + w_top * L_top + w_bot * L_bot + w_per * L_per
    return {"pde": L_pde, "top": L_top, "bottom": L_bot, "periodic": L_per, "total": total}


def evaluate_losses_from_checkpoint(
    ckpt_path: str,
    pts: dict[str, torch.Tensor],
    physics: "PhysicsConfig",
    w_pde: float,
    w_top: float,
    w_bot: float,
    w_per: float,
    scattered: bool = False,
) -> dict[str, float]:
    """Load checkpoint and evaluate losses on given collocation points.

    Used for checkpoint consistency verification: the reloaded model
    must produce the same loss as the in-memory model at save time.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    dev  = torch.device("cpu")
    dt   = torch.float64

    # Infer architecture from checkpoint's model_state_dict rather than config
    # (CLI may have overridden hidden_layers/width vs config defaults)
    state = ckpt["model_state_dict"]
    # Count linear layers to infer depth; first linear: net.0.weight shape (width, in_dim)
    first_w = state["net.0.weight"]
    in_dim = int(first_w.shape[1])
    hidden_width_ckpt = int(first_w.shape[0])
    # Count hidden layers: every other key in net is a linear layer
    linear_keys = [k for k in state if k.endswith(".weight")]
    hidden_layers_ckpt = len(linear_keys) - 1  # subtract output layer
    num_fourier = ckpt.get("num_fourier_levels", in_dim // 4)

    model = Maxwell2DMLP(
        cfg.physics,
        hidden_layers=hidden_layers_ckpt,
        hidden_width=hidden_width_ckpt,
        num_fourier_levels=num_fourier,
    ).to(device=dev, dtype=dt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Move pts to cpu/float64
    pts_cpu = {k: v.to(device=dev, dtype=dt) for k, v in pts.items()}

    with torch.enable_grad():
        losses = maxwell_2d_total_loss(model, pts_cpu, physics, w_pde, w_top, w_bot, w_per, scattered)

    return {
        "best_epoch":       int(ckpt.get("best_epoch", -1)),
        "saved_best_loss":  float(ckpt.get("best_loss", float("nan"))),
        "eval_total":       float(losses["total"].detach()),
        "eval_pde":         float(losses["pde"].detach()),
        "eval_top":         float(losses["top"].detach()),
        "eval_bottom":      float(losses["bottom"].detach()),
        "eval_periodic":    float(losses["periodic"].detach()),
    }
