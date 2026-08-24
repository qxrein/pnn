"""Layered-background scattered-field formulation for the 2-D Maxwell PINN.

Motivation
----------
The free-space scattered-field formulation (E_total = E_inc + E_scat) has a
large source term in the substrate:

    (eps_sub - 1) * E_inc  ~  1.1 * E_inc

This drives the substrate PDE residual high even when the scattered field
is small.

The layered-background formulation replaces the incident plane wave with the
full planar-interface solution (flat substrate, no ridge):

    E_bg(z) = exp(-ik1*z) + r_eff*exp(+ik1*z)     z <= z_interface
    E_bg(z) = tau * exp(-ik2*(z - z_interface))    z > z_interface

where r_eff and tau satisfy the TMM boundary conditions at z_interface.

The contrast source then only involves the permittivity difference between
the actual geometry and the flat background:

    delta_eps(x,z) = eps_r(x,z) - eps_bg(x,z)

This is zero everywhere except inside the grating ridge, where:
    delta_eps = n_ridge^2 - n_substrate^2  (ridge sits on the substrate)

Key properties
--------------
1. Source = 0 in substrate (below substrate interface, outside ridge)
2. Source = 0 in air (above substrate interface, outside ridge)
3. Source ≠ 0 only inside the grating ridge itself
4. E_bg satisfies Maxwell with eps_bg exactly
5. Scattered field E_scat only needs to capture the ridge-driven scattering

Background H field
------------------
From H_tilde = (i/k0) * dE_bg/dz:

    Air:  H_bg_x = n1 * (E_fwd - r_eff * E_rfl)   ... H = (i/k0)*dE/dz
    Sub:  H_bg_x = n2 * tau * exp(-ik2*(z-z_int))

H_bg_z = 0  (z-propagating field, no x-variation)

Scattered-field PDE (nondim xbar=k0*x, zbar=k0*z)
---------------------------------------------------
Same as before but source uses delta_eps * E_bg:

    (Ar)  dEs_r/dzbar  =  Hs_x_i
    (Ai)  dEs_i/dzbar  = -Hs_x_r
    (Br)  dEs_r/dxbar  = -Hs_z_i
    (Bi)  dEs_i/dxbar  =  Hs_z_r
    (Cr)  dHs_x_r/dzbar - dHs_z_r/dxbar =  eps_r*Es_i + delta_eps*Ebg_i
    (Ci)  dHs_x_i/dzbar - dHs_z_i/dxbar = -eps_r*Es_r - delta_eps*Ebg_r

Boundary conditions for scattered field
-----------------------------------------
Top (z=0):  outgoing upward  → Hs_x = -n_air * Es   (same as before)
Bottom:     outgoing downward → CORRECTED for layered background:
    H_total_x = n_sub * E_total  (pure downward in substrate)
    H_bg_x    = (k2/k0)*tau*exp(-ik2*(z-z_int))  at bottom
    E_bg      = tau*exp(-ik2*(z-z_int))           at bottom
    H_scat_x  = H_total_x - H_bg_x
              = n_sub * E_scat + (n_sub - n_sub)*E_bg  -- wait
              = n_sub * (E_scat + E_bg) - H_bg_x
    So H_scat condition: H_scat_x = n_sub * E_scat  (since H_bg already propagating correctly)

Verification: flat-interface test
----------------------------------
If ridge_contrast = 0 (n_ridge = n_substrate), then delta_eps = 0 everywhere,
so E_scat = 0, H_scat = 0. Total field = E_bg.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# Background field coefficients
# ---------------------------------------------------------------------------


def compute_background_coefficients(physics: "PhysicsConfig") -> dict:
    """Compute Fresnel reflection/transmission for the flat substrate background.

    Returns r_eff, tau, k1, k2, z_interface so that:
        E_bg(z) = exp(-ik1*z) + r_eff*exp(+ik1*z)    for z <= z_interface
        E_bg(z) = tau*exp(-ik2*(z-z_interface))       for z >  z_interface
    with E and dE/dz continuous at z_interface.
    """
    k0  = physics.k0
    n1  = physics.n_air
    n2  = physics.n_substrate
    k1  = k0 * n1
    k2  = k0 * n2
    # flat interface at z = ridge_base_z (top of substrate)
    z_int = physics.ridge_base_z

    # Solve: exp(-ik1*z_int) + r_eff*exp(+ik1*z_int) = tau
    #         k1*(exp(-ik1*z_int) - r_eff*exp(+ik1*z_int)) = k2*tau
    E0  = np.exp(-1j*k1*z_int)
    E0r = np.exp(+1j*k1*z_int)
    A   = np.array([[E0r, -1], [k1*E0r, k2]], dtype=complex)
    b   = np.array([-E0, k1*E0], dtype=complex)
    sol = np.linalg.solve(A, b)
    r_eff, tau = sol[0], sol[1]

    # Verify
    E_air_check = E0 + r_eff*E0r
    E_sub_check = tau
    assert abs(E_air_check - E_sub_check) < 1e-10, "Background E continuity failed"
    dE_air = -1j*k1*E0 + 1j*k1*r_eff*E0r
    dE_sub = -1j*k2*tau
    assert abs(dE_air - dE_sub) < 1e-8, "Background dE/dz continuity failed"

    energy = abs(r_eff)**2 + abs(tau)**2*(k2/k1)
    assert abs(energy - 1.0) < 1e-8, f"Background energy not conserved: {energy}"

    return {
        "r_eff": r_eff, "tau": tau,
        "k1": k1, "k2": k2,
        "z_interface": z_int,
        "n1": n1, "n2": n2,
        "reflectance": abs(r_eff)**2,
        "transmittance": abs(tau)**2 * (k2/k1),
    }


def background_field_np(
    z: np.ndarray,
    coeff: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate E_bg and H_bg_x at physical z values.

    Returns (Ebg_r, Ebg_i, Hbg_x_r, Hbg_x_i) — all real arrays.
    H_bg_z = 0 everywhere.
    """
    r_eff, tau = coeff["r_eff"], coeff["tau"]
    k1, k2     = coeff["k1"], coeff["k2"]
    z_int      = coeff["z_interface"]
    k0         = k1 / coeff["n1"]  # recover k0

    E_bg = np.zeros_like(z, dtype=complex)
    H_bg = np.zeros_like(z, dtype=complex)

    air = z <= z_int
    sub = z > z_int

    # Air region
    E_bg[air]  = np.exp(-1j*k1*z[air]) + r_eff*np.exp(+1j*k1*z[air])
    H_bg[air]  = (1j/k0) * (-1j*k1*np.exp(-1j*k1*z[air]) + 1j*k1*r_eff*np.exp(+1j*k1*z[air]))
    # = (k1/k0)*(exp(-ik1*z) - r_eff*exp(+ik1*z)) = n1*(E_fwd - r_eff*E_rfl)

    # Substrate region
    zpp         = z[sub] - z_int
    E_bg[sub]   = tau * np.exp(-1j*k2*zpp)
    H_bg[sub]   = (1j/k0) * (-1j*k2*tau*np.exp(-1j*k2*zpp))
    # = (k2/k0)*tau*exp(-ik2*zpp) = n2*E_sub

    return np.real(E_bg), np.imag(E_bg), np.real(H_bg), np.imag(H_bg)


def background_field_torch(
    z: torch.Tensor,
    coeff: dict,
    physics: "PhysicsConfig",
) -> tuple[torch.Tensor, ...]:
    """Evaluate E_bg, H_bg_x at physical z (torch, differentiable-safe).

    Returns (Ebg_r, Ebg_i, Hbg_r, Hbg_i) — z is detached internally.
    """
    r_eff, tau = coeff["r_eff"], coeff["tau"]
    k1, k2     = coeff["k1"], coeff["k2"]
    z_int      = coeff["z_interface"]

    z_det = z.detach()
    air   = z_det <= z_int
    sub   = z_det > z_int

    Ebg_r = torch.zeros_like(z_det)
    Ebg_i = torch.zeros_like(z_det)
    Hbg_r = torch.zeros_like(z_det)
    Hbg_i = torch.zeros_like(z_det)

    if air.any():
        za = z_det[air]
        Efwd_r = torch.cos(k1*za); Efwd_i = -torch.sin(k1*za)
        Erfl_r = float(r_eff.real)*torch.cos(k1*za) + float(-r_eff.imag)*(-torch.sin(k1*za))
        Erfl_i = float(r_eff.real)*(-torch.sin(k1*za)) + float(r_eff.imag)*torch.cos(k1*za)
        # r_eff*exp(+ik1*z) = r_eff*(cos+isin)
        Erfl_r = float(r_eff.real)*torch.cos(k1*za) - float(r_eff.imag)*torch.sin(k1*za)
        Erfl_i = float(r_eff.imag)*torch.cos(k1*za) + float(r_eff.real)*torch.sin(k1*za)
        Ebg_r[air] = Efwd_r + Erfl_r
        Ebg_i[air] = Efwd_i + Erfl_i
        # H = n1*(E_fwd - r_eff*E_rfl) but r_eff is complex
        # H = (k1/k0)*(exp(-ik1*z) - r_eff*exp(+ik1*z))
        n1 = coeff["n1"]
        Hbg_r[air] = n1*(Efwd_r - Erfl_r)
        Hbg_i[air] = n1*(Efwd_i - Erfl_i)

    if sub.any():
        zpp_s = z_det[sub] - z_int
        tau_r, tau_i = float(tau.real), float(tau.imag)
        # tau*exp(-ik2*zpp) = tau*(cos(k2*zpp) - i*sin(k2*zpp))
        cos_s = torch.cos(k2*zpp_s); sin_s = torch.sin(k2*zpp_s)
        Ebg_r[sub] = tau_r*cos_s + tau_i*sin_s
        Ebg_i[sub] = -tau_r*sin_s + tau_i*cos_s
        n2 = coeff["n2"]
        Hbg_r[sub] = n2*Ebg_r[sub]
        Hbg_i[sub] = n2*Ebg_i[sub]

    return Ebg_r, Ebg_i, Hbg_r, Hbg_i


def delta_eps_tensor(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
) -> torch.Tensor:
    """Permittivity contrast: eps_r(x,z) - eps_bg(x,z).

    eps_bg = n_air^2 above ridge_base_z, n_sub^2 below.
    Nonzero ONLY inside the grating ridge.

    Ridge: x in [ridge_x_min, ridge_x_max], z in [ridge_z_min, ridge_z_max].
    Inside ridge: eps_r = n_ridge^2, eps_bg = n_sub^2 (ridge sits on substrate)
    """
    from src.geometry import epsilon_r

    eps_r = epsilon_r(x, z, physics)

    # Background permittivity
    eps_bg = torch.where(
        z >= physics.ridge_base_z,
        torch.full_like(z, physics.eps_substrate),
        torch.full_like(z, physics.eps_air),
    )

    return eps_r - eps_bg


def delta_eps_np(x: np.ndarray, z: np.ndarray, physics: "PhysicsConfig") -> np.ndarray:
    """Numpy version of permittivity contrast."""
    eps_r = np.full_like(z, physics.n_air**2)
    eps_r[z >= physics.ridge_base_z] = physics.n_substrate**2
    in_ridge = (
        (x >= physics.ridge_x_min) & (x <= physics.ridge_x_max) &
        (z >= physics.ridge_z_min) & (z <= physics.ridge_z_max)
    )
    eps_r[in_ridge] = physics.n_ridge**2

    eps_bg = np.where(z >= physics.ridge_base_z, physics.n_substrate**2, physics.n_air**2)
    return eps_r - eps_bg


# ---------------------------------------------------------------------------
# PDE residuals with layered background
# ---------------------------------------------------------------------------


def _g(field, wrt, ones):
    g = torch.autograd.grad(field, wrt, grad_outputs=ones,
                             create_graph=True, retain_graph=True, allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(wrt)


def maxwell_2d_lbg_pde_residual(
    subnet,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: "PhysicsConfig",
    eps_val: float,
    coeff: dict,
    source_alpha: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Layered-background scattered-field PDE residuals (nondim xbar, zbar).

    Equations (no k0 factor, E_s = scattered field, E_bg = background):
        (Ar)  dEs_r/dzbar  =  Hs_x_i
        (Ai)  dEs_i/dzbar  = -Hs_x_r
        (Br)  dEs_r/dxbar  = -Hs_z_i
        (Bi)  dEs_i/dxbar  =  Hs_z_r
        (Cr)  dHs_x_r/dzbar - dHs_z_r/dxbar =  eps_r*Es_i + delta*Ebg_i
        (Ci)  dHs_x_i/dzbar - dHs_z_i/dxbar = -eps_r*Es_r - delta*Ebg_r

    where delta = eps_r - eps_bg (contrast source, nonzero only in ridge).
    """
    k0 = physics.k0
    xbar = (x * k0).detach().clone().requires_grad_(True)
    zbar = (z * k0).detach().clone().requires_grad_(True)
    x_phys = xbar / k0; z_phys = zbar / k0

    Er_s, Ei_s, Hr_x, Hi_x, Hr_z, Hi_z = subnet.field_components(x_phys, z_phys)
    ones = torch.ones_like(Er_s)

    dEr_dz  = _g(Er_s, zbar, ones); dEi_dz  = _g(Ei_s, zbar, ones)
    dHrx_dz = _g(Hr_x, zbar, ones); dHix_dz = _g(Hi_x, zbar, ones)
    dEr_dx  = _g(Er_s, xbar, ones); dEi_dx  = _g(Ei_s, xbar, ones)
    dHrz_dx = _g(Hr_z, xbar, ones); dHiz_dx = _g(Hi_z, xbar, ones)

    # Background field at these points
    Ebg_r, Ebg_i, _, _ = background_field_torch(z_phys.detach(), coeff, physics)

    # Contrast source
    delta = delta_eps_tensor(x_phys.detach(), z_phys.detach(), physics)

    res = [
        dEr_dz  -  Hi_x,
        dEi_dz  +  Hr_x,
        dEr_dx  +  Hi_z,
        dEi_dx  -  Hr_z,
        dHrx_dz - dHrz_dx - eps_val * Ei_s - source_alpha * delta * Ebg_i,
        dHix_dz - dHiz_dx + eps_val * Er_s + source_alpha * delta * Ebg_r,
    ]

    return tuple(res)


# ---------------------------------------------------------------------------
# Boundary conditions for layered-background scattered field
# ---------------------------------------------------------------------------


def lbg_top_bc(subnet, x_pts: torch.Tensor, physics: "PhysicsConfig",
               use_dtn: bool = False, n_dtn_orders: int = 8) -> torch.Tensor:
    """Top BC z=0: outgoing upward scattered field.

    Robin (default):   H_scat_x = -n_air * E_scat
    Modal DtN:         H_scat_x = IDFT(+(kz_m/k0) * DFT(E_scat))

    The DtN condition correctly handles all grating orders including ±1 and
    evanescent orders, whereas Robin is only exact for m=0 normal incidence.
    """
    if use_dtn:
        from src.modal_dtn import modal_dtn_loss_lbg
        return modal_dtn_loss_lbg(subnet, x_pts, 0.0, physics, {},
                                  "top", physics.n_air, n_dtn_orders)
    z = torch.zeros_like(x_pts)
    Er, Ei, Hr_x, Hi_x, _, _ = subnet.field_components(x_pts, z)
    n = physics.n_air
    return torch.mean((Hr_x + n * Er)**2 + (Hi_x + n * Ei)**2)


def lbg_bottom_bc(subnet, x_pts: torch.Tensor, physics: "PhysicsConfig",
                   coeff: dict, use_dtn: bool = False, n_dtn_orders: int = 8) -> torch.Tensor:
    """Bottom BC z=domain_height: outgoing scattered wave in substrate.

    With layered background, E_bg already contains the correct substrate wave.
    The scattered field in the substrate is purely outgoing from the grating.

    Robin (default):   H_scat_x = n_sub * E_scat
    Modal DtN:         H_scat_x = IDFT(-(kz_m/k0) * DFT(E_scat))

    The DtN correctly handles the ±1 transmitted diffraction orders which carry
    ~8% each of the incident power for the Λ=0.8λ grating.
    """
    if use_dtn:
        from src.modal_dtn import modal_dtn_loss_lbg
        return modal_dtn_loss_lbg(subnet, x_pts, physics.domain_height, physics, coeff,
                                  "bottom", physics.n_substrate, n_dtn_orders)
    z_bot = physics.domain_height
    z = torch.full_like(x_pts, z_bot)
    Er_s, Ei_s, Hr_x, Hi_x, _, _ = subnet.field_components(x_pts, z)
    n_sub = physics.n_substrate
    return torch.mean((Hr_x - n_sub * Er_s)**2 + (Hi_x - n_sub * Ei_s)**2)


def lbg_vertical_interface_loss(subnet, x_interface: float, z_pts: torch.Tensor,
                                offset: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided tangential E_y and H_z continuity at a vertical ridge edge."""
    xl = torch.full_like(z_pts, x_interface - offset)
    xr = torch.full_like(z_pts, x_interface + offset)
    Elr, Eli, _, _, Hzlr, Hzli = subnet.field_components(xl, z_pts)
    Err, Eri, _, _, Hzrr, Hzri = subnet.field_components(xr, z_pts)
    return (torch.mean((Elr-Err)**2 + (Eli-Eri)**2),
            torch.mean((Hzlr-Hzrr)**2 + (Hzli-Hzri)**2))


# ---------------------------------------------------------------------------
# Total field reconstruction
# ---------------------------------------------------------------------------


def reconstruct_total_field(
    pinn_E_scat_r: np.ndarray,
    pinn_E_scat_i: np.ndarray,
    z_np: np.ndarray,
    physics: "PhysicsConfig",
    coeff: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Add background field to scattered field to get total field."""
    Ebg_r, Ebg_i, _, _ = background_field_np(z_np, coeff)
    return pinn_E_scat_r + Ebg_r, pinn_E_scat_i + Ebg_i


# ---------------------------------------------------------------------------
# Diagnostic: source map
# ---------------------------------------------------------------------------


def compute_source_map(
    physics: "PhysicsConfig",
    coeff: dict,
    nx: int = 128,
    nz: int = 256,
) -> dict[str, np.ndarray]:
    """Compute and return:
    - eps_r(x,z)
    - eps_bg(x,z)
    - delta_eps = eps_r - eps_bg
    - |contrast_source| = |delta_eps * E_bg|
    """
    x1d = np.linspace(0, physics.period, nx)
    z1d = np.linspace(0, physics.domain_height, nz)
    X, Z = np.meshgrid(x1d, z1d)
    x_flat = X.ravel(); z_flat = Z.ravel()

    # eps_r
    eps_r = np.full_like(z_flat, physics.n_air**2)
    eps_r[z_flat >= physics.ridge_base_z] = physics.n_substrate**2
    in_ridge = (
        (x_flat >= physics.ridge_x_min) & (x_flat <= physics.ridge_x_max) &
        (z_flat >= physics.ridge_z_min) & (z_flat <= physics.ridge_z_max)
    )
    eps_r[in_ridge] = physics.n_ridge**2

    # eps_bg
    eps_bg = np.where(z_flat >= physics.ridge_base_z, physics.n_substrate**2, physics.n_air**2)

    # delta
    delta = eps_r - eps_bg

    # E_bg
    Ebg_r, Ebg_i, _, _ = background_field_np(z_flat, coeff)
    E_bg_mag = np.sqrt(Ebg_r**2 + Ebg_i**2)

    # |source| = |delta * E_bg|
    source_mag = np.abs(delta) * E_bg_mag

    return {
        "x": X, "z": Z,
        "eps_r":     eps_r.reshape(X.shape),
        "eps_bg":    eps_bg.reshape(X.shape),
        "delta_eps": delta.reshape(X.shape),
        "source_mag": source_mag.reshape(X.shape),
        "E_bg_mag":  E_bg_mag.reshape(X.shape),
    }
