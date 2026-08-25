#!/usr/bin/env python3
"""Generate a reference electromagnetic field using Rigorous Coupled-Wave Analysis (RCWA).

Scalar TE Helmholtz equation, normal incidence, 2D binary grating.
Coordinate convention matches the PINN (z=0 top, z increases downward).

Output: outputs/reference_grating.npz with keys x, z, E_real, E_imag.

Method: Fourier Modal Method with Redheffer S-matrix cascading.
Amplitudes at each layer boundary are extracted via partial S-matrices
(no phase-factor inversion) to ensure numerical stability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kx_orders(N: int, k0: float, period: float) -> np.ndarray:
    """Bloch kx for orders -N..+N at normal incidence."""
    return np.arange(-N, N + 1) * (2.0 * np.pi / period)


def _kz_branch(kz2: np.ndarray) -> np.ndarray:
    """Choose kz branch for forward-propagating / decaying modes.

    For exp(-i kz z) in +z direction:
      - Propagating (kz2 > 0):  kz real, positive → oscillatory.
      - Evanescent  (kz2 < 0):  kz = -i|β| so exp(-i kz z) = exp(-|β|z) decays.
    np.sqrt gives Im(kz) >= 0, so for evanescent orders we negate.
    """
    kz = np.sqrt(kz2.astype(complex))
    evanescent = kz2.real < -1e-14         # strictly negative → evanescent
    kz = np.where(evanescent, -kz, kz)     # flip to Im < 0
    kz = np.where(kz.real < 0, -kz, kz)   # ensure Re >= 0 for propagating
    return kz


def _kz_uniform(kx: np.ndarray, k0: float, eps: float) -> np.ndarray:
    return _kz_branch(k0**2 * eps - kx**2)


def _fourier_eps(eps_x: np.ndarray, N: int) -> np.ndarray:
    """Toeplitz Fourier-coefficient matrix of a 1-D permittivity profile."""
    Nfine = len(eps_x)
    eps_hat = np.fft.fft(eps_x) / Nfine
    n = 2 * N + 1
    E = np.empty((n, n), dtype=complex)
    for m in range(n):
        for p in range(n):
            E[m, p] = eps_hat[(m - p) % Nfine]
    return E


def _grating_modes(E_mat: np.ndarray, kx: np.ndarray, k0: float):
    """Eigenvalues γ and mode matrix W for the grating layer.

    FMM equation: (Kx² - k0² E) W = W diag(-γ²).
    Same branch convention as _kz_branch.
    """
    M = np.diag(kx**2) - k0**2 * E_mat
    eigvals, W = np.linalg.eig(-M)        # eigenvalues = +γ²
    gamma = _kz_branch(eigvals.real)
    return gamma, W


# ---------------------------------------------------------------------------
# S-matrix building blocks
# ---------------------------------------------------------------------------


def _s_interface(W_L, kz_L, W_R, kz_R) -> np.ndarray:
    """S-matrix for an interface between two media.

    Convention: [c_R+; c_L-] = S @ [c_L+; c_R-].
    TE boundary conditions (continuity of Ey and dEy/dz):
        W_L (c_L+ + c_L-) = W_R (c_R+ + c_R-)
        W_L KZ_L (-c_L+ + c_L-) = W_R KZ_R (-c_R+ + c_R-)
    """
    KZ_L = np.diag(kz_L)
    KZ_R = np.diag(kz_R)
    # Rearranging: unknowns x = [c_R+; c_L-], RHS depends on [c_L+; c_R-]
    # A x = B [c_L+; c_R-]
    A = np.block([[W_R, -W_L], [-W_R @ KZ_R, -W_L @ KZ_L]])
    B = np.block([[W_L, -W_R], [-W_L @ KZ_L, -W_R @ KZ_R]])
    return np.linalg.solve(A, B)


def _s_propagate(kz: np.ndarray, h: float) -> np.ndarray:
    """S-matrix for propagation through a uniform slab of thickness h.

    Both forward and backward waves accumulate phase exp(-i kz h).
    """
    n = len(kz)
    phi = np.exp(-1j * kz * h)
    S = np.zeros((2 * n, 2 * n), dtype=complex)
    S[:n, :n] = np.diag(phi)
    S[n:, n:] = np.diag(phi)
    return S


def _star(Sa: np.ndarray, Sb: np.ndarray) -> np.ndarray:
    """Redheffer star product Sa ★ Sb.

    Convention: [c_R+; c_L-] = S @ [c_L+; c_R-]
    Blocks: Sa = [[A,B],[C,D]], Sb = [[E,F],[G,H]]

    Derivation:
        c_M+ = A c_L+ + B c_M-          (Sa forward)
        c_L- = C c_L+ + D c_M-          (Sa backward)
        c_R+ = E c_M+ + F c_R-          (Sb forward)
        c_M- = G c_M+ + H c_R-          (Sb backward)

    Eliminating c_M+ and c_M- via M = (I - B G)^{-1}:
        c_M+ = M A c_L+ + M B H c_R-
        c_M- = G M A c_L+ + (G M B + I) H c_R-

    Combined S-matrix:
        S[0,0] = E M A
        S[0,1] = F + E M B H
        S[1,0] = C + D G M A
        S[1,1] = D (G M B + I) H

    This reduces to identity when Sa = I or Sb = I, and satisfies
    energy conservation for lossless media.
    """
    N = Sa.shape[0] // 2
    A, B, C, D = Sa[:N, :N], Sa[:N, N:], Sa[N:, :N], Sa[N:, N:]
    E_, F, G, H = Sb[:N, :N], Sb[:N, N:], Sb[N:, :N], Sb[N:, N:]
    I = np.eye(N, dtype=complex)
    M = np.linalg.inv(I - B @ G)   # (I - Sa[0,1] @ Sb[1,0])^{-1}
    return np.block([
        [E_ @ M @ A,               F + E_ @ M @ B @ H],
        [C + D @ G @ M @ A,        D @ (G @ M @ B + I) @ H],
    ])


# ---------------------------------------------------------------------------
# Amplitude extraction
# ---------------------------------------------------------------------------


def _amplitudes(SL: np.ndarray, SR: np.ndarray, c_inc: np.ndarray) -> tuple:
    """Forward and backward amplitudes at a boundary.

    Given partial S-matrix SL from z=0 to this boundary, and SR from this
    boundary to z_bot, with incident amplitude c_inc at z=0:

        c_fwd = (I - SL[0,1] SR[1,0])^{-1} SL[0,0] c_inc
        c_bwd = SR[1,0] c_fwd
    """
    n = len(c_inc)
    SL11, SL12 = SL[:n, :n], SL[:n, n:]
    SR21 = SR[n:, :n]
    A = np.eye(n, dtype=complex) - SL12 @ SR21
    c_fwd = np.linalg.solve(A, SL11 @ c_inc)
    c_bwd = SR21 @ c_fwd
    return c_fwd, c_bwd


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def compute_rcwa_state(
    physics: PhysicsConfig,
    N_harmonics: int = 25,
    Nfine: int = 2048,
) -> dict:
    """Layerwise RCWA/FMM state from the canonical generator.

    Magnetic fields are obtained from the same modal coefficients as E_y,
    never from finite differences of a visualization raster.
    """
    k0 = physics.k0
    N = N_harmonics
    n = 2 * N + 1

    z_g_bot = physics.ridge_z_min
    z_g_top = physics.ridge_z_max
    z_bot = physics.domain_height
    h_air = z_g_bot
    h_grat = z_g_top - z_g_bot
    h_sub = z_bot - z_g_top

    kx = _kx_orders(N, k0, physics.period)
    kz_air = _kz_uniform(kx, k0, physics.eps_air)
    kz_sub = _kz_uniform(kx, k0, physics.eps_substrate)

    x_fine = np.linspace(0.0, physics.period, Nfine, endpoint=False)
    # The ridge is a dielectric protrusion *on the substrate*.  Its grating
    # layer therefore has substrate permittivity outside the ridge footprint,
    # exactly as ``src.geometry.epsilon_r`` and the layered-background contrast
    # formulation do.  Using air here made n_ridge=n_substrate spuriously
    # diffract, violating the zero-source limit.
    eps_x = np.where(
        (x_fine >= physics.ridge_x_min) & (x_fine <= physics.ridge_x_max),
        physics.eps_ridge, physics.eps_substrate,
    )
    E_mat = _fourier_eps(eps_x, N)
    gamma, W = _grating_modes(E_mat, kx, k0)

    I_n = np.eye(n, dtype=complex)
    S_I = np.block([[I_n, np.zeros_like(I_n)], [np.zeros_like(I_n), I_n]])  # identity

    Sa = _s_propagate(kz_air, h_air)         # air slab propagation
    Sb = _s_interface(I_n, kz_air, W, gamma)  # air → grating interface
    Sc = _s_propagate(gamma, h_grat)          # grating slab propagation
    Sd = _s_interface(W, gamma, I_n, kz_sub)  # grating → substrate interface
    Se = _s_propagate(kz_sub, h_sub)          # substrate slab propagation

    # Cumulative partial S-matrices from the LEFT (from z=0 downward)
    SL_a    = Sa
    SL_ab   = _star(Sa, Sb)
    SL_abc  = _star(SL_ab, Sc)
    SL_abcd = _star(SL_abc, Sd)
    SL_all  = _star(SL_abcd, Se)   # = S_global

    # Cumulative partial S-matrices from the RIGHT (from z=z_bot upward)
    SR_e    = Se
    SR_de   = _star(Sd, Se)
    SR_cde  = _star(Sc, SR_de)
    SR_bcde = _star(Sb, SR_cde)

    # Incident amplitude (order 0, unit amplitude)
    c_inc = np.zeros(n, dtype=complex)
    c_inc[N] = 1.0

    # Amplitudes at each layer boundary (stable, no phase inversion):
    # z = 0          : top of air slab
    c_fwd_air_top, c_bwd_air_top = _amplitudes(S_I, SL_all, c_inc)
    # z = z_g_bot    : bottom of air slab (left of air-grating interface)
    c_fwd_air_bot, c_bwd_air_bot = _amplitudes(SL_a, SR_bcde, c_inc)
    # z = z_g_bot    : right of air-grating interface (= top of grating slab)
    c_fwd_grat_top, c_bwd_grat_top = _amplitudes(SL_ab, SR_cde, c_inc)
    # z = z_g_top    : bottom of grating slab (left of grating-sub interface)
    c_fwd_grat_bot, c_bwd_grat_bot = _amplitudes(SL_abc, SR_de, c_inc)
    # z = z_g_top    : right of grating-sub interface (= top of substrate slab)
    c_fwd_sub_top, c_bwd_sub_top = _amplitudes(SL_abcd, SR_e, c_inc)

    return {
        "physics": physics,
        "N_harmonics": N,
        "Nfine": Nfine,
        "k0": k0,
        "n": n,
        "z_g_bot": z_g_bot,
        "z_g_top": z_g_top,
        "z_bot": z_bot,
        "h_air": h_air,
        "h_grat": h_grat,
        "h_sub": h_sub,
        "kx": kx,
        "kz_air": kz_air,
        "kz_sub": kz_sub,
        "gamma": gamma,
        "W": W,
        "c_inc": c_inc,
        "c_fwd_air_top": c_fwd_air_top,
        "c_bwd_air_top": c_bwd_air_top,
        "c_fwd_air_bot": c_fwd_air_bot,
        "c_bwd_air_bot": c_bwd_air_bot,
        "c_fwd_grat_top": c_fwd_grat_top,
        "c_bwd_grat_top": c_bwd_grat_top,
        "c_fwd_grat_bot": c_fwd_grat_bot,
        "c_bwd_grat_bot": c_bwd_grat_bot,
        "c_fwd_sub_top": c_fwd_sub_top,
        "c_bwd_sub_top": c_bwd_sub_top,
    }


def _auto_layer(state: dict, zv: float) -> str:
    """Raster layer assignment used by the canonical E_y reconstruction."""
    if zv <= state["z_g_bot"]:
        return "air"
    if zv <= state["z_g_top"]:
        return "grating"
    return "substrate"


def fourier_eh_at_z(state: dict, zv: float, layer: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fourier coefficients of (E_y, H̃_x, H̃_z) at one z from modal amplitudes.

    TE convention (exp(+iωt), H̃ = Z0 H):
        H̃_x = (i/k0) ∂E_y/∂z
        H̃_z = -(i/k0) ∂E_y/∂x
    so a downward wave exp(-i kz z) has H̃_x = (kz/k0) E_y.
    """
    k0 = state["k0"]
    kx = state["kx"]
    layer = _auto_layer(state, zv) if layer is None else layer

    if layer == "air":
        kz = state["kz_air"]
        fwd = state["c_fwd_air_top"] * np.exp(-1j * kz * zv)
        bwd = state["c_bwd_air_bot"] * np.exp(+1j * kz * (zv - state["z_g_bot"]))
        ey = fwd + bwd
        hx = (kz / k0) * (fwd - bwd)
        hz = (kx / k0) * ey
        return ey, hx, hz

    if layer == "grating":
        gamma = state["gamma"]
        W = state["W"]
        zl = zv - state["z_g_bot"]
        fwd = state["c_fwd_grat_top"] * np.exp(-1j * gamma * zl)
        bwd = state["c_bwd_grat_bot"] * np.exp(+1j * gamma * (zl - state["h_grat"]))
        modal = fwd + bwd
        ey = W @ modal
        hx = W @ ((gamma / k0) * (fwd - bwd))
        hz = (kx / k0) * ey
        return ey, hx, hz

    if layer == "substrate":
        kz = state["kz_sub"]
        fwd = state["c_fwd_sub_top"] * np.exp(-1j * kz * (zv - state["z_g_top"]))
        ey = fwd
        hx = (kz / k0) * fwd
        hz = (kx / k0) * ey
        return ey, hx, hz

    raise ValueError(f"Unknown RCWA layer: {layer}")


def fourier_eh_derivatives_at_z(
    state: dict, zv: float, layer: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Fourier coefficients of E_y, H̃_x, H̃_z and z-derivatives.

    Returns (Ey, Hx, Hz, dEy_dz, dHx_dz) in Fourier space.  x-derivatives
    follow from multiplication by i kx.
    """
    k0 = state["k0"]
    kx = state["kx"]
    layer = _auto_layer(state, zv) if layer is None else layer

    if layer == "air":
        kz = state["kz_air"]
        fwd = state["c_fwd_air_top"] * np.exp(-1j * kz * zv)
        bwd = state["c_bwd_air_bot"] * np.exp(+1j * kz * (zv - state["z_g_bot"]))
        ey = fwd + bwd
        hx = (kz / k0) * (fwd - bwd)
        d_fwd = -1j * kz * fwd
        d_bwd = +1j * kz * bwd
        return ey, hx, (kx / k0) * ey, d_fwd + d_bwd, (kz / k0) * (d_fwd - d_bwd)

    if layer == "grating":
        gamma = state["gamma"]
        W = state["W"]
        zl = zv - state["z_g_bot"]
        fwd = state["c_fwd_grat_top"] * np.exp(-1j * gamma * zl)
        bwd = state["c_bwd_grat_bot"] * np.exp(+1j * gamma * (zl - state["h_grat"]))
        d_fwd = -1j * gamma * fwd
        d_bwd = +1j * gamma * bwd
        ey = W @ (fwd + bwd)
        hx = W @ ((gamma / k0) * (fwd - bwd))
        return ey, hx, (kx / k0) * ey, W @ (d_fwd + d_bwd), W @ ((gamma / k0) * (d_fwd - d_bwd))

    if layer == "substrate":
        kz = state["kz_sub"]
        fwd = state["c_fwd_sub_top"] * np.exp(-1j * kz * (zv - state["z_g_top"]))
        d_fwd = -1j * kz * fwd
        ey = fwd
        hx = (kz / k0) * fwd
        return ey, hx, (kx / k0) * ey, d_fwd, (kz / k0) * d_fwd

    raise ValueError(f"Unknown RCWA layer: {layer}")


def reconstruct_ey_grid(state: dict, x: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
    """Canonical E_y reconstruction on a visualization grid (Nz, Nx)."""
    X_phase = np.exp(1j * np.outer(x, state["kx"]))
    E_field = np.zeros((len(z_arr), len(x)), dtype=complex)
    for iz, zv in enumerate(z_arr):
        ey, _, _ = fourier_eh_at_z(state, float(zv), layer=None)
        E_field[iz, :] = X_phase @ ey
    return E_field


def reconstruct_eh_grid(
    state: dict, x: np.ndarray, z_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal (E_y, H̃_x, H̃_z) on a visualization grid."""
    X_phase = np.exp(1j * np.outer(x, state["kx"]))
    nz, nx = len(z_arr), len(x)
    Ey = np.zeros((nz, nx), dtype=complex)
    Hx = np.zeros((nz, nx), dtype=complex)
    Hz = np.zeros((nz, nx), dtype=complex)
    for iz, zv in enumerate(z_arr):
        ey, hx, hz = fourier_eh_at_z(state, float(zv), layer=None)
        Ey[iz, :] = X_phase @ ey
        Hx[iz, :] = X_phase @ hx
        Hz[iz, :] = X_phase @ hz
    return Ey, Hx, Hz


def solve_rcwa(
    physics: PhysicsConfig,
    N_harmonics: int = 25,
    Nfine: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """RCWA + FMM solution for the binary grating.

    Returns x (Nx,), z (Nz,), E_real (Nz, Nx), E_imag (Nz, Nx), amplitudes dict.

    amplitudes dict contains:
        c_refl  : (n,) complex reflection amplitudes at z=0
        c_trans : (n,) complex transmission amplitudes into substrate
        kz_air  : (n,) z-wavenumbers in air
        kz_sub  : (n,) z-wavenumbers in substrate
        kx      : (n,) x-wavenumbers (Bloch orders)
        R_total, T_total, R_m, T_m, energy_check
    """
    state = compute_rcwa_state(physics, N_harmonics=N_harmonics, Nfine=Nfine)
    Nx = physics.nx_visualization
    Nz = physics.nz_visualization
    x = np.linspace(0.0, physics.period, Nx)
    z_arr = np.linspace(0.0, state["z_bot"], Nz)
    E_field = reconstruct_ey_grid(state, x, z_arr)
    amplitudes = _compute_amplitudes_and_energy(
        state["c_bwd_air_top"], state["c_fwd_sub_top"],
        state["kz_air"], state["kz_sub"], state["kx"], state["N_harmonics"],
    )
    return x, z_arr, np.real(E_field), np.imag(E_field), amplitudes


def _compute_amplitudes_and_energy(
    c_bwd_air_top, c_fwd_sub_top, kz_air, kz_sub, kx, N
):
    """Compute reflection/transmission amplitudes and energy balance."""
    kz0_air = kz_air[N]
    R_m = np.zeros(len(kx)); T_m = np.zeros(len(kx))
    for m in range(len(kx)):
        if kz_air[m].real > 1e-6:
            R_m[m] = abs(c_bwd_air_top[m])**2 * kz_air[m].real / kz0_air.real
        if kz_sub[m].real > 1e-6:
            T_m[m] = abs(c_fwd_sub_top[m])**2 * kz_sub[m].real / kz0_air.real
    return {
        "c_refl":  c_bwd_air_top,
        "c_trans": c_fwd_sub_top,
        "kz_air": kz_air, "kz_sub": kz_sub, "kx": kx,
        "R_m": R_m, "T_m": T_m,
        "R_total": float(R_m.sum()),
        "T_total": float(T_m.sum()),
        "energy_check": float(R_m.sum() + T_m.sum()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RCWA reference field")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/reference_grating.npz")
    parser.add_argument("--n-harmonics", type=int, default=25)
    parser.add_argument("--convergence-check", action="store_true")
    parser.add_argument("--period", type=float, default=None,
                        help="Override period (in units of wavelength). Default: use config.")
    parser.add_argument("--ridge-width-fraction", type=float, default=None,
                        help="ridge_width / period. Default: keep config value.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    # Override period if requested
    if args.period is not None:
        new_period = args.period * config.physics.wavelength
        fill = args.ridge_width_fraction or (config.physics.ridge_width / config.physics.period)
        config.physics.period = new_period
        config.physics.ridge_width = fill * new_period
        print(f"  Override: period={new_period:.4f}  ridge_width={config.physics.ridge_width:.4f}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = args.n_harmonics
    print(f"RCWA  N_harmonics={N}  (total orders={2*N+1})")
    x, z, E_real, E_imag, amps = solve_rcwa(config.physics, N_harmonics=N)
    mag = np.sqrt(E_real**2 + E_imag**2)
    print(f"  |E| range : [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  |E| mean  : {mag.mean():.4f}")
    print(f"  |r_0|     : {abs(amps['c_refl'][args.n_harmonics]):.6f}  (m=0 reflection amplitude)")
    print(f"  |t_0|     : {abs(amps['c_trans'][args.n_harmonics]):.6f}  (m=0 transmission amplitude)")
    print(f"  R_total   : {amps['R_total']:.6f}")
    print(f"  T_total   : {amps['T_total']:.6f}")
    print(f"  R+T       : {amps['energy_check']:.6f}  (should be 1.0)")

    if args.convergence_check:
        N2 = N + 10
        print(f"Convergence check  N_harmonics={N2} ...")
        _, _, Er2, Ei2, _ = solve_rcwa(config.physics, N_harmonics=N2)
        mag2 = np.sqrt(Er2**2 + Ei2**2)
        rel = np.linalg.norm(mag - mag2) / (np.linalg.norm(mag2) + 1e-30)
        print(f"  Relative |E| diff N={N} vs N={N2}: {rel:.2e}")
        if rel > 0.02:
            print("  WARNING: > 2% — consider increasing --n-harmonics.")
        else:
            print("  Converged (< 2%).")

    np.savez(output_path, x=x, z=z, E_real=E_real, E_imag=E_imag,
             field_representation="total",
             wavelength=config.physics.wavelength,
             period=config.physics.period,
             ridge_width=config.physics.ridge_width,
             ridge_height=config.physics.ridge_height,
             ridge_base_fraction=config.physics.ridge_base_fraction,
             n_air=config.physics.n_air,
             n_ridge=config.physics.n_ridge,
             n_substrate=config.physics.n_substrate,
             k0=config.physics.k0,
             domain_height=config.physics.domain_height,
             coordinate_convention="z=0 top, z increases downward, E_inc=exp(-ik0*z)",
             geometry_convention="dielectric ridge on substrate; substrate outside ridge footprint",
             z_top_monitor=0.08 * config.physics.domain_height,
             z_bot_monitor=0.92 * config.physics.domain_height,
             c_refl=amps["c_refl"], c_trans=amps["c_trans"],
             kz_air=amps["kz_air"], kz_sub=amps["kz_sub"], kx=amps["kx"],
             R_m=amps["R_m"], T_m=amps["T_m"],
             R_total=amps["R_total"], T_total=amps["T_total"])
    print(f"Saved → {output_path}")
    print(f"  x: {x.shape}  z: {z.shape}  E_real: {E_real.shape}")


if __name__ == "__main__":
    main()
