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
    """Redheffer star product Sa ★ Sb."""
    N = Sa.shape[0] // 2
    A, B, C, D = Sa[:N, :N], Sa[:N, N:], Sa[N:, :N], Sa[N:, N:]
    E_, F, G, H = Sb[:N, :N], Sb[:N, N:], Sb[N:, :N], Sb[N:, N:]
    I = np.eye(N, dtype=complex)
    X = np.linalg.inv(I - E_ @ D)
    Y = np.linalg.inv(I - D @ E_)
    return np.block([
        [A + B @ X @ E_ @ C,   B @ X @ F  ],
        [G @ Y @ C,             H + G @ Y @ D @ F],
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


def solve_rcwa(
    physics: PhysicsConfig,
    N_harmonics: int = 25,
    Nfine: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RCWA + FMM solution for the binary grating.

    Returns x (Nx,), z (Nz,), E_real (Nz, Nx), E_imag (Nz, Nx).
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
    eps_x = np.where(
        (x_fine >= physics.ridge_x_min) & (x_fine <= physics.ridge_x_max),
        physics.eps_ridge, physics.eps_air,
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
    SR_all  = _star(Sa, SR_bcde)   # = S_global again

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

    # Visualization grid
    Nx = physics.nx_visualization
    Nz = physics.nz_visualization
    x = np.linspace(0.0, physics.period, Nx)
    z_arr = np.linspace(0.0, z_bot, Nz)
    X_phase = np.exp(1j * np.outer(x, kx))   # (Nx, n)

    E_field = np.zeros((Nz, Nx), dtype=complex)

    for iz, zv in enumerate(z_arr):
        if zv <= z_g_bot:
            zl = zv
            fwd = c_fwd_air_top * np.exp(-1j * kz_air * zl)
            bwd = c_bwd_air_top * np.exp(+1j * kz_air * zl)
            E_field[iz, :] = X_phase @ (fwd + bwd)
        elif zv <= z_g_top:
            zl = zv - z_g_bot
            fwd = c_fwd_grat_top * np.exp(-1j * gamma * zl)
            bwd = c_bwd_grat_top * np.exp(+1j * gamma * zl)
            E_field[iz, :] = X_phase @ (W @ (fwd + bwd))
        else:
            zl = zv - z_g_top
            fwd = c_fwd_sub_top * np.exp(-1j * kz_sub * zl)
            bwd = c_bwd_sub_top * np.exp(+1j * kz_sub * zl)
            E_field[iz, :] = X_phase @ (fwd + bwd)

    return x, z_arr, np.real(E_field), np.imag(E_field)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RCWA reference field")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/reference_grating.npz")
    parser.add_argument("--n-harmonics", type=int, default=25)
    parser.add_argument("--convergence-check", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = args.n_harmonics
    print(f"RCWA  N_harmonics={N}  (total orders={2*N+1})")
    x, z, E_real, E_imag = solve_rcwa(config.physics, N_harmonics=N)
    mag = np.sqrt(E_real**2 + E_imag**2)
    print(f"  |E| range : [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"  |E| mean  : {mag.mean():.4f}")

    if args.convergence_check:
        N2 = N + 10
        print(f"Convergence check  N_harmonics={N2} ...")
        _, _, Er2, Ei2 = solve_rcwa(config.physics, N_harmonics=N2)
        mag2 = np.sqrt(Er2**2 + Ei2**2)
        rel = np.linalg.norm(mag - mag2) / (np.linalg.norm(mag2) + 1e-30)
        print(f"  Relative |E| diff N={N} vs N={N2}: {rel:.2e}")
        if rel > 0.02:
            print("  WARNING: > 2% — consider increasing --n-harmonics.")
        else:
            print("  Converged (< 2%).")

    np.savez(output_path, x=x, z=z, E_real=E_real, E_imag=E_imag)
    print(f"Saved → {output_path}")
    print(f"  x: {x.shape}  z: {z.shape}  E_real: {E_real.shape}")


if __name__ == "__main__":
    main()
