"""Modal Dirichlet-to-Neumann (DtN) radiation boundary conditions.

Motivation
----------
The simple Robin condition  H_x = ±n · E  is exact only for a single plane
wave at normal incidence.  For a grating scattering problem the field at the
boundary is a sum of diffraction orders:

    E(x, z_b) = Σ_m  A_m · exp(i G_m x) · exp(-i kz_m · (z - z_b))

where  G_m = G_0 + m · (2π/Λ)  and  kz_m = sqrt((n·k0)² - kx_m²).

The correct boundary condition per mode is:

    ∂E/∂z = -i kz_m · A_m · exp(i G_m x)     (for outgoing +z mode)

Integrating over x against exp(-i G_m x) gives the DtN map:

    (∂E/∂z)_m  =  -i kz_m · E_m

In the first-order TE Maxwell system  ∂E/∂z = i k0 H_x  (physical coords),
i.e.  i k0 H_x = -i kz_m E_m,  so:

    H_x_m = -(kz_m / k0) · E_m    (outgoing downward at bottom, TE)

At the top boundary (z = 0) the scattered field is outgoing upward (−z):

    H_x_m = +(kz_m / k0) · E_m    (outgoing upward at top)

For the scattered field only:
- The background field already satisfies its own boundary conditions.
- We apply the DtN to the scattered E_scat and H_scat only.

Evanescent orders (kz_m purely imaginary, Im(kz_m) > 0 after branch choice):
- Must decay away from the grating.
- Bottom:  exp(-i kz_m z)  with Im(kz_m) > 0  decays as z → +∞.
- Top:     exp(+i kz_m z)  with Im(kz_m) > 0  decays as z → -∞.
- The DtN relation still holds with the complex kz_m.

Sign convention (matches maxwell_2d_nondim.py)
-----------------------------------------------
Physical Maxwell: ∂E/∂z = k0 · H̃_x_i,  ∂E/∂z = -k0 · H̃_x_r  (separated real/imag)
Wait — the TE equations are:
    (Ar)  ∂Er/∂zbar = H̃i_x       ↔  k0·∂Er/∂z = k0·H̃i_x  ↔  ∂Er/∂z = k0·H̃i_x
    (Ai)  ∂Ei/∂zbar = -H̃r_x
H̃_x is defined with the 1/k0 absorbed:  H̃_x = (1/μ0 ω) · dE/dx gives the
physical H.  In the convention used here  H̃_x = H_physical / (k0/ω·μ0).

From E_m = A_m (complex scalar for order m):
    ∂E/∂z = -i kz_m · E_m        (outgoing +z)
    k0 · H̃_x_complex = ∂E/∂z   ↔  H̃_x = -(i kz_m / k0) · E_m

In real/imag:
    H̃r_x_m = -(Re(i kz_m / k0)) · Re(E_m) + Im(i kz_m / k0) · Im(E_m)
            = (kz_r/k0)·Im(E_m) - (-kz_i/k0)·Re(E_m)   [where kz = kz_r + i kz_i]

But it is cleaner to work in complex and split at the end.

Implementation
--------------
For each boundary and each point x_j (on a uniform grid over [0, Λ]):

1. Evaluate E_scat and H_scat at z = z_b using the PINN.
2. DFT to get modal amplitudes:  E_m = DFT(E_scat)[m],  H_m = DFT(H_scat)[m]
3. Compute expected  H_m^DtN = -(kz_m / k0) · E_m  (bottom)  or  +(kz_m/k0)·E_m  (top)
4. Loss = Σ_m  |H_m - H_m^DtN|²  (or |H_m + kz_m/k0 · E_m|² pointwise after IDFT)

For efficiency we implement the loss as a *pointwise* condition in x-space
by performing the DFT, applying the DtN, and inverting:

    H_DtN(x) = IDFT( -(kz_m / k0) · DFT(E_scat)(m) )

This gives a smooth x-dependent target for H_scat that correctly encodes
every propagating and evanescent order.

The loss is then MSE( H_scat - H_DtN ) over the boundary points.

Since the PINN may not have a dense uniform x grid we sample N_dtn equally
spaced points explicitly (separate from interior collocation points).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# kz branch for outgoing scattered modes
# ---------------------------------------------------------------------------


def _kz_outgoing(kx_m: np.ndarray, n: float, k0: float) -> np.ndarray:
    """Outgoing kz for each Bloch order at normal incidence.

    Convention: outgoing in +z direction → exp(-i kz_m z) decays/propagates +z.
    Branch:
      - Propagating (kz² > 0): kz real, positive.
      - Evanescent  (kz² < 0): kz = -i |β|  so exp(-i kz_m z) = exp(-|β|z) decays.
    """
    kz2 = (n * k0)**2 - kx_m**2
    kz  = np.sqrt(kz2.astype(complex))
    evan = kz2 < -1e-14
    # numpy sqrt gives Im >= 0; for evanescent we want Im < 0 → flip
    kz = np.where(evan, -kz, kz)
    # Ensure Re >= 0 for propagating
    kz = np.where(kz.real < 0, -kz, kz)
    return kz


def _kz_outgoing_upward(kx_m: np.ndarray, n: float, k0: float) -> np.ndarray:
    """Outgoing kz in −z direction (upward, reflected orders at top).

    exp(+i kz_m z) decays as z → -∞ when Im(kz_m) < 0, i.e. same branch as +z.
    The sign of the DtN relation flips: H̃_x = +(kz_m / k0) · E_m.
    """
    return _kz_outgoing(kx_m, n, k0)


# ---------------------------------------------------------------------------
# Modal DtN loss (torch, differentiable)
# ---------------------------------------------------------------------------


def modal_dtn_loss(
    subnet,
    x_pts: torch.Tensor,
    z_val: float,
    physics: "PhysicsConfig",
    boundary: str,              # "top" or "bottom"
    n_medium: float,
    n_dtn_orders: int = 10,
) -> torch.Tensor:
    """DtN boundary loss for the scattered field at a horizontal boundary.

    Computes pointwise DtN target H̃_x^DtN(x) and returns MSE vs PINN H̃_x.

    Parameters
    ----------
    subnet
        The PINN subnet (net_air for top, net_sub for bottom).
    x_pts
        Tensor of x points, shape (N,).  Need NOT be uniform — DFT uses
        the ordering to reconstruct phases, but we sum over all m.
    z_val
        Physical z coordinate of the boundary.
    physics
        PhysicsConfig.
    boundary
        "top":    H̃_x_scat = +(kz_m / k0) · E_scat_m   (upward outgoing)
        "bottom": H̃_x_scat = -(kz_m / k0) · E_scat_m   (downward outgoing)
    n_medium
        Refractive index of the half-space at this boundary.
    n_dtn_orders
        Number of grating orders each side (total = 2*n_dtn_orders + 1).

    Returns
    -------
    torch.Tensor (scalar)
        Mean squared boundary residual summed over all DtN orders.
    """
    k0     = physics.k0
    period = physics.period
    sign   = +1.0 if boundary == "top" else -1.0

    # Use uniform x grid for DFT accuracy
    N_x    = max(len(x_pts), 4 * n_dtn_orders + 4)
    x_uni  = torch.linspace(0.0, period, N_x + 1, dtype=x_pts.dtype, device=x_pts.device)[:-1]
    z_uni  = torch.full_like(x_uni, z_val)

    # Evaluate PINN at uniform boundary points
    Er_s, Ei_s, Hr_x, Hi_x, _, _ = subnet.field_components(x_uni, z_uni)
    E_scat_c = Er_s + 1j * Ei_s          # (N_x,) complex — but torch complex

    # Build Bloch wavenumbers and kz
    orders = np.arange(-n_dtn_orders, n_dtn_orders + 1)
    G0     = 2.0 * np.pi / period
    kx_m   = k0 * np.sin(0.0) + orders * G0   # normal incidence: kx_inc = 0
    kz_m   = _kz_outgoing(kx_m, n_medium, k0)
    kz_over_k0 = (kz_m / k0).astype(complex)   # complex numpy

    # DFT: E_m = (1/N_x) Σ_j  E(x_j) * exp(-i G_m x_j)
    x_np    = x_uni.detach().cpu().numpy()
    dx      = period / N_x
    G_m_np  = kx_m  # already the full kx_m (not just G offsets) but for DFT we need just Gm
    # Actually kx_m = m * G0 for normal incidence, so DFT basis is exp(i m G0 x)
    # The DFT coefficient E_m = (1/period) int_0^period E(x) exp(-i m G0 x) dx
    # Numerically: E_m = (dx/period) Σ_j E(x_j) exp(-i m G0 x_j)

    # Compute DFT in torch for differentiability
    # For each order m, E_m = sum_j E(x_j) * exp(-i m G0 x_j) * dx/period
    E_scat_c_re = Er_s   # (N_x,)
    E_scat_c_im = Ei_s   # (N_x,)
    H_scat_c_re = Hr_x
    H_scat_c_im = Hi_x

    # Build DtN target H̃_x^DtN pointwise in x-space via IDFT(kz_m/k0 * DFT(E_m))
    # H_DtN(x) = Σ_m  (sign * kz_m/k0) * E_m * exp(i m G0 x)
    # where E_m is the DFT coefficient of E_scat

    loss = torch.zeros(1, dtype=x_pts.dtype, device=x_pts.device)
    H_dtn_re = torch.zeros(N_x, dtype=x_pts.dtype, device=x_pts.device)
    H_dtn_im = torch.zeros(N_x, dtype=x_pts.dtype, device=x_pts.device)

    for i_m, m in enumerate(orders):
        gm      = float(m) * G0
        # DFT coefficient (differentiable through E_scat)
        phase_j = gm * x_uni  # (N_x,)
        cos_j   = torch.cos(phase_j)
        sin_j   = torch.sin(phase_j)

        # E_m = (dx/period) Σ_j (Er+iEi)(x_j) * (cos - i sin)(phase_j)
        Em_re   = (dx / period) * torch.sum(E_scat_c_re * cos_j + E_scat_c_im * sin_j)
        Em_im   = (dx / period) * torch.sum(E_scat_c_im * cos_j - E_scat_c_re * sin_j)

        # kz/k0 coefficient (complex, numpy)
        c_re    = float(sign * kz_over_k0[i_m].real)
        c_im    = float(sign * kz_over_k0[i_m].imag)

        # H_m^DtN = (kz_m/k0) * E_m  [complex multiply]
        Hm_dtn_re = c_re * Em_re - c_im * Em_im
        Hm_dtn_im = c_re * Em_im + c_im * Em_re

        # IDFT: add contribution to H_dtn(x) = Σ_m H_m * exp(i m G0 x)
        H_dtn_re = H_dtn_re + Hm_dtn_re * cos_j - Hm_dtn_im * sin_j
        H_dtn_im = H_dtn_im + Hm_dtn_re * sin_j + Hm_dtn_im * cos_j

    loss = torch.mean((H_scat_c_re - H_dtn_re)**2 + (H_scat_c_im - H_dtn_im)**2)
    return loss


def modal_dtn_loss_lbg(
    subnet,
    x_pts: torch.Tensor,
    z_val: float,
    physics: "PhysicsConfig",
    coeff: dict,
    boundary: str,
    n_medium: float,
    n_dtn_orders: int = 10,
) -> torch.Tensor:
    """DtN loss for layered-background scattered field.

    The background already satisfies its own BC.  We apply DtN only to the
    scattered field E_scat and H_scat output by the PINN subnet.

    Background H at boundary is already accounted for:
      H_total = H_bg + H_scat
      H_bg satisfies flat-interface condition exactly.
      So only H_scat needs the DtN correction.

    This function is identical to modal_dtn_loss since the subnet already
    outputs scattered fields only.
    """
    return modal_dtn_loss(subnet, x_pts, z_val, physics, boundary, n_medium, n_dtn_orders)


# ---------------------------------------------------------------------------
# Diagnostic: boundary spectral audit
# ---------------------------------------------------------------------------


def boundary_spectral_audit(
    subnet,
    z_val: float,
    physics: "PhysicsConfig",
    boundary: str,
    n_medium: float,
    n_dtn_orders: int = 10,
    N_x: int = 256,
) -> dict:
    """Compute and return spectral diagnostics at a boundary.

    For each Fourier order m:
      - |E_scat_m|, phase
      - |H_scat_m|, phase
      - Expected |H_m^DtN| = |kz_m/k0| * |E_scat_m|
      - Actual |H_scat_m|
      - Relative residual  ||H_m - H_m^DtN|| / |H_m^DtN|

    Returns dict with arrays indexed by order.
    """
    k0     = physics.k0
    period = physics.period
    sign   = +1.0 if boundary == "top" else -1.0

    x_uni  = np.linspace(0, period, N_x, endpoint=False)
    z_uni  = np.full(N_x, z_val)

    x_t = torch.as_tensor(x_uni, dtype=torch.float64)
    z_t = torch.as_tensor(z_uni, dtype=torch.float64)

    # Some representations reconstruct H from autograd derivatives of E.
    # Keep this local graph during an audit, then detach below for NumPy.
    with torch.enable_grad():
        Er_s, Ei_s, Hr_x, Hi_x, _, _ = subnet.field_components(x_t, z_t)
    E_scat = Er_s.detach().numpy() + 1j * Ei_s.detach().numpy()
    H_scat = Hr_x.detach().numpy() + 1j * Hi_x.detach().numpy()

    orders = np.arange(-n_dtn_orders, n_dtn_orders + 1)
    G0     = 2.0 * np.pi / period
    kx_m   = orders * G0
    kz_m   = _kz_outgoing(kx_m, n_medium, k0)
    kz_k0  = kz_m / k0

    dx = period / N_x
    rows = []
    for i_m, m in enumerate(orders):
        gm   = float(m) * G0
        kern = np.exp(-1j * gm * x_uni) * (dx / period)
        Em   = np.sum(E_scat * kern)
        Hm   = np.sum(H_scat * kern)
        Hm_dtn = sign * kz_k0[i_m] * Em
        res  = abs(Hm - Hm_dtn) / (abs(Hm_dtn) + 1e-12)
        rows.append({
            "m": int(m),
            "kx_m": float(kx_m[i_m]),
            "kz_m_re": float(kz_m[i_m].real),
            "kz_m_im": float(kz_m[i_m].imag),
            "propagating": bool(kz_m[i_m].real > 1e-6),
            "E_m_abs": float(abs(Em)),
            "E_m_phase_deg": float(np.angle(Em) * 180 / np.pi),
            "H_m_abs": float(abs(Hm)),
            "H_m_phase_deg": float(np.angle(Hm) * 180 / np.pi),
            "H_m_dtn_abs": float(abs(Hm_dtn)),
            "H_m_dtn_phase_deg": float(np.angle(Hm_dtn) * 180 / np.pi),
            "relative_residual": float(res),
        })

    return {
        "boundary": boundary,
        "z_val": float(z_val),
        "n_medium": float(n_medium),
        "orders": rows,
        "propagating_orders": [r for r in rows if r["propagating"]],
        "summary": {
            "max_relative_residual_propagating": max(
                (r["relative_residual"] for r in rows if r["propagating"]), default=0.0
            ),
            "mean_relative_residual_propagating": float(np.mean(
                [r["relative_residual"] for r in rows if r["propagating"]]
            )) if any(r["propagating"] for r in rows) else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Pure-mode test functions
# ---------------------------------------------------------------------------


def make_single_mode_field(
    m: int,
    A: complex,
    x_pts: np.ndarray,
    z_val: float,
    physics: "PhysicsConfig",
    n_medium: float,
    direction: str = "down",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (E_scat, H_scat) for a single outgoing diffraction order.

    E(x, z) = A * exp(i kx_m x) * exp(-i kz_m z)   [direction=down]
    H̃_x = -(kz_m / k0) * E                          [outgoing down]

    Returns (E_complex, H_complex) at the given x_pts, z=z_val.
    """
    k0     = physics.k0
    G0     = 2.0 * np.pi / physics.period
    kx_m   = m * G0
    kz_m   = _kz_outgoing(np.array([kx_m]), n_medium, k0)[0]
    sign   = -1.0 if direction == "down" else +1.0

    phase  = kx_m * x_pts - kz_m * z_val
    E_c    = A * np.exp(1j * phase)
    H_c    = sign * (kz_m / k0) * E_c

    return E_c, H_c


def check_dtn_residual_single_mode(
    m: int,
    A: complex,
    physics: "PhysicsConfig",
    n_medium: float,
    boundary: str,
    N_x: int = 128,
    n_dtn_orders: int = 5,
) -> dict:
    """Verify DtN residual for a known single-mode field (analytical test).

    The DtN loss should be exactly zero for a field that is already a pure
    outgoing mode.  Returns the computed DtN residual and related diagnostics.
    """
    k0     = physics.k0
    period = physics.period
    G0     = 2.0 * np.pi / period
    kx_m_val = m * G0
    kz_m_val = _kz_outgoing(np.array([kx_m_val]), n_medium, k0)[0]
    sign     = +1.0 if boundary == "top" else -1.0
    z_val    = 0.0 if boundary == "top" else physics.domain_height

    x_np  = np.linspace(0, period, N_x, endpoint=False)
    E_c, H_c = make_single_mode_field(m, A, x_np, z_val, physics, n_medium,
                                       direction="up" if boundary == "top" else "down")

    # Manually apply DFT-DtN and compute residual
    dx = period / N_x
    orders_dtn = np.arange(-n_dtn_orders, n_dtn_orders + 1)
    kx_all = orders_dtn * G0
    kz_all = _kz_outgoing(kx_all, n_medium, k0)
    kz_k0  = kz_all / k0

    H_dtn = np.zeros(N_x, dtype=complex)
    for i_o, mo in enumerate(orders_dtn):
        gm   = float(mo) * G0
        kern = np.exp(-1j * gm * x_np) * (dx / period)
        Em   = np.sum(E_c * kern)
        Hm_dtn = sign * kz_k0[i_o] * Em
        H_dtn += Hm_dtn * np.exp(1j * gm * x_np)

    residual_abs = np.max(np.abs(H_c - H_dtn))
    residual_rel = float(residual_abs / (np.max(np.abs(H_c)) + 1e-12))

    return {
        "m": m,
        "boundary": boundary,
        "n_medium": n_medium,
        "kz_m": complex(kz_m_val),
        "propagating": bool(kz_m_val.real > 1e-6),
        "max_abs_residual": float(residual_abs),
        "relative_residual": float(residual_rel),
        "passed": residual_rel < 1e-6,
    }
