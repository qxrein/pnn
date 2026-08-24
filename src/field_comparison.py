"""Dual total-field / scattered-field comparison between PINN and RCWA.

Background
----------
The LBG PINN predicts **scattered** fields:
    E_total_pinn = E_bg + E_scat_pinn
    H_total_pinn = H_bg + H_scat_pinn

The RCWA reference contains the **total** field:
    field_representation = "total"

Canonical extraction APIs
--------------------------
Two functions handle the two possible input representations.  Never mix them.

``extract_total_modal_amplitudes(E_total_2d, ..., formulation)``
    Input is the total field E_total = E_bg + E_scat.
    At the top monitor the incident field is subtracted to isolate the
    *total* reflection.  This deliberately retains the flat-background
    reflection in the layered-background formulation:
        E_refl,total = E_total(z_top) - E_inc(z_top)
    At the bottom monitor the total field carries only outgoing waves:
        t_m = DFT[E_total(z_bot)]

``extract_scattered_modal_amplitudes(E_scat_2d, ..., formulation, coeff)``
    Input is the scattered field only (E_total - E_bg).
    At the top monitor the scattered field already has the background removed:
        r_m_scat = DFT[E_scat(z_top)]   (no further subtraction)
    At the bottom monitor:
        t_m_scat = DFT[E_scat(z_bot)]

The public legacy function ``extract_modal_amplitudes`` is a wrapper that now
requires an explicit ``formulation`` argument and delegates to the canonical API.

Region masks
------------
``compare_fields`` accepts an optional ``region_mask`` parameter:
    "air"           — z < ridge_base_z only
    "grating"       — ridge_z_min <= z <= ridge_z_max
    "substrate"     — z > ridge_z_max only
    "external_only" — air + substrate (excludes grating interior)
    "full_domain"   — all points (default; raises warning if grating interior
                      has non-trivial E_scat since RCWA total field there is not
                      directly comparable to a scattered-field PINN output)

If ``region_mask`` is not "full_domain", only the masked points contribute to
metrics and field arrays are masked in the returned dict.

Metadata
--------
Every metric dict returned by compare_fields, extract_total_modal_amplitudes,
and extract_scattered_modal_amplitudes includes:
    pinn_representation:   "total" or "scattered"
    reference_representation: "total" or "scattered"
    formulation:           "free_space" or "layered_bg"
    background_added:      bool
    monitor_plane_top_z:   float
    monitor_plane_bot_z:   float
    reference_plane_refl_z: float (always 0.0 — c_refl defined at z=0)
    reference_plane_trans_z: float (ridge_z_max — c_trans defined there)
    deembedding_applied:   bool
    region_mask:           str

Conventions
-----------
- E_inc(z) = exp(-ik0 z),  time convention exp(+iωt) suppressed
- Background = E_bg from maxwell_layered_bg.compute_background_coefficients
- z=0 top, z increases downward
- Only propagating orders (Re kz > 0) contribute to power
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# Background field helpers (numpy)
# ---------------------------------------------------------------------------


def _incident_field_np(z: np.ndarray, k0: float) -> tuple[np.ndarray, np.ndarray]:
    """E_inc = exp(-ik0 z).  Returns (E_inc_r, E_inc_i)."""
    return np.cos(k0 * z), -np.sin(k0 * z)


def _background_field_np(z: np.ndarray, physics: "PhysicsConfig") -> tuple[np.ndarray, np.ndarray]:
    """Layered background field (flat substrate, no ridge).

    Returns (Ebg_r, Ebg_i).
    """
    from src.maxwell_layered_bg import background_field_np, compute_background_coefficients
    coeff = compute_background_coefficients(physics)
    Ebg_r, Ebg_i, _, _ = background_field_np(z, coeff)
    return Ebg_r, Ebg_i


def _background_coeff(physics: "PhysicsConfig") -> dict:
    from src.maxwell_layered_bg import compute_background_coefficients
    return compute_background_coefficients(physics)


# ---------------------------------------------------------------------------
# kz branch (matches generate_reference.py and modal_dtn.py)
# ---------------------------------------------------------------------------

def _kz_branch_array(kx_m: np.ndarray, n: float, k0: float) -> np.ndarray:
    """Outgoing kz branch: Re>=0 for propagating, Im<0 for evanescent (decays +z)."""
    kz2 = (n * k0) ** 2 - kx_m ** 2
    kz = np.sqrt(kz2.astype(complex))
    evan = kz.real < 1e-6
    kz[evan] = -1j * np.abs(kz[evan])
    return kz


# ---------------------------------------------------------------------------
# DFT helper
# ---------------------------------------------------------------------------

def _dft_amplitudes(E_slice: np.ndarray, x1d: np.ndarray,
                    G_m: np.ndarray, period: float) -> np.ndarray:
    """Modal amplitudes A_m = (1/Λ) ∫₀^Λ E(x) exp(-i G_m x) dx.

    Parameters
    ----------
    E_slice : (Nx,) complex
    x1d     : (Nx,) float, uniform in [0, Λ)
    G_m     : (M,) float, Bloch wavenumbers
    period  : float

    Returns (M,) complex amplitudes.
    """
    dx = period / len(x1d)
    amps = np.zeros(len(G_m), dtype=complex)
    for mi, gm in enumerate(G_m):
        amps[mi] = np.sum(E_slice * np.exp(-1j * gm * x1d)) * dx / period
    return amps


# ---------------------------------------------------------------------------
# Power efficiency helper
# ---------------------------------------------------------------------------

def _modal_power(r_m: np.ndarray, t_m: np.ndarray,
                 kz_air_m: np.ndarray, kz_sub_m: np.ndarray,
                 k0: float, n_orders_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (R_m, T_m, P_inc) arrays."""
    kz_inc = kz_air_m[n_orders_idx]
    P_inc  = 0.5 * kz_inc.real / k0
    R_m = np.zeros(len(r_m))
    T_m = np.zeros(len(t_m))
    for mi in range(len(r_m)):
        if kz_air_m[mi].real > 1e-6:
            R_m[mi] = (0.5 * kz_air_m[mi].real / k0 * abs(r_m[mi]) ** 2
                       / (P_inc + 1e-30))
        if kz_sub_m[mi].real > 1e-6:
            T_m[mi] = (0.5 * kz_sub_m[mi].real / k0 * abs(t_m[mi]) ** 2
                       / (P_inc + 1e-30))
    return R_m, T_m, float(P_inc)


# ---------------------------------------------------------------------------
# Canonical API 1: extract from TOTAL field
# ---------------------------------------------------------------------------

def extract_total_modal_amplitudes(
    E_total_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: "PhysicsConfig",
    formulation: str = "layered_bg",
    n_orders: int = 5,
    z_top_frac: float = 0.08,
    z_bot_frac: float = 0.92,
) -> dict:
    """Extract complex modal amplitudes r_m and t_m from E_total.

    This is the canonical path for any field that represents the complete
    electromagnetic field (E_bg + E_scat_pinn, or pure RCWA total field).

    Parameters
    ----------
    E_total_2d : complex (Nz, Nx)
        Total field = E_bg + E_scat_pinn (for PINN) or RCWA total.
    x1d, z1d : grid coordinates
    physics : PhysicsConfig
    formulation : "free_space" | "layered_bg"
        Recorded as provenance.  Total-field extraction always subtracts
        only E_inc at the top monitor, so r_m includes the background
        reflection when the formulation is ``layered_bg``.
    n_orders, z_top_frac, z_bot_frac : monitor geometry

    Returns
    -------
    dict with all modal amplitudes, powers, energy check, and metadata.
    """
    k0     = physics.k0
    n_air  = physics.n_air
    n_sub  = physics.n_substrate
    period = physics.period

    orders = np.arange(-n_orders, n_orders + 1)
    G_m    = orders * (2.0 * np.pi / period)

    # Monitor planes
    z_top = z1d[np.argmin(np.abs(z1d - z_top_frac * physics.domain_height))]
    z_bot = z1d[np.argmin(np.abs(z1d - z_bot_frac * physics.domain_height))]
    iz_top = int(np.argmin(np.abs(z1d - z_top)))
    iz_bot = int(np.argmin(np.abs(z1d - z_bot)))

    kz_air_m = _kz_branch_array(G_m, n_air, k0)
    kz_sub_m = _kz_branch_array(G_m, n_sub, k0)
    idx0     = n_orders  # m=0 index

    if formulation not in {"free_space", "layered_bg"}:
        raise ValueError("formulation must be 'free_space' or 'layered_bg'")

    # Total field: remove only the incident wave.  In particular, do not
    # subtract r_bg here: r_total = r_bg + r_scat for layered-background PINNs.
    E_inc_top = np.exp(-1j * k0 * z_top)
    E_refl_slice = E_total_2d[iz_top, :] - E_inc_top

    # Transmitted: total at bottom (all outgoing)
    E_trans_slice = E_total_2d[iz_bot, :]

    r_m = _dft_amplitudes(E_refl_slice,  x1d, G_m, period)
    t_m = _dft_amplitudes(E_trans_slice, x1d, G_m, period)

    R_m, T_m, P_inc = _modal_power(r_m, t_m, kz_air_m, kz_sub_m, k0, idx0)
    R_total = float(R_m.sum())
    T_total = float(T_m.sum())

    return {
        # Amplitudes
        "orders":           orders.tolist(),
        "r_m_complex":      r_m,
        "t_m_complex":      t_m,
        "r_m_abs":          np.abs(r_m).tolist(),
        "t_m_abs":          np.abs(t_m).tolist(),
        "r_m_phase_deg":    (np.angle(r_m) * 180 / np.pi).tolist(),
        "t_m_phase_deg":    (np.angle(t_m) * 180 / np.pi).tolist(),
        # Power
        "R_m":              R_m.tolist(),
        "T_m":              T_m.tolist(),
        "R0":               float(R_m[idx0]),
        "T0":               float(T_m[idx0]),
        "r0_complex":       complex(r_m[idx0]),
        "t0_complex":       complex(t_m[idx0]),
        "R_total":          R_total,
        "T_total":          T_total,
        "energy_check":     R_total + T_total,
        "P_inc":            P_inc,
        # Monitor geometry
        "z_top_monitor":    float(z_top),
        "z_bot_monitor":    float(z_bot),
        # Metadata
        "pinn_representation":       "total",
        "reference_representation":  "total",
        "formulation":               formulation,
        "background_added":          False,
        "background_subtracted_top": "incident_only",
        "reference_plane_refl_z":    0.0,
        "reference_plane_trans_z":   float(physics.ridge_z_max),
        "deembedding_applied":       False,
    }


# ---------------------------------------------------------------------------
# Canonical API 2: extract from SCATTERED field
# ---------------------------------------------------------------------------

def extract_scattered_modal_amplitudes(
    E_scat_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: "PhysicsConfig",
    formulation: str = "layered_bg",
    n_orders: int = 5,
    z_top_frac: float = 0.08,
    z_bot_frac: float = 0.92,
) -> dict:
    """Extract scattered modal amplitudes from E_scat = E_total - E_bg.

    The scattered field already has the background removed.  At the top
    monitor, NO additional subtraction is performed — the DFT of E_scat
    at z_top directly yields the scattered reflected amplitudes.
    At the bottom monitor, the DFT of E_scat yields scattered transmission
    amplitudes (background subtracted for m=0 by definition).

    Power for the scattered field alone is NOT required to equal 1.  Only
    the total-field power must satisfy R+T = 1.

    Parameters
    ----------
    E_scat_2d : complex (Nz, Nx)
        Scattered field = E_total - E_bg.
    x1d, z1d, physics, formulation, n_orders, z_top_frac, z_bot_frac
        Same as extract_total_modal_amplitudes.

    Returns
    -------
    dict with scattered modal amplitudes, powers, and metadata.
    Note: R_scat + T_scat ≠ 1 in general.
    """
    k0     = physics.k0
    n_air  = physics.n_air
    n_sub  = physics.n_substrate
    period = physics.period

    orders = np.arange(-n_orders, n_orders + 1)
    G_m    = orders * (2.0 * np.pi / period)

    z_top = z1d[np.argmin(np.abs(z1d - z_top_frac * physics.domain_height))]
    z_bot = z1d[np.argmin(np.abs(z1d - z_bot_frac * physics.domain_height))]
    iz_top = int(np.argmin(np.abs(z1d - z_top)))
    iz_bot = int(np.argmin(np.abs(z1d - z_bot)))

    kz_air_m = _kz_branch_array(G_m, n_air, k0)
    kz_sub_m = _kz_branch_array(G_m, n_sub, k0)
    idx0     = n_orders

    # No subtraction: E_scat already has background removed
    r_m_scat = _dft_amplitudes(E_scat_2d[iz_top, :], x1d, G_m, period)
    t_m_scat = _dft_amplitudes(E_scat_2d[iz_bot, :], x1d, G_m, period)

    # Power uses the same formula but normalised by the same P_inc
    R_m, T_m, P_inc = _modal_power(r_m_scat, t_m_scat,
                                    kz_air_m, kz_sub_m, k0, idx0)
    R_total = float(R_m.sum())
    T_total = float(T_m.sum())

    return {
        "orders":           orders.tolist(),
        "r_m_complex":      r_m_scat,
        "t_m_complex":      t_m_scat,
        "r_m_abs":          np.abs(r_m_scat).tolist(),
        "t_m_abs":          np.abs(t_m_scat).tolist(),
        "r_m_phase_deg":    (np.angle(r_m_scat) * 180 / np.pi).tolist(),
        "t_m_phase_deg":    (np.angle(t_m_scat) * 180 / np.pi).tolist(),
        "R_m":              R_m.tolist(),
        "T_m":              T_m.tolist(),
        "R0":               float(R_m[idx0]),
        "T0":               float(T_m[idx0]),
        "r0_complex":       complex(r_m_scat[idx0]),
        "t0_complex":       complex(t_m_scat[idx0]),
        "R_total":          R_total,
        "T_total":          T_total,
        "energy_check":     R_total + T_total,
        "energy_check_note": (
            "R_scat+T_scat != 1 by design: scattered field carries only "
            "grating-scattered power. Energy conservation is checked on "
            "the total field."
        ),
        "P_inc":            P_inc,
        "z_top_monitor":    float(z_top),
        "z_bot_monitor":    float(z_bot),
        # Metadata
        "pinn_representation":       "scattered",
        "reference_representation":  "scattered",
        "formulation":               formulation,
        "background_added":          False,
        "background_subtracted_top": "none — E_scat has no background to subtract",
        "reference_plane_refl_z":    0.0,
        "reference_plane_trans_z":   float(physics.ridge_z_max),
        "deembedding_applied":       False,
    }


def reconstruct_total_modal_amplitudes(
    scattered_modal: dict,
    physics: "PhysicsConfig",
    formulation: str | None = None,
) -> dict:
    """Convert monitor-plane scattered amplitudes to total amplitudes.

    The background is x-independent, so it affects m=0 only.  At the top it
    contributes the reflected background; at the bottom it contributes the
    transmitted background.  This is the inverse of the representation split
    used by :func:`extract_scattered_modal_amplitudes`.
    """
    formulation = formulation or scattered_modal.get("formulation", "layered_bg")
    if formulation not in {"free_space", "layered_bg"}:
        raise ValueError("formulation must be 'free_space' or 'layered_bg'")

    orders = np.asarray(scattered_modal["orders"], dtype=int)
    idx0 = int(np.where(orders == 0)[0][0])
    r_total = np.asarray(scattered_modal["r_m_complex"], dtype=complex).copy()
    t_total = np.asarray(scattered_modal["t_m_complex"], dtype=complex).copy()
    z_top = float(scattered_modal["z_top_monitor"])
    z_bot = float(scattered_modal["z_bot_monitor"])

    if formulation == "layered_bg":
        coeff = _background_coeff(physics)
        r_bg = coeff["r_eff"] * np.exp(1j * coeff["k1"] * z_top)
        t_bg = coeff["tau"] * np.exp(-1j * coeff["k2"] * (z_bot - coeff["z_interface"]))
    else:
        r_bg = 0.0j
        t_bg = np.exp(-1j * physics.k0 * z_bot)

    r_total[idx0] += r_bg
    t_total[idx0] += t_bg
    n_orders = int((len(orders) - 1) // 2)
    kz_air = _kz_branch_array(orders * (2 * np.pi / physics.period), physics.n_air, physics.k0)
    kz_sub = _kz_branch_array(orders * (2 * np.pi / physics.period), physics.n_substrate, physics.k0)
    R_m, T_m, P_inc = _modal_power(r_total, t_total, kz_air, kz_sub, physics.k0, n_orders)
    return {
        **scattered_modal,
        "r_m_complex": r_total,
        "t_m_complex": t_total,
        "r_m_abs": np.abs(r_total).tolist(),
        "t_m_abs": np.abs(t_total).tolist(),
        "r_m_phase_deg": (np.angle(r_total) * 180 / np.pi).tolist(),
        "t_m_phase_deg": (np.angle(t_total) * 180 / np.pi).tolist(),
        "r0_complex": complex(r_total[idx0]),
        "t0_complex": complex(t_total[idx0]),
        "R_m": R_m.tolist(), "T_m": T_m.tolist(),
        "R0": float(R_m[idx0]), "T0": float(T_m[idx0]),
        "R_total": float(R_m.sum()), "T_total": float(T_m.sum()),
        "energy_check": float(R_m.sum() + T_m.sum()), "P_inc": P_inc,
        "pinn_representation": "total",
        "reference_representation": "total",
        "background_added": True,
        "background_reconstructed": True,
        "background_reflection_m0": complex(r_bg),
        "background_transmission_m0": complex(t_bg),
        "formulation": formulation,
    }


# ---------------------------------------------------------------------------
# Legacy public API — now a wrapper around extract_total_modal_amplitudes
# ---------------------------------------------------------------------------

def extract_modal_amplitudes(
    E_field_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: "PhysicsConfig",
    n_orders: int = 5,
    z_top_frac: float = 0.08,
    z_bot_frac: float = 0.92,
    formulation: str = "layered_bg",
    field_representation: str = "total",
) -> dict:
    """Extract complex modal amplitudes r_m and t_m from E_total.

    Legacy wrapper around ``extract_total_modal_amplitudes``.

    IMPORTANT: ``formulation`` now defaults to "layered_bg".  For the
    free-space PINN pass ``formulation="free_space"`` explicitly.

    The background subtracted at the top monitor depends on ``formulation``:
        "layered_bg" : E_bg(z_top) = E_inc + r_eff*exp(+ik1*z_top)   (CORRECT)
        "free_space" : E_inc(z_top) = exp(-ik0*z_top)

    Parameters
    ----------
    E_total_2d : complex (Nz, Nx)
        Total field (E_bg + E_scat_pinn) on the visualization grid.
    x1d : (Nx,)
    z1d : (Nz,)
    physics : PhysicsConfig
    n_orders : number of grating orders each side of zeroth (±n_orders)
    z_top_frac, z_bot_frac : monitor plane positions as fraction of domain_height
    formulation : "layered_bg" | "free_space"

    Returns
    -------
    Same dict as extract_total_modal_amplitudes (superset of legacy keys).
    """
    if field_representation == "total":
        return extract_total_modal_amplitudes(
            E_field_2d, x1d, z1d, physics, formulation=formulation,
            n_orders=n_orders, z_top_frac=z_top_frac, z_bot_frac=z_bot_frac,
        )
    if field_representation == "scattered":
        return extract_scattered_modal_amplitudes(
            E_field_2d, x1d, z1d, physics, formulation=formulation,
            n_orders=n_orders, z_top_frac=z_top_frac, z_bot_frac=z_bot_frac,
        )
    raise ValueError("field_representation must be 'total' or 'scattered'")


# ---------------------------------------------------------------------------
# Region mask helper
# ---------------------------------------------------------------------------

def _region_mask(z_grid: np.ndarray, physics: "PhysicsConfig",
                 region_mask: str) -> np.ndarray:
    """Boolean mask selecting points in the requested region.

    Parameters
    ----------
    z_grid : (Nz,) or (Nz, Nx) — physical z coordinates
    physics : PhysicsConfig
    region_mask : one of
        "full_domain"   — all True
        "air"           — z < ridge_base_z
        "grating"       — ridge_z_min <= z <= ridge_z_max
        "substrate"     — z > ridge_z_max
        "external_only" — air + substrate (excludes grating interior)

    Returns boolean ndarray of same shape as z_grid.
    """
    z = np.asarray(z_grid)
    if z.ndim not in (1, 2):
        raise ValueError("z_grid must be one- or two-dimensional")
    rbase  = physics.ridge_base_z      # = ridge_z_min for this geometry
    rmax   = physics.ridge_z_max

    if region_mask == "full_domain":
        mask = np.ones(z.shape, dtype=bool)
    elif region_mask == "air":
        mask = z < rbase
    elif region_mask == "grating":
        mask = (z >= rbase) & (z <= rmax)
    elif region_mask == "substrate":
        mask = z > rmax
    elif region_mask == "external_only":
        mask = (z < rbase) | (z > rmax)
    else:
        raise ValueError(
            f"Unknown region_mask '{region_mask}'. "
            "Choose: full_domain, air, grating, substrate, external_only."
        )

    assert mask.shape == z.shape, (mask.shape, z.shape)
    return mask


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_fields(
    pinn_E_scat_r: np.ndarray,
    pinn_E_scat_i: np.ndarray,
    rcwa_E_total_r: np.ndarray,
    rcwa_E_total_i: np.ndarray,
    z_grid: np.ndarray,
    physics: "PhysicsConfig",
    formulation: str = "layered_bg",
    region_mask: str = "external_only",
    field_representation: str = "scattered",
    reference_field_representation: str = "total",
) -> dict:
    """Dual comparison of PINN vs RCWA in both total and scattered representations.

    Valid usage
    -----------
    PINN scattered field vs RCWA total field.
    Never pass a PINN scattered field as if it were total (or vice versa).

    Parameters
    ----------
    pinn_E_scat_r, pinn_E_scat_i : (Nz, Nx)
        Scattered-field output of the PINN.
        For "free_space": E_scat relative to E_inc.
        For "layered_bg": E_scat relative to E_bg.
    rcwa_E_total_r, rcwa_E_total_i : (Nz, Nx)
        Total field from RCWA.
    z_grid : (Nz, Nx) or (Nz,)
    physics : PhysicsConfig
    formulation : "free_space" | "layered_bg"
    region_mask : "full_domain" | "air" | "grating" | "substrate" | "external_only"
        Default "external_only" excludes the grating interior (1.2 < z < 1.4),
        where the RCWA total field cannot be directly compared to a scattered
        PINN output without eigenmodes that are not stored in the NPZ.
        Use "full_domain" only when both fields are complete and consistent.

    Returns
    -------
    dict with keys:
        total/*, scattered/*     — metrics for each representation
        pinn_E_total_r/i         — reconstructed PINN total field
        pinn_E_scat_r/i          — PINN scattered field (input)
        rcwa_E_total_r/i         — RCWA total field (input)
        rcwa_E_scat_r/i          — RCWA scattered field (= total - bg)
        metadata/*               — representation labels and provenance
    """
    arrays = tuple(np.asarray(a) for a in (
        pinn_E_scat_r, pinn_E_scat_i, rcwa_E_total_r, rcwa_E_total_i))
    shape = arrays[0].shape
    if not shape or len(shape) > 2:
        raise ValueError("field arrays must be flat (N,) or grid-shaped (Nz, Nx)")
    if any(a.shape != shape for a in arrays[1:]):
        raise ValueError("all field arrays must have exactly the same shape")
    if field_representation not in {"total", "scattered"}:
        raise ValueError("field_representation must be 'total' or 'scattered'")
    if reference_field_representation not in {"total", "scattered"}:
        raise ValueError("reference_field_representation must be 'total' or 'scattered'")
    if formulation not in {"free_space", "layered_bg"}:
        raise ValueError("formulation must be 'free_space' or 'layered_bg'")

    z_values = _z_values_for_fields(np.asarray(z_grid), shape)

    # Background field
    if formulation == "layered_bg":
        Ebg_r_values, Ebg_i_values = _background_field_np(z_values, physics)
    else:
        Ebg_r_values, Ebg_i_values = _incident_field_np(z_values, physics.k0)

    Ebg_r = Ebg_r_values.reshape(shape)
    Ebg_i = Ebg_i_values.reshape(shape)

    pinn_input_r, pinn_input_i, rcwa_input_r, rcwa_input_i = arrays
    if field_representation == "scattered":
        pinn_scat_r, pinn_scat_i = pinn_input_r, pinn_input_i
        pinn_total_r, pinn_total_i = pinn_scat_r + Ebg_r, pinn_scat_i + Ebg_i
    else:
        pinn_total_r, pinn_total_i = pinn_input_r, pinn_input_i
        pinn_scat_r, pinn_scat_i = pinn_total_r - Ebg_r, pinn_total_i - Ebg_i

    if reference_field_representation == "total":
        rcwa_total_r, rcwa_total_i = rcwa_input_r, rcwa_input_i
        rcwa_scat_r, rcwa_scat_i = rcwa_total_r - Ebg_r, rcwa_total_i - Ebg_i
    else:
        rcwa_scat_r, rcwa_scat_i = rcwa_input_r, rcwa_input_i
        rcwa_total_r, rcwa_total_i = rcwa_scat_r + Ebg_r, rcwa_scat_i + Ebg_i

    # Region mask — applied to all metric computations
    active = _region_mask(z_values.reshape(shape), physics, region_mask)
    assert active.shape == shape, ("mask shape must match field shape", active.shape, shape)

    def _metrics(pred_r, pred_i, ref_r, ref_i, label):
        valid = (active
                 & np.isfinite(pred_r) & np.isfinite(pred_i)
                 & np.isfinite(ref_r)  & np.isfinite(ref_i))
        if not valid.any():
            return {
                f"{label}/complex_l2": float("nan"),
                f"{label}/magnitude_l2": float("nan"),
                f"{label}/real_l2": float("nan"),
                f"{label}/imag_l2": float("nan"),
                f"{label}/phase_rmse_deg": float("nan"),
                f"{label}/global_phase_offset_deg": float("nan"),
                f"{label}/n_valid": 0,
            }
        pred_c = pred_r[valid] + 1j * pred_i[valid]
        ref_c  = ref_r[valid]  + 1j * ref_i[valid]
        eps    = 1e-12
        pred_mag = np.abs(pred_c)
        ref_mag  = np.abs(ref_c)

        complex_l2  = float(np.linalg.norm(pred_c - ref_c) /
                            (np.linalg.norm(ref_c) + eps))
        mag_l2      = float(np.linalg.norm(pred_mag - ref_mag) /
                            (np.linalg.norm(ref_mag) + eps))
        real_l2     = float(np.linalg.norm(pred_r[valid] - ref_r[valid]) /
                            (np.linalg.norm(ref_r[valid]) + eps))
        imag_l2     = float(np.linalg.norm(pred_i[valid] - ref_i[valid]) /
                            (np.linalg.norm(ref_i[valid]) + eps))
        phase_err   = np.angle(pred_c / (ref_c + eps))
        phase_rmse  = float(np.sqrt(np.mean(phase_err ** 2)) * 180 / np.pi)
        global_ph   = float(np.angle(np.sum(pred_c * np.conj(ref_c))) * 180 / np.pi)

        return {
            f"{label}/complex_l2":             complex_l2,
            f"{label}/magnitude_l2":           mag_l2,
            f"{label}/real_l2":                real_l2,
            f"{label}/imag_l2":                imag_l2,
            f"{label}/phase_rmse_deg":         phase_rmse,
            f"{label}/global_phase_offset_deg": global_ph,
            f"{label}/n_valid":                int(valid.sum()),
        }

    result = {}
    result.update(_metrics(pinn_total_r, pinn_total_i,
                           rcwa_total_r, rcwa_total_i, "total"))
    result.update(_metrics(pinn_scat_r, pinn_scat_i,
                           rcwa_scat_r,   rcwa_scat_i,   "scattered"))

    # Field arrays
    result["pinn_E_total_r"] = pinn_total_r
    result["pinn_E_total_i"] = pinn_total_i
    result["pinn_E_scat_r"]  = pinn_scat_r
    result["pinn_E_scat_i"]  = pinn_scat_i
    result["rcwa_E_total_r"] = rcwa_total_r
    result["rcwa_E_total_i"] = rcwa_total_i
    result["rcwa_E_scat_r"]  = rcwa_scat_r
    result["rcwa_E_scat_i"]  = rcwa_scat_i

    # Metadata
    result["field_representation_pinn"]  = field_representation
    result["field_representation_rcwa"]  = reference_field_representation
    result["formulation"]                = formulation
    result["background_added"]           = True
    result["region_mask"]                = region_mask
    result["pinn_representation"]        = field_representation
    result["reference_representation"]   = reference_field_representation
    result["monitor_plane_top_z"]        = float(0.08 * physics.domain_height)
    result["monitor_plane_bot_z"]        = float(0.92 * physics.domain_height)
    result["reference_plane_refl_z"]     = 0.0
    result["reference_plane_trans_z"]    = float(physics.ridge_z_max)
    result["deembedding_applied"]        = False
    result["ridge_base_z"]               = float(physics.ridge_base_z)
    result["ridge_z_max"]                = float(physics.ridge_z_max)

    return result


def _z_values_for_fields(z_grid: np.ndarray, field_shape: tuple[int, ...]) -> np.ndarray:
    """Return one z coordinate per field element without implicit broadcasting."""
    if len(field_shape) == 1:
        if z_grid.ndim != 1 or z_grid.shape != field_shape:
            raise ValueError("flat fields require a flat z_grid of identical shape")
        return z_grid

    nz, nx = field_shape
    if z_grid.ndim == 2:
        if z_grid.shape != field_shape:
            raise ValueError("2-D z_grid must have the same (Nz, Nx) shape as fields")
        return z_grid.ravel()
    if z_grid.ndim == 1 and z_grid.shape == (nz,):
        return np.repeat(z_grid, nx)
    raise ValueError("2-D fields require z_grid shaped (Nz, Nx) or (Nz,)")


# ---------------------------------------------------------------------------
# compare_modal_with_rcwa
# ---------------------------------------------------------------------------

def compare_modal_with_rcwa(
    pinn_modal: dict,
    rcwa_path: "str | None",
    n_harmonics_center: int,
    physics: "PhysicsConfig | None" = None,
) -> dict:
    """Compare PINN modal amplitudes against RCWA.

    Parameters
    ----------
    pinn_modal
        Output of extract_total_modal_amplitudes or extract_scattered_modal_amplitudes.
    rcwa_path
        Path to the reference NPZ.
    n_harmonics_center
        N such that the m=0 mode is at index N in the RCWA amplitude arrays.
    physics
        When supplied, reference amplitudes are de-embedded from their RCWA
        planes to the extractor monitor planes and converted to the same field
        representation as ``pinn_modal``.  This is the production path.

    Returns
    -------
    dict: per-order amplitude/phase errors and power efficiency comparison.
    """
    if rcwa_path is None:
        return {"note": "no RCWA reference provided"}

    from pathlib import Path
    data = np.load(Path(rcwa_path), allow_pickle=True)
    if "c_refl" not in data:
        return {"note": "RCWA reference does not contain amplitude data "
                        "(regenerate with scripts/generate_reference.py)"}

    c_refl   = data["c_refl"].astype(complex)
    c_trans  = data["c_trans"].astype(complex)
    R_m_rcwa = data["R_m"].astype(float)
    T_m_rcwa = data["T_m"].astype(float)

    N      = n_harmonics_center
    orders = np.array(pinn_modal["orders"])

    r_pinn = np.array(pinn_modal["r_m_complex"])
    t_pinn = np.array(pinn_modal["t_m_complex"])

    representation = pinn_modal.get("pinn_representation", "total")
    formulation = pinn_modal.get("formulation", "layered_bg")
    if representation not in {"total", "scattered"}:
        raise ValueError("pinn_modal must declare total or scattered representation")
    if physics is not None and formulation not in {"free_space", "layered_bg"}:
        raise ValueError("pinn_modal formulation is invalid")

    z_top = float(pinn_modal.get("z_top_monitor", 0.0))
    z_bot = float(pinn_modal.get("z_bot_monitor", physics.ridge_z_max if physics else 0.0))
    coeff = _background_coeff(physics) if physics and formulation == "layered_bg" else None

    results = {}
    for mi, m in enumerate(orders):
        rcwa_idx = N + m
        if 0 <= rcwa_idx < len(c_refl):
            r_rcwa_m = c_refl[rcwa_idx]
            t_rcwa_m = c_trans[rcwa_idx]
            if physics is not None:
                r_rcwa_m *= np.exp(1j * data["kz_air"][rcwa_idx] * z_top)
                t_rcwa_m *= np.exp(-1j * data["kz_sub"][rcwa_idx]
                                   * (z_bot - physics.ridge_z_max))
                if representation == "scattered" and m == 0:
                    if formulation == "layered_bg":
                        r_rcwa_m -= coeff["r_eff"] * np.exp(1j * coeff["k1"] * z_top)
                        t_rcwa_m -= coeff["tau"] * np.exp(
                            -1j * coeff["k2"] * (z_bot - coeff["z_interface"]))
                    else:
                        t_rcwa_m -= np.exp(-1j * physics.k0 * z_bot)
            r_pinn_m = r_pinn[mi]
            t_pinn_m = t_pinn[mi]
            R_rcwa_m = float(R_m_rcwa[rcwa_idx]) if rcwa_idx < len(R_m_rcwa) else 0.0
            T_rcwa_m = float(T_m_rcwa[rcwa_idx]) if rcwa_idx < len(T_m_rcwa) else 0.0
            R_pinn_m = float(pinn_modal["R_m"][mi])
            T_pinn_m = float(pinn_modal["T_m"][mi])
            results[f"m={m}"] = {
                "r_pinn_abs":       float(abs(r_pinn_m)),
                "r_rcwa_abs":       float(abs(r_rcwa_m)),
                "r_pinn_phase_deg": float(np.angle(r_pinn_m) * 180 / np.pi),
                "r_rcwa_phase_deg": float(np.angle(r_rcwa_m) * 180 / np.pi),
                "r_abs_err":        float(abs(abs(r_pinn_m) - abs(r_rcwa_m))),
                "r_phase_err_deg":  float(abs(
                    np.angle(r_pinn_m / (r_rcwa_m + 1e-30)) * 180 / np.pi)),
                "t_pinn_abs":       float(abs(t_pinn_m)),
                "t_rcwa_abs":       float(abs(t_rcwa_m)),
                "t_pinn_phase_deg": float(np.angle(t_pinn_m) * 180 / np.pi),
                "t_rcwa_phase_deg": float(np.angle(t_rcwa_m) * 180 / np.pi),
                "t_abs_err":        float(abs(abs(t_pinn_m) - abs(t_rcwa_m))),
                "t_phase_err_deg":  float(abs(
                    np.angle(t_pinn_m / (t_rcwa_m + 1e-30)) * 180 / np.pi)),
                "R_pinn": R_pinn_m, "R_rcwa": R_rcwa_m,
                "T_pinn": T_pinn_m, "T_rcwa": T_rcwa_m,
            }

    R_rcwa_total = float(np.sum(R_m_rcwa))
    T_rcwa_total = float(np.sum(T_m_rcwa))
    results["summary"] = {
        "R_pinn_total":      pinn_modal["R_total"],
        "T_pinn_total":      pinn_modal["T_total"],
        "pinn_energy_check": pinn_modal["energy_check"],
        "R_rcwa_total":      R_rcwa_total,
        "T_rcwa_total":      T_rcwa_total,
        "rcwa_energy_check": R_rcwa_total + T_rcwa_total,
        # Provenance: carry through whatever representation the extractor used
        "pinn_representation":    representation,
        "reference_representation": representation if physics is not None else "total_at_reference_planes",
        "formulation":            formulation,
        "background_subtracted_top": pinn_modal.get(
            "background_subtracted_top", "unknown"),
        "deembedding_applied": physics is not None,
    }
    return results


# ---------------------------------------------------------------------------
# Flat-background zero-contrast test
# ---------------------------------------------------------------------------

def flat_contrast_test(
    pinn_E_scat_r: np.ndarray,
    pinn_E_scat_i: np.ndarray,
    rcwa_E_total_r: np.ndarray,
    rcwa_E_total_i: np.ndarray,
    z_grid: np.ndarray,
    physics: "PhysicsConfig",
    formulation: str = "layered_bg",
    tol: float = 0.1,
) -> dict:
    """Zero-contrast test: when ridge contrast = 0, E_scat should be ~0.

    For a flat grating (n_ridge == n_substrate):
        E_scat = 0  everywhere
        E_total = E_bg
        t_m(m≠0) = 0,  r_m(m≠0) = 0

    Pass the PINN scattered field for this test (not the total field).

    Returns pass/fail and max |E_scat|.
    """
    max_scat = float(np.max(np.sqrt(pinn_E_scat_r ** 2 + pinn_E_scat_i ** 2)))
    passed   = max_scat < tol

    z_1d  = z_grid.ravel() if z_grid.ndim == 2 else z_grid
    shape = pinn_E_scat_r.shape
    if formulation == "layered_bg":
        Ebg_r_1d, Ebg_i_1d = _background_field_np(z_1d, physics)
    else:
        Ebg_r_1d, Ebg_i_1d = _incident_field_np(z_1d, physics.k0)
    Ebg_r = Ebg_r_1d.reshape(shape)
    Ebg_i = Ebg_i_1d.reshape(shape)

    E_total_r = pinn_E_scat_r + Ebg_r
    E_total_i = pinn_E_scat_i + Ebg_i
    max_diff_r = float(np.max(np.abs(E_total_r - Ebg_r)))
    max_diff_i = float(np.max(np.abs(E_total_i - Ebg_i)))

    return {
        "max_E_scat_magnitude":  max_scat,
        "max_total_minus_bg_r":  max_diff_r,
        "max_total_minus_bg_i":  max_diff_i,
        "passed":                passed,
        "tolerance":             tol,
        "note": "For zero contrast, E_scat should be ~0 and E_total ~ E_bg",
        # Metadata
        "pinn_representation":        "scattered",
        "formulation":                formulation,
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_comparison_summary(result: dict, rcwa_modal: dict | None = None) -> None:
    """Print a formatted summary of the dual comparison."""
    print("\n" + "=" * 65)
    print("FIELD REPRESENTATION AUDIT")
    print("=" * 65)
    print(f"  PINN formulation:  {result.get('formulation', '?')}")
    print(f"  PINN outputs:      {result.get('field_representation_pinn', '?')}")
    print(f"  RCWA reference:    {result.get('field_representation_rcwa', '?')}")
    print(f"  Region mask:       {result.get('region_mask', '?')}")
    print()

    for label in ("total", "scattered"):
        print(f"  [{label.upper()} FIELD]")
        for metric in ("complex_l2", "magnitude_l2", "real_l2", "imag_l2",
                       "phase_rmse_deg", "global_phase_offset_deg"):
            key = f"{label}/{metric}"
            if key in result:
                print(f"    {metric:35s}: {result[key]:.4f}")
        print()

    if rcwa_modal:
        print("  [MODAL COMPARISON vs RCWA]")
        sm = rcwa_modal.get("summary", {})
        for k, v in sm.items():
            print(f"    {k:35s}: {v:.4f}" if isinstance(v, float)
                  else f"    {k}: {v}")
        print()
        for m_key, m_val in rcwa_modal.items():
            if m_key.startswith("m=") and isinstance(m_val, dict):
                if m_val.get("R_rcwa", 0) > 1e-4 or m_val.get("T_rcwa", 0) > 1e-4:
                    print(f"    {m_key}:")
                    print(f"      r: pinn|{m_val['r_pinn_abs']:.4f}|∠{m_val['r_pinn_phase_deg']:.1f}°  "
                          f"rcwa|{m_val['r_rcwa_abs']:.4f}|∠{m_val['r_rcwa_phase_deg']:.1f}°  "
                          f"err|{m_val['r_abs_err']:.4f}|  phase_err {m_val['r_phase_err_deg']:.1f}°")
                    print(f"      t: pinn|{m_val['t_pinn_abs']:.4f}|∠{m_val['t_pinn_phase_deg']:.1f}°  "
                          f"rcwa|{m_val['t_rcwa_abs']:.4f}|∠{m_val['t_rcwa_phase_deg']:.1f}°  "
                          f"err|{m_val['t_abs_err']:.4f}|  phase_err {m_val['t_phase_err_deg']:.1f}°")
                    print(f"      R: pinn={m_val['R_pinn']:.4f}  rcwa={m_val['R_rcwa']:.4f}")
                    print(f"      T: pinn={m_val['T_pinn']:.4f}  rcwa={m_val['T_rcwa']:.4f}")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Modal data loss using RCWA reference amplitudes (training loss)
# ---------------------------------------------------------------------------

def modal_data_loss(
    subnet,
    z_val: float,
    physics: "PhysicsConfig",
    rcwa_amplitudes: np.ndarray,
    rcwa_n_harmonics: int,
    boundary: str,
    n_medium: float,
    n_data_orders: int = 3,
    weight_propagating: float = 1.0,
    weight_evanescent: float = 0.0,
    verbose: bool = False,
) -> "torch.Tensor":
    """Supervised modal data loss: ||E_scat_pinn_m - E_scat_rcwa_m||^2 per order.

    This function operates on SCATTERED field subnets (LBG PINN output).
    The target for each order is derived from RCWA total amplitudes by:
      1. De-embedding from reference plane to monitor plane.
      2. Subtracting background:
           Bottom: full E_bg(z_val)  for m=0
           Top:    reflected bg only r_eff*exp(+ik1*z_val)  for m=0
           m≠0:    no subtraction (background is x-independent, contributes 0)

    Background subtraction at top boundary
    ----------------------------------------
    At the top monitor the PINN scattered field has already removed the
    full background (E_inc + E_bg_refl).  The RCWA c_refl includes the
    m=0 total reflected amplitude.  The scattered target is:

        A_scat_top_m0 = A_total_m0(z_top) - E_bg_refl_m0(z_top)

    where E_bg_refl_m0 = r_eff * exp(+ik1*z_val) is the reflected background
    component ONLY (not the incident wave, which is not part of E_scat).

    This is DIFFERENT from extract_total_modal_amplitudes which subtracts
    E_bg = E_inc + E_bg_refl at the top monitor.  The difference:
      - extract_total_modal_amplitudes: removes total background from total field
      - modal_data_loss: removes reflected-bg from total RCWA target so it
        matches the LBG PINN scattered amplitude
    Both are correct for their respective purposes.
    """
    import torch
    from src.modal_dtn import _kz_outgoing
    from src.maxwell_layered_bg import compute_background_coefficients, background_field_np
    import math

    k0     = physics.k0
    period = physics.period
    G0     = 2.0 * math.pi / period
    N_x    = max(128, 8 * n_data_orders + 8)

    x_uni = torch.linspace(0.0, period, N_x + 1, dtype=torch.float64)[:-1]
    z_uni = torch.full_like(x_uni, z_val)

    Er_s, Ei_s, _, _, _, _ = subnet.field_components(x_uni, z_uni)

    orders  = np.arange(-n_data_orders, n_data_orders + 1)
    kx_m    = orders * G0
    kz_m    = _kz_outgoing(kx_m, n_medium, k0)
    dx      = period / N_x

    coeff = compute_background_coefficients(physics)

    if boundary == "bottom":
        z_rcwa_ref = physics.ridge_z_max
        Ebg_r_v, Ebg_i_v, _, _ = background_field_np(np.array([z_val]), coeff)
        E_bg_m0 = complex(float(Ebg_r_v[0]), float(Ebg_i_v[0]))
    else:  # top
        z_rcwa_ref = 0.0
        # At top: subtract reflected background only (not incident)
        r_eff = coeff["r_eff"]
        k1    = coeff["k1"]
        E_bg_m0 = r_eff * np.exp(+1j * k1 * z_val)

    loss     = torch.zeros(1, dtype=torch.float64)
    n_active = 0
    debug_rows = []

    for i_m, m in enumerate(orders):
        rcwa_idx = rcwa_n_harmonics + m
        if rcwa_idx < 0 or rcwa_idx >= len(rcwa_amplitudes):
            continue

        A_rcwa  = complex(rcwa_amplitudes[rcwa_idx])
        kz      = kz_m[i_m]
        is_prop = kz.real > 1e-6
        w       = weight_propagating if is_prop else weight_evanescent
        if w == 0:
            if verbose:
                debug_rows.append((m, "skip", abs(A_rcwa), 0.0, 0.0, 0.0))
            continue

        # De-embed to monitor plane
        if boundary == "bottom":
            dz           = z_val - z_rcwa_ref
            phase_factor = np.exp(-1j * kz * dz)
        else:
            dz           = z_val - z_rcwa_ref
            phase_factor = np.exp(+1j * kz * dz)

        A_at_monitor  = A_rcwa * phase_factor
        A_scat_target = A_at_monitor - E_bg_m0 if m == 0 else A_at_monitor

        target_r = float(A_scat_target.real)
        target_i = float(A_scat_target.imag)

        gm    = float(m) * G0
        x_np  = x_uni.detach().numpy()
        cos_j = torch.as_tensor(np.cos(gm * x_np), dtype=torch.float64)
        sin_j = torch.as_tensor(np.sin(gm * x_np), dtype=torch.float64)

        Em_r  = (dx / period) * torch.sum(Er_s * cos_j + Ei_s * sin_j)
        Em_i  = (dx / period) * torch.sum(Ei_s * cos_j - Er_s * sin_j)

        loss      = loss + w * ((Em_r - target_r) ** 2 + (Em_i - target_i) ** 2)
        n_active += 1

        if verbose:
            Em_c    = complex(float(Em_r.detach()), float(Em_i.detach()))
            err_abs = abs(Em_c - A_scat_target)
            err_ph  = abs(np.angle(Em_c / (A_scat_target + 1e-30)) * 180 / np.pi)
            debug_rows.append((m, "prop" if is_prop else "evan",
                               abs(A_rcwa), abs(A_at_monitor),
                               abs(A_scat_target), abs(Em_c), err_abs, err_ph))

    if verbose:
        print(f"  Modal loss debug ({boundary}, z={z_val:.3f}, z_ref={z_rcwa_ref:.3f}):")
        print(f"  {'m':>4}  {'type':>5}  {'|A_rcwa|':>9}  {'|A_mon|':>8}  "
              f"{'|A_scat_tgt|':>12}  {'|E_pinn_m|':>11}  {'amp_err':>8}  {'ph_err':>8}")
        for row in debug_rows:
            if len(row) == 8:
                m_, t_, ar, am, ast, ep, ea, ephase = row
                print(f"  {m_:>+4}  {t_:>5}  {ar:>9.4f}  {am:>8.4f}  "
                      f"{ast:>12.4f}  {ep:>11.4f}  {ea:>8.4f}  {ephase:>8.1f}°")
            else:
                print(f"  {row[0]:>+4}  {row[1]:>5}  |A|={row[2]:.4f}  (skipped)")

    return loss / max(n_active, 1)


def modal_data_loss_per_term(
    subnet_air,
    subnet_sub,
    physics: "PhysicsConfig",
    rcwa_amps: dict,
    z_top_monitor: float | None = None,
    z_bot_monitor: float | None = None,
    n_data_orders: int = 2,
) -> dict:
    """Compute individual modal loss terms L_r0, L_t0, L_tminus1, L_tplus1."""
    import torch
    import math

    if z_top_monitor is None:
        z_top_monitor = 0.08 * physics.domain_height
    if z_bot_monitor is None:
        z_bot_monitor = 0.92 * physics.domain_height

    k0 = physics.k0
    G0 = 2 * math.pi / physics.period
    N  = rcwa_amps["N_harmonics"]

    from src.modal_dtn import _kz_outgoing

    result = {}

    kz_sub = {m: _kz_outgoing(np.array([m * G0]), physics.n_substrate, k0)[0]
              for m in range(-n_data_orders, n_data_orders + 1)}

    for name, m in [("t0", 0), ("tminus1", -1), ("tplus1", +1)]:
        if abs(m) > n_data_orders:
            continue
        kz = kz_sub[m]
        if kz.real <= 1e-6:
            continue
        result[f"L_{name}"] = modal_data_loss(
            subnet_sub, z_bot_monitor, physics,
            rcwa_amps["c_trans"], N,
            "bottom", physics.n_substrate,
            n_data_orders=abs(m) if m != 0 else 1,
            weight_propagating=1.0, weight_evanescent=0.0,
        )

    kz_air_0 = _kz_outgoing(np.array([0.0]), physics.n_air, k0)[0]
    if kz_air_0.real > 1e-6:
        result["L_r0"] = modal_data_loss(
            subnet_air, z_top_monitor, physics,
            rcwa_amps["c_refl"], N,
            "top", physics.n_air,
            n_data_orders=1,
            weight_propagating=1.0, weight_evanescent=0.0,
        )

    return result


def audit_modal_loss(
    subnet_air,
    subnet_sub,
    physics: "PhysicsConfig",
    rcwa_amps: dict,
    z_top_monitor: float | None = None,
    z_bot_monitor: float | None = None,
    n_data_orders: int = 2,
) -> dict:
    """Full per-mode audit: modal_data_loss targets vs extract_total_modal_amplitudes."""
    if z_top_monitor is None:
        z_top_monitor = 0.08 * physics.domain_height
    if z_bot_monitor is None:
        z_bot_monitor = 0.92 * physics.domain_height

    print("\n  [BOTTOM boundary audit]")
    modal_data_loss(
        subnet_sub, z_bot_monitor, physics,
        rcwa_amps["c_trans"], rcwa_amps["N_harmonics"],
        "bottom", physics.n_substrate,
        n_data_orders=n_data_orders,
        weight_propagating=1.0, weight_evanescent=0.0, verbose=True,
    )

    print("\n  [TOP boundary audit]")
    modal_data_loss(
        subnet_air, z_top_monitor, physics,
        rcwa_amps["c_refl"], rcwa_amps["N_harmonics"],
        "top", physics.n_air,
        n_data_orders=n_data_orders,
        weight_propagating=1.0, weight_evanescent=0.0, verbose=True,
    )

    return {}


# ---------------------------------------------------------------------------
# RCWA amplitude loader
# ---------------------------------------------------------------------------

def load_rcwa_amplitudes(rcwa_path: "str | Path") -> dict:
    """Load RCWA complex amplitudes from reference NPZ.

    Returns dict with c_refl, c_trans, N_harmonics, R_m, T_m, R_total, T_total.
    """
    from pathlib import Path
    data = np.load(Path(rcwa_path), allow_pickle=True)
    required = ["c_refl", "c_trans", "R_m", "T_m"]
    missing  = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"RCWA NPZ missing keys {missing}. "
            "Regenerate with scripts/generate_reference.py."
        )
    n_orders = len(data["c_refl"])
    N        = (n_orders - 1) // 2
    return {
        "c_refl":      data["c_refl"].astype(complex),
        "c_trans":     data["c_trans"].astype(complex),
        "N_harmonics": N,
        "R_m":         data["R_m"].astype(float),
        "T_m":         data["T_m"].astype(float),
        "R_total":     float(data["R_total"]),
        "T_total":     float(data["T_total"]),
    }
