"""Dual total-field / scattered-field comparison between PINN and RCWA.

Background
----------
The LBG PINN predicts **scattered** fields:
    E_total_pinn = E_bg + E_scat_pinn
    H_total_pinn = H_bg + H_scat_pinn

The RCWA reference contains the **total** field:
    field_representation = "total"

This module implements two explicit comparison modes:

    "total"
        Compare E_total_pinn directly against E_total_rcwa.

    "scattered"
        Subtract the background from the RCWA reference:
            E_scat_rcwa = E_total_rcwa - E_bg
        Then compare E_scat_pinn against E_scat_rcwa.

For the free-space PINN (E_total = E_inc + E_scat), `E_bg = E_inc`.

All metrics are computed for both representations.

Conventions
-----------
- E_inc(z) = exp(-ik0 z)
- Background = E_bg from `maxwell_layered_bg.compute_background_coefficients`
- field_representation key is saved in every results NPZ

Modal coefficients
------------------
At two monitor planes (z_top inside air, z_bot inside substrate):
    - DFT along x → modal amplitudes r_m (reflected) and t_m (transmitted)
    - Reflected = total − incident at top monitor
    - Transmitted = total at bottom monitor (all downward)
    - Only propagating orders (Re kz > 0) contribute to power

RCWA provides complex r_m, t_m directly from c_refl / c_trans arrays.
PINN-derived r_m, t_m come from DFT of E_total at monitor planes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import PhysicsConfig


# ---------------------------------------------------------------------------
# Background field helper (numpy)
# ---------------------------------------------------------------------------


def _incident_field_np(z: np.ndarray, k0: float) -> tuple[np.ndarray, np.ndarray]:
    """E_inc = exp(-ik0 z).  Returns (E_inc_r, E_inc_i)."""
    return np.cos(k0 * z), -np.sin(k0 * z)


def _background_field_np(z: np.ndarray, physics: "PhysicsConfig") -> tuple[np.ndarray, np.ndarray]:
    """Layered background field (flat substrate, no ridge).

    Delegates to maxwell_layered_bg.  Returns (Ebg_r, Ebg_i).
    """
    from src.maxwell_layered_bg import background_field_np, compute_background_coefficients
    coeff = compute_background_coefficients(physics)
    Ebg_r, Ebg_i, _, _ = background_field_np(z, coeff)
    return Ebg_r, Ebg_i


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
) -> dict:
    """Dual comparison of PINN vs RCWA in both total and scattered representations.

    Parameters
    ----------
    pinn_E_scat_r, pinn_E_scat_i
        Scattered-field output of the PINN, shape (Nz, Nx).
        For free-space formulation, this is E_scat relative to E_inc.
        For layered_bg formulation, this is E_scat relative to E_bg.
    rcwa_E_total_r, rcwa_E_total_i
        Total field from RCWA (the correct reference), shape (Nz, Nx).
    z_grid
        Physical z coordinates, shape (Nz, Nx) or (Nz,).
    physics
        PhysicsConfig for background field computation.
    formulation : "free_space" | "layered_bg"
        Controls what background is subtracted.

    Returns
    -------
    dict with keys:
        total/*, scattered/*
            Metrics for each representation.
        pinn_E_total_r, pinn_E_total_i
        pinn_E_scat_r,  pinn_E_scat_i
        rcwa_E_total_r, rcwa_E_total_i
        rcwa_E_scat_r,  rcwa_E_scat_i
        field_representation_pinn  : "scattered"
        field_representation_rcwa  : "total"
    """
    # Flatten z to 1D for background computation
    z_1d = z_grid.ravel() if z_grid.ndim == 2 else z_grid
    shape = pinn_E_scat_r.shape

    # Background field
    if formulation == "layered_bg":
        Ebg_r_1d, Ebg_i_1d = _background_field_np(z_1d, physics)
    else:  # free_space: background = incident
        Ebg_r_1d, Ebg_i_1d = _incident_field_np(z_1d, physics.k0)

    Ebg_r = Ebg_r_1d.reshape(shape)
    Ebg_i = Ebg_i_1d.reshape(shape)

    # PINN total field
    pinn_total_r = pinn_E_scat_r + Ebg_r
    pinn_total_i = pinn_E_scat_i + Ebg_i

    # RCWA scattered field  (= total − background)
    rcwa_scat_r = rcwa_E_total_r - Ebg_r
    rcwa_scat_i = rcwa_E_total_i - Ebg_i

    # Compute all metrics
    def _metrics(pred_r, pred_i, ref_r, ref_i, label):
        valid = (np.isfinite(pred_r) & np.isfinite(pred_i)
                 & np.isfinite(ref_r)  & np.isfinite(ref_i))
        pred_c = pred_r[valid] + 1j * pred_i[valid]
        ref_c  = ref_r[valid]  + 1j * ref_i[valid]
        pred_mag = np.abs(pred_c)
        ref_mag  = np.abs(ref_c)
        eps = 1e-12

        complex_l2  = float(np.linalg.norm(pred_c - ref_c) /
                            (np.linalg.norm(ref_c) + eps))
        mag_l2      = float(np.linalg.norm(pred_mag - ref_mag) /
                            (np.linalg.norm(ref_mag) + eps))
        real_l2     = float(np.linalg.norm(pred_r[valid] - ref_r[valid]) /
                            (np.linalg.norm(ref_r[valid]) + eps))
        imag_l2     = float(np.linalg.norm(pred_i[valid] - ref_i[valid]) /
                            (np.linalg.norm(ref_i[valid]) + eps))

        # Phase RMSE (pointwise angle between complex numbers)
        phase_err = np.angle(pred_c / (ref_c + eps))  # handles branch cut
        phase_rmse_deg = float(np.sqrt(np.mean(phase_err**2)) * 180 / np.pi)

        # Global phase offset: angle of sum(pred * conj(ref))
        global_phase = float(np.angle(np.sum(pred_c * np.conj(ref_c))) * 180 / np.pi)

        return {
            f"{label}/complex_l2":    complex_l2,
            f"{label}/magnitude_l2":  mag_l2,
            f"{label}/real_l2":       real_l2,
            f"{label}/imag_l2":       imag_l2,
            f"{label}/phase_rmse_deg": phase_rmse_deg,
            f"{label}/global_phase_offset_deg": global_phase,
            f"{label}/n_valid": int(valid.sum()),
        }

    result = {}
    result.update(_metrics(pinn_total_r, pinn_total_i,
                           rcwa_E_total_r, rcwa_E_total_i, "total"))
    result.update(_metrics(pinn_E_scat_r, pinn_E_scat_i,
                           rcwa_scat_r, rcwa_scat_i, "scattered"))

    # Store the four field arrays for downstream use
    result["pinn_E_total_r"] = pinn_total_r
    result["pinn_E_total_i"] = pinn_total_i
    result["pinn_E_scat_r"]  = pinn_E_scat_r
    result["pinn_E_scat_i"]  = pinn_E_scat_i
    result["rcwa_E_total_r"] = rcwa_E_total_r
    result["rcwa_E_total_i"] = rcwa_E_total_i
    result["rcwa_E_scat_r"]  = rcwa_scat_r
    result["rcwa_E_scat_i"]  = rcwa_scat_i
    result["field_representation_pinn"] = "scattered"
    result["field_representation_rcwa"] = "total"
    result["formulation"] = formulation

    return result


# ---------------------------------------------------------------------------
# Modal coefficient extraction
# ---------------------------------------------------------------------------


def extract_modal_amplitudes(
    E_total_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: "PhysicsConfig",
    n_orders: int = 5,
    z_top_frac: float = 0.08,
    z_bot_frac: float = 0.92,
) -> dict:
    """Extract complex modal amplitudes r_m and t_m from E_total.

    Uses DFT decomposition at two monitor planes.

    Reflected amplitudes:
        E_refl(x) = E_total(x, z_top) - E_inc(z_top)
        r_m = (1/period) * integral_0^period E_refl(x) * exp(-i G_m x) dx

    Transmitted amplitudes:
        t_m = (1/period) * integral_0^period E_total(x, z_bot) * exp(-i G_m x) dx

    Power (diffraction efficiency):
        R_m = |r_m|^2 * Re(kz_m^air) / Re(kz_inc)
        T_m = |t_m|^2 * Re(kz_m^sub) / Re(kz_inc)

    Parameters
    ----------
    E_total_2d : complex (Nz, Nx)
        Total field (E_bg + E_scat_pinn) on the visualization grid.
    x1d : (Nx,)
    z1d : (Nz,)
    physics : PhysicsConfig
    n_orders : number of grating orders each side of zeroth (±n_orders)
    z_top_frac, z_bot_frac : monitor plane positions as fraction of domain_height

    Returns
    -------
    dict:
        orders, r_m, t_m, R_m, T_m, R_total, T_total, energy_check,
        r_m_complex, t_m_complex
    """
    k0     = physics.k0
    n_air  = physics.n_air
    n_sub  = physics.n_substrate
    period = physics.period

    orders = np.arange(-n_orders, n_orders + 1)
    G_m    = orders * (2.0 * np.pi / period)

    # Monitor plane indices
    z_top = z1d[np.argmin(np.abs(z1d - z_top_frac * physics.domain_height))]
    z_bot = z1d[np.argmin(np.abs(z1d - z_bot_frac * physics.domain_height))]
    iz_top = np.argmin(np.abs(z1d - z_top))
    iz_bot = np.argmin(np.abs(z1d - z_bot))

    # kz per order
    kz_air_m = np.sqrt(((k0 * n_air)**2 - G_m**2).astype(complex))
    kz_sub_m = np.sqrt(((k0 * n_sub)**2 - G_m**2).astype(complex))
    # Outgoing branch (evanescent decays away from grating)
    for kz in (kz_air_m, kz_sub_m):
        evan = kz.real < 1e-6
        kz[evan] = -1j * np.abs(kz[evan])

    kz_inc = kz_air_m[n_orders]  # m=0 order in air
    P_inc  = 0.5 * kz_inc.real / k0  # incident Poynting flux per unit period

    dx = period / len(x1d)

    def _dft_slice(E_slice):
        """Complex DFT amplitudes for each order."""
        amps = np.zeros(len(orders), dtype=complex)
        for mi, gm in enumerate(G_m):
            amps[mi] = np.sum(E_slice * np.exp(-1j * gm * x1d)) * dx / period
        return amps

    # Reflected: total at top − incident
    E_inc_top = np.exp(-1j * k0 * z_top)  # scalar (x-independent for normal incidence)
    E_refl_slice = E_total_2d[iz_top, :] - E_inc_top

    # Transmitted: total at bottom
    E_trans_slice = E_total_2d[iz_bot, :]

    r_m_complex = _dft_slice(E_refl_slice)
    t_m_complex = _dft_slice(E_trans_slice)

    # Power per order
    R_m = np.zeros(len(orders)); T_m = np.zeros(len(orders))
    for mi in range(len(orders)):
        if kz_air_m[mi].real > 1e-6:
            R_m[mi] = 0.5 * kz_air_m[mi].real / k0 * abs(r_m_complex[mi])**2 / (P_inc + 1e-30)
        if kz_sub_m[mi].real > 1e-6:
            T_m[mi] = 0.5 * kz_sub_m[mi].real / k0 * abs(t_m_complex[mi])**2 / (P_inc + 1e-30)

    R_total = float(R_m.sum())
    T_total = float(T_m.sum())
    idx0    = n_orders  # index of m=0 order

    return {
        "orders":    orders.tolist(),
        "r_m_complex": r_m_complex,
        "t_m_complex": t_m_complex,
        "r_m_abs":   np.abs(r_m_complex).tolist(),
        "t_m_abs":   np.abs(t_m_complex).tolist(),
        "r_m_phase_deg": (np.angle(r_m_complex) * 180 / np.pi).tolist(),
        "t_m_phase_deg": (np.angle(t_m_complex) * 180 / np.pi).tolist(),
        "R_m":       R_m.tolist(),
        "T_m":       T_m.tolist(),
        "R0":        float(R_m[idx0]),
        "T0":        float(T_m[idx0]),
        "r0_complex": complex(r_m_complex[idx0]),
        "t0_complex": complex(t_m_complex[idx0]),
        "R_total":   R_total,
        "T_total":   T_total,
        "energy_check": R_total + T_total,
        "z_top_monitor": float(z_top),
        "z_bot_monitor": float(z_bot),
        "P_inc":     float(P_inc),
    }


def compare_modal_with_rcwa(
    pinn_modal: dict,
    rcwa_path: str | None,
    n_harmonics_center: int,
) -> dict:
    """Compare PINN modal amplitudes against RCWA.

    Parameters
    ----------
    pinn_modal
        Output of extract_modal_amplitudes.
    rcwa_path
        Path to the reference NPZ (must contain c_refl, c_trans, kx, kz_air, kz_sub).
    n_harmonics_center
        N such that the m=0 mode is at index N in the RCWA amplitude arrays.

    Returns
    -------
    dict: per-order amplitude/phase errors and power efficiency comparison.
    """
    if rcwa_path is None:
        return {"note": "no RCWA reference provided"}

    import numpy as np
    from pathlib import Path
    data = np.load(Path(rcwa_path), allow_pickle=True)
    if "c_refl" not in data:
        return {"note": "RCWA reference does not contain amplitude data (regenerate with fixed solver)"}

    c_refl  = data["c_refl"]   # (2N+1,) complex
    c_trans = data["c_trans"]  # (2N+1,) complex
    kz_air  = data["kz_air"]
    kz_sub  = data["kz_sub"]
    kx      = data["kx"]
    R_m_rcwa = data["R_m"]
    T_m_rcwa = data["T_m"]

    N = n_harmonics_center
    orders = np.array(pinn_modal["orders"])
    n_orders = (len(orders) - 1) // 2

    r_pinn = np.array(pinn_modal["r_m_complex"])
    t_pinn = np.array(pinn_modal["t_m_complex"])

    # Align PINN orders with RCWA orders (RCWA has 2N+1 orders centered at N)
    results = {}
    for mi, m in enumerate(orders):
        rcwa_idx = N + m  # index into RCWA amplitude arrays
        if 0 <= rcwa_idx < len(c_refl):
            r_rcwa_m  = c_refl[rcwa_idx]
            t_rcwa_m  = c_trans[rcwa_idx]
            r_pinn_m  = r_pinn[mi]
            t_pinn_m  = t_pinn[mi]
            R_rcwa_m  = float(R_m_rcwa[rcwa_idx]) if rcwa_idx < len(R_m_rcwa) else 0.0
            T_rcwa_m  = float(T_m_rcwa[rcwa_idx]) if rcwa_idx < len(T_m_rcwa) else 0.0
            R_pinn_m  = float(pinn_modal["R_m"][mi])
            T_pinn_m  = float(pinn_modal["T_m"][mi])
            results[f"m={m}"] = {
                "r_pinn_abs":  float(abs(r_pinn_m)),
                "r_rcwa_abs":  float(abs(r_rcwa_m)),
                "r_pinn_phase_deg": float(np.angle(r_pinn_m) * 180 / np.pi),
                "r_rcwa_phase_deg": float(np.angle(r_rcwa_m) * 180 / np.pi),
                "r_abs_err":   float(abs(abs(r_pinn_m) - abs(r_rcwa_m))),
                "r_phase_err_deg": float(abs(np.angle(r_pinn_m / (r_rcwa_m + 1e-30)) * 180 / np.pi)),
                "t_pinn_abs":  float(abs(t_pinn_m)),
                "t_rcwa_abs":  float(abs(t_rcwa_m)),
                "t_pinn_phase_deg": float(np.angle(t_pinn_m) * 180 / np.pi),
                "t_rcwa_phase_deg": float(np.angle(t_rcwa_m) * 180 / np.pi),
                "t_abs_err":   float(abs(abs(t_pinn_m) - abs(t_rcwa_m))),
                "t_phase_err_deg": float(abs(np.angle(t_pinn_m / (t_rcwa_m + 1e-30)) * 180 / np.pi)),
                "R_pinn": R_pinn_m, "R_rcwa": R_rcwa_m,
                "T_pinn": T_pinn_m, "T_rcwa": T_rcwa_m,
            }

    # RCWA energy check from file
    R_rcwa_total = float(np.sum(R_m_rcwa))
    T_rcwa_total = float(np.sum(T_m_rcwa))
    results["summary"] = {
        "R_pinn_total": pinn_modal["R_total"],
        "T_pinn_total": pinn_modal["T_total"],
        "pinn_energy_check": pinn_modal["energy_check"],
        "R_rcwa_total": R_rcwa_total,
        "T_rcwa_total": T_rcwa_total,
        "rcwa_energy_check": R_rcwa_total + T_rcwa_total,
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

    Parameters
    ----------
    Accepts the same PINN scatter outputs for a **flat** physics config
    (n_ridge = n_substrate).

    Returns pass/fail status and max |E_scat|.
    """
    max_scat = float(np.max(np.sqrt(pinn_E_scat_r**2 + pinn_E_scat_i**2)))
    passed   = max_scat < tol

    # Total field should equal background
    z_1d = z_grid.ravel() if z_grid.ndim == 2 else z_grid
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
        "max_E_scat_magnitude": max_scat,
        "max_total_minus_bg_r": max_diff_r,
        "max_total_minus_bg_i": max_diff_i,
        "passed": passed,
        "tolerance": tol,
        "note": "For zero contrast, E_scat should be ~0 and E_total ~ E_bg",
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
            print(f"    {k:35s}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
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
# Modal data loss using RCWA reference amplitudes
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

    Derivation
    ----------
    LBG PINN outputs E_scat.  RCWA stores total field amplitudes c_refl / c_trans
    at the grating boundaries:
      - c_refl[N+m] at z=0       (reflected wave amplitude, upward)
      - c_trans[N+m] at ridge_z_max  (transmitted wave amplitude, downward)

    De-embedding to monitor plane z_val:
      Bottom (downward wave): A(z_val) = c_trans * exp(-i kz_m * (z_val - z_sub_top))
      Top    (upward wave):   A(z_val) = c_refl  * exp(+i kz_m * (z_val - 0))
                                       = c_refl  * exp(+i kz_m * z_val)
    Note: for upward waves at z>0 (z_val = 0.08*H > 0):
      kz_m branch gives Re(kz)>0 for propagating, Im(kz)<0 for evanescent.
      exp(+i kz_m * z_val) with Im(kz)<0 → decays (correct).

    PINN E_scat at the monitor = DFT of E_scat_pinn(x, z_val).

    Target E_scat_m = A_total_m(z_val) - E_bg_m(z_val):
      - Background E_bg has no grating orders (flat interface), so E_bg_m=0 for m≠0.
      - For m=0:
          Bottom: E_bg = tau * exp(-ik2*(z_val-z_int))  [full background transmitted]
          Top:    E_bg_refl = r_eff * exp(+ik1*z_val)   [only reflected BG, no incident]
                  The incident wave is NOT in E_scat, so subtract only reflected BG.

    Evanescent orders (Im(kz)<0): exp(±i kz * z) decays naturally.  We skip them
    by default (weight_evanescent=0) because the PINN scatter at the monitor is
    dominated by propagating orders.
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

    # Background field at monitor for m=0 subtraction
    coeff = compute_background_coefficients(physics)

    if boundary == "bottom":
        z_rcwa_ref = physics.ridge_z_max   # c_trans defined here
        # Full background at monitor (includes both incident+reflected propagation
        # through the substrate)
        Ebg_r_v, Ebg_i_v, _, _ = background_field_np(np.array([z_val]), coeff)
        E_bg_m0 = complex(float(Ebg_r_v[0]), float(Ebg_i_v[0]))
    else:
        z_rcwa_ref = 0.0   # c_refl defined at z=0
        # At top monitor, E_bg = E_inc + E_bg_refl
        # PINN E_scat does NOT include the incident wave.
        # E_scat_rcwa_top = c_refl_prop - (E_bg_refl only, NOT incident)
        # The background reflected component at z_val:
        r_eff = coeff['r_eff']; k1 = coeff['k1']
        E_bg_m0 = r_eff * np.exp(+1j * k1 * z_val)   # only reflected BG at top

    loss = torch.zeros(1, dtype=torch.float64)
    n_active = 0
    debug_rows = []

    for i_m, m in enumerate(orders):
        rcwa_idx = rcwa_n_harmonics + m
        if rcwa_idx < 0 or rcwa_idx >= len(rcwa_amplitudes):
            continue

        A_rcwa = complex(rcwa_amplitudes[rcwa_idx])
        kz = kz_m[i_m]
        is_prop = kz.real > 1e-6
        w = weight_propagating if is_prop else weight_evanescent
        if w == 0:
            if verbose:
                debug_rows.append((m, 'skip', abs(A_rcwa), 0.0, 0.0, 0.0))
            continue

        # De-embed to monitor plane
        if boundary == "bottom":
            dz = z_val - z_rcwa_ref          # positive (monitor below ref)
            phase_factor = np.exp(-1j * kz * dz)  # downward wave
        else:
            dz = z_val - z_rcwa_ref          # positive (monitor below z=0 in z↓ convention)
            phase_factor = np.exp(+1j * kz * dz)  # upward wave

        A_at_monitor = A_rcwa * phase_factor    # total field amplitude at monitor

        # Scattered field target = total - background (background only at m=0)
        if m == 0:
            A_scat_target = A_at_monitor - E_bg_m0
        else:
            A_scat_target = A_at_monitor        # no background at m≠0

        target_r = float(A_scat_target.real)
        target_i = float(A_scat_target.imag)

        # DFT of PINN scattered field
        gm    = float(m) * G0
        x_np  = x_uni.detach().numpy()
        cos_j = torch.as_tensor(np.cos(gm * x_np), dtype=torch.float64)
        sin_j = torch.as_tensor(np.sin(gm * x_np), dtype=torch.float64)

        Em_r = (dx / period) * torch.sum(Er_s * cos_j + Ei_s * sin_j)
        Em_i = (dx / period) * torch.sum(Ei_s * cos_j - Er_s * sin_j)

        loss = loss + w * ((Em_r - target_r)**2 + (Em_i - target_i)**2)
        n_active += 1

        if verbose:
            Em_c = complex(float(Em_r.detach()), float(Em_i.detach()))
            err_abs = abs(Em_c - A_scat_target)
            err_ph  = abs(np.angle(Em_c / (A_scat_target + 1e-30)) * 180 / np.pi)
            debug_rows.append((m, 'prop' if is_prop else 'evan',
                               abs(A_rcwa), abs(A_at_monitor), abs(A_scat_target),
                               abs(Em_c), err_abs, err_ph))

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
    """Compute individual modal loss terms L_r0, L_t0, L_tminus1, L_tplus1.

    Returns a dict of named scalar tensors for logging and selective weighting.
    """
    import torch, math

    if z_top_monitor is None:
        z_top_monitor = 0.08 * physics.domain_height
    if z_bot_monitor is None:
        z_bot_monitor = 0.92 * physics.domain_height

    k0 = physics.k0; G0 = 2*math.pi/physics.period
    N  = rcwa_amps["N_harmonics"]

    from src.modal_dtn import _kz_outgoing

    result = {}

    # -- BOTTOM: transmitted orders --
    kz_sub = {m: _kz_outgoing(np.array([m*G0]), physics.n_substrate, k0)[0]
              for m in range(-n_data_orders, n_data_orders+1)}

    for name, m in [("t0", 0), ("tminus1", -1), ("tplus1", +1)]:
        if abs(m) > n_data_orders:
            continue
        kz = kz_sub[m]
        if kz.real <= 1e-6:
            continue   # evanescent in substrate — skip

        result[f"L_{name}"] = modal_data_loss(
            subnet_sub, z_bot_monitor, physics,
            rcwa_amps["c_trans"], N,
            "bottom", physics.n_substrate,
            n_data_orders=abs(m) if m != 0 else 1,
            weight_propagating=1.0, weight_evanescent=0.0,
        )

    # -- TOP: reflected m=0 only (m=±1 evanescent in air for Λ=0.8λ) --
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
    """Full per-mode audit comparing modal_data_loss targets vs extract_modal_amplitudes.

    Prints and returns a table showing:
    - mode, boundary, monitor z, reference z
    - PINN E_scat_m (from DFT of subnet output)
    - RCWA target E_scat_m (de-embedded + background-subtracted)
    - amplitude error, phase error
    """
    import torch, math

    if z_top_monitor is None:
        z_top_monitor = 0.08 * physics.domain_height
    if z_bot_monitor is None:
        z_bot_monitor = 0.92 * physics.domain_height

    from src.modal_dtn import _kz_outgoing

    results = {}

    # Bottom
    print("\n  [BOTTOM boundary audit]")
    modal_data_loss(
        subnet_sub, z_bot_monitor, physics,
        rcwa_amps["c_trans"], rcwa_amps["N_harmonics"],
        "bottom", physics.n_substrate,
        n_data_orders=n_data_orders,
        weight_propagating=1.0, weight_evanescent=0.0,
        verbose=True,
    )

    # Top
    print("\n  [TOP boundary audit]")
    modal_data_loss(
        subnet_air, z_top_monitor, physics,
        rcwa_amps["c_refl"], rcwa_amps["N_harmonics"],
        "top", physics.n_air,
        n_data_orders=n_data_orders,
        weight_propagating=1.0, weight_evanescent=0.0,
        verbose=True,
    )

    return results
    """Load RCWA complex amplitudes from reference NPZ.

    Returns dict with:
        c_refl       : complex reflection amplitudes (2N+1,)
        c_trans      : complex transmission amplitudes (2N+1,)
        N_harmonics  : int, N such that m=0 is at index N
        R_m, T_m     : modal power efficiencies
        R_total, T_total
    """
    from pathlib import Path
    data = np.load(Path(rcwa_path), allow_pickle=True)
    required = ["c_refl", "c_trans", "R_m", "T_m"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"RCWA NPZ missing keys {missing}. Regenerate with scripts/generate_reference.py "
            f"(fixed _star version)."
        )
    n_orders = len(data["c_refl"])
    N = (n_orders - 1) // 2
    return {
        "c_refl":       data["c_refl"].astype(complex),
        "c_trans":      data["c_trans"].astype(complex),
        "N_harmonics":  N,
        "R_m":          data["R_m"],
        "T_m":          data["T_m"],
        "R_total":      float(data["R_total"]),
        "T_total":      float(data["T_total"]),
    }


def load_rcwa_amplitudes(rcwa_path: "str | Path") -> dict:
    """Load RCWA complex amplitudes from reference NPZ.

    Returns dict with:
        c_refl       : complex reflection amplitudes (2N+1,)
        c_trans      : complex transmission amplitudes (2N+1,)
        N_harmonics  : int, N such that m=0 is at index N
        R_m, T_m     : modal power efficiencies
        R_total, T_total
    """
    from pathlib import Path
    data = np.load(Path(rcwa_path), allow_pickle=True)
    required = ["c_refl", "c_trans", "R_m", "T_m"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"RCWA NPZ missing keys {missing}. Regenerate with scripts/generate_reference.py "
            f"(fixed _star version)."
        )
    n_orders = len(data["c_refl"])
    N = (n_orders - 1) // 2
    return {
        "c_refl":       data["c_refl"].astype(complex),
        "c_trans":      data["c_trans"].astype(complex),
        "N_harmonics":  N,
        "R_m":          data["R_m"],
        "T_m":          data["T_m"],
        "R_total":      float(data["R_total"]),
        "T_total":      float(data["T_total"]),
    }
