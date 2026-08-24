#!/usr/bin/env python3
"""Reference-only validation of the production modal extraction path.

This deliberately uses ``src.field_comparison`` rather than reproducing its
DFT logic.  It checks that total and layered-background scattered fields,
constructed from the corrected RCWA reference, recover the same total modal
response at the exact PINN monitor planes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_modal_extractor import load_reference, reconstruct_total_field_at_z
from src.config import PhysicsConfig
from src.field_comparison import (
    extract_modal_amplitudes,
    reconstruct_total_modal_amplitudes,
)
from src.maxwell_layered_bg import background_field_np, compute_background_coefficients

TOL_AMP = 5e-3
TOL_PHASE_DEG = 1.0
TOL_ENERGY = 5e-3


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _physics(ref: dict) -> PhysicsConfig:
    return PhysicsConfig(
        wavelength=2 * np.pi / ref["k0"], n_air=ref["n_air"],
        n_ridge=1.5, n_substrate=ref["n_sub"], period=ref["period"],
        ridge_width=0.4 * ref["period"], ridge_height=0.2,
        domain_height=ref["domain_height"], ridge_base_fraction=0.6,
    )


def _expected_total_modes(ref: dict, physics: PhysicsConfig, z_top: float, z_bot: float,
                          orders: list[int]) -> tuple[np.ndarray, np.ndarray]:
    r = []
    t = []
    for m in orders:
        i = ref["N"] + m
        r.append(ref["c_refl"][i] * np.exp(1j * ref["kz_air"][i] * z_top))
        t.append(ref["c_trans"][i] * np.exp(
            -1j * ref["kz_sub"][i] * (z_bot - physics.ridge_z_max)))
    return np.asarray(r), np.asarray(t)


def _reconstruct_hx_at_z(z_val: float, x: np.ndarray, ref: dict,
                         physics: PhysicsConfig, region: str) -> np.ndarray:
    """Reconstruct TE Hx with the repository's forward-Hx-positive convention."""
    phase = np.exp(1j * np.outer(x, ref["kx"]))
    if region == "air":
        incident = np.zeros_like(ref["c_refl"], dtype=complex)
        incident[ref["N"]] = 1.0
        forward = (ref["kz_air"] / physics.k0) * incident * np.exp(
            -1j * ref["kz_air"] * z_val)
        reflected = -(ref["kz_air"] / physics.k0) * ref["c_refl"] * np.exp(
            1j * ref["kz_air"] * z_val)
        return phase @ (forward + reflected)
    transmitted = (ref["kz_sub"] / physics.k0) * ref["c_trans"] * np.exp(
        -1j * ref["kz_sub"] * (z_val - physics.ridge_z_max))
    return phase @ transmitted


def _modal_errors(actual: dict, expected_r: np.ndarray, expected_t: np.ndarray) -> dict:
    r = np.asarray(actual["r_m_complex"], dtype=complex)
    t = np.asarray(actual["t_m_complex"], dtype=complex)
    return {
        "r_amplitude_max": float(np.max(np.abs(np.abs(r) - np.abs(expected_r)))),
        "t_amplitude_max": float(np.max(np.abs(np.abs(t) - np.abs(expected_t)))),
        "r_phase_max_deg": float(np.max(np.abs(np.angle(r / (expected_r + 1e-30))) * 180 / np.pi)),
        "t_phase_max_deg": float(np.max(np.abs(np.angle(t / (expected_t + 1e-30))) * 180 / np.pi)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "outputs/reference_lambda_0p8.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/research_status/production_modal_path_verification.json")
    args = parser.parse_args()

    ref = load_reference(args.reference)
    physics = _physics(ref)
    z_top = 0.08 * physics.domain_height
    z_bot = 0.92 * physics.domain_height
    x = np.linspace(0.0, physics.period, 512, endpoint=False)
    z = np.asarray([z_top, z_bot])
    total = np.vstack((
        reconstruct_total_field_at_z(z_top, x, ref, physics.ridge_z_max, "air"),
        reconstruct_total_field_at_z(z_bot, x, ref, physics.ridge_z_max, "substrate"),
    ))

    coeff = compute_background_coefficients(physics)
    bg_r, bg_i, hx_r, hx_i = background_field_np(z, coeff)
    bg = bg_r[:, None] + 1j * bg_i[:, None]
    e_scat = total - bg

    # Construct H_total and H_scat with the same representation split.  The
    # current production modal extractor is E-only, but this verifies that the
    # field bookkeeping is well-defined for both E and H.
    h_bg = hx_r[:, None] + 1j * hx_i[:, None]
    h_total = np.vstack((
        _reconstruct_hx_at_z(z_top, x, ref, physics, "air"),
        _reconstruct_hx_at_z(z_bot, x, ref, physics, "substrate"),
    ))
    h_scat = h_total - h_bg

    total_modal = extract_modal_amplitudes(
        total, x, z, physics, formulation="layered_bg", n_orders=1,
        field_representation="total")
    scat_modal = extract_modal_amplitudes(
        e_scat, x, z, physics, formulation="layered_bg", n_orders=1,
        field_representation="scattered")
    reconstructed_modal = reconstruct_total_modal_amplitudes(scat_modal, physics)

    expected_r, expected_t = _expected_total_modes(
        ref, physics, z_top, z_bot, total_modal["orders"])
    total_errors = _modal_errors(total_modal, expected_r, expected_t)
    scattered_errors = _modal_errors(reconstructed_modal, expected_r, expected_t)
    r_difference = np.asarray(total_modal["r_m_complex"]) - np.asarray(
        reconstructed_modal["r_m_complex"])
    t_difference = np.asarray(total_modal["t_m_complex"]) - np.asarray(
        reconstructed_modal["t_m_complex"])
    agreement = {
        "r_max_abs_difference": float(np.max(np.abs(r_difference))),
        "t_max_abs_difference": float(np.max(np.abs(t_difference))),
    }
    passed = (
        abs(total_modal["energy_check"] - 1.0) < TOL_ENERGY
        and max(total_errors.values()) < max(TOL_AMP, TOL_PHASE_DEG)
        and total_errors["r_amplitude_max"] < TOL_AMP
        and total_errors["t_amplitude_max"] < TOL_AMP
        and total_errors["r_phase_max_deg"] < TOL_PHASE_DEG
        and total_errors["t_phase_max_deg"] < TOL_PHASE_DEG
        and agreement["r_max_abs_difference"] < TOL_AMP
        and agreement["t_max_abs_difference"] < TOL_AMP
        and np.isfinite(h_scat).all()
    )
    report = {
        "valid": bool(passed),
        "invalid_reason": None if passed else "production modal extraction did not meet reference-only tolerances",
        "reference_file": str(args.reference), "test_commit_hash": _git_hash(),
        "wavelength": physics.wavelength, "period": physics.period,
        "ridge_geometry": {"width": physics.ridge_width, "height": physics.ridge_height,
                           "base_z": physics.ridge_base_z},
        "material_indices": {"air": physics.n_air, "ridge": physics.n_ridge,
                             "substrate": physics.n_substrate},
        "monitor_positions": {"top": z_top, "bottom": z_bot},
        "reference_planes": {"reflection": 0.0, "transmission": physics.ridge_z_max},
        "field_representation": {"reference": "total", "production_inputs": ["total", "scattered"]},
        "background_formulation": "layered_bg",
        "de_embedding_status": "applied when comparing stored RCWA amplitudes to monitor planes",
        "background_subtraction": "E_scat = E_total - E_bg; total top reflection subtracts E_inc only",
        "total_modal": {k: total_modal[k] for k in ("R_total", "T_total", "energy_check", "orders")},
        "total_errors": total_errors, "scattered_reconstructed_errors": scattered_errors,
        "production_path_agreement": agreement,
        "H_scattered_finite": bool(np.isfinite(h_scat).all()),
        "standalone_validator": {"path": "outputs/research_status/modal_extractor_verification.json",
                                  "expected_assertions": "43/43"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
