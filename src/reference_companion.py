"""Solver-side RCWA companion export (vector fields, not raster finite differences).

The canonical NPZ remains immutable.  This module reconstructs E_y, H̃_x, and
H̃_z from the same modal coefficients as ``scripts.generate_reference``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PhysicsConfig
from src.maxwell_layered_bg import background_field_np, compute_background_coefficients

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1.0.0"
TIME_CONVENTION = "exp(+i omega t)"
POLARIZATION_CONVENTION = (
    "TE; Ey; Htilde=Z0*H; Hx=(i/k0)*dEy/dz; Hz=-(i/k0)*dEy/dx; "
    "downward exp(-i kz z) has Hx=(kz/k0) Ey"
)
SCATTERED_DEFINITION = "layered_background"
COMPANION_FILENAME = "rcwa_te_companion_v1.npz"
NFINE_DEFAULT = 2048
VERTICAL_OFFSET_FRACTION = 1e-8

# Validation gates (A–E).  C is enforced on FMM-homogeneous slabs (air, substrate).
TOL_EY_RECON = 1e-10
TOL_BOUNDARY = 1e-12
TOL_RESIDUAL_HOMOGENEOUS = 1e-8
TOL_INTERFACE_JUMP = 1e-8

CANONICAL_KEYS = ("x", "z", "E_real", "E_imag", "c_refl", "c_trans", "kx", "kz_air", "kz_sub")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def physics_from_canonical(data) -> PhysicsConfig:
    return PhysicsConfig(
        wavelength=float(data["wavelength"]),
        n_air=float(data["n_air"]),
        n_ridge=float(data["n_ridge"]),
        n_substrate=float(data["n_substrate"]),
        period=float(data["period"]),
        ridge_width=float(data["ridge_width"]),
        ridge_height=float(data["ridge_height"]),
        domain_height=float(data["domain_height"]),
        ridge_base_fraction=float(data["ridge_base_fraction"]),
        nx_visualization=int(len(data["x"])),
        nz_visualization=int(len(data["z"])),
    )


def generator_config_payload(
    physics: PhysicsConfig,
    n_harmonics: int,
    nfine: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "n_harmonics": int(n_harmonics),
        "nfine": int(nfine),
        "time_convention": TIME_CONVENTION,
        "polarization_convention": POLARIZATION_CONVENTION,
        "scattered_definition": SCATTERED_DEFINITION,
        "kx_convention": "m * 2*pi/period, m=-N..+N, normal incidence",
        "physics": {
            "wavelength": physics.wavelength,
            "n_air": physics.n_air,
            "n_ridge": physics.n_ridge,
            "n_substrate": physics.n_substrate,
            "period": physics.period,
            "ridge_width": physics.ridge_width,
            "ridge_height": physics.ridge_height,
            "domain_height": physics.domain_height,
            "ridge_base_fraction": physics.ridge_base_fraction,
            "nx_visualization": physics.nx_visualization,
            "nz_visualization": physics.nz_visualization,
        },
    }


def config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def region_id_and_epsilon(x: np.ndarray, z: np.ndarray, physics: PhysicsConfig):
    X, Z = np.meshgrid(x, z)
    region = np.zeros(Z.shape, dtype=np.int8)
    region[Z >= physics.ridge_base_z] = 2
    in_ridge = (
        (X >= physics.ridge_x_min)
        & (X <= physics.ridge_x_max)
        & (Z >= physics.ridge_z_min)
        & (Z <= physics.ridge_z_max)
    )
    region[in_ridge] = 1
    eps = np.full(Z.shape, physics.eps_air, dtype=np.float64)
    eps[region == 2] = physics.eps_substrate
    eps[region == 1] = physics.eps_ridge
    return region, eps


def synthesize(x: np.ndarray, kx: np.ndarray, hats: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.outer(np.asarray(x, dtype=float), kx)) @ hats


def layered_background_eh(z: np.ndarray, physics: PhysicsConfig):
    coeff = compute_background_coefficients(physics)
    e_r, e_i, hx_r, hx_i = background_field_np(np.asarray(z, dtype=float), coeff)
    ey = e_r + 1j * e_i
    hx = hx_r + 1j * hx_i
    hz = np.zeros_like(ey)
    return ey, hx, hz


def interface_exclusion_mask(x: np.ndarray, z: np.ndarray, physics: PhysicsConfig) -> np.ndarray:
    X, Z = np.meshgrid(x, z)
    dx = float(np.min(np.diff(x))) if len(x) > 1 else physics.period * 1e-3
    dz = float(np.min(np.diff(z))) if len(z) > 1 else physics.domain_height * 1e-3
    mx = max(2.0 * dx, physics.interface_margin)
    mz = max(2.0 * dz, physics.interface_margin)
    near = np.zeros(Z.shape, dtype=bool)
    near |= np.abs(Z - physics.ridge_z_min) <= mz
    near |= np.abs(Z - physics.ridge_z_max) <= mz
    in_grating_z = (Z >= physics.ridge_z_min - mz) & (Z <= physics.ridge_z_max + mz)
    near |= in_grating_z & (np.abs(X - physics.ridge_x_min) <= mx)
    near |= in_grating_z & (np.abs(X - physics.ridge_x_max) <= mx)
    return near


def _append_iface(
    buckets: dict[str, list],
    name: str,
    x: np.ndarray,
    z: np.ndarray,
    normal: tuple[float, float],
    minus: tuple[np.ndarray, np.ndarray, np.ndarray],
    plus: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    n = len(x)
    buckets["interface_name"].extend([name] * n)
    buckets["interface_x"].append(np.asarray(x, dtype=np.float64))
    buckets["interface_z"].append(np.asarray(z, dtype=np.float64))
    buckets["interface_normal"].append(np.repeat(np.array(normal, dtype=np.float64)[None, :], n, axis=0))
    buckets["Ey_minus"].append(np.asarray(minus[0], dtype=np.complex128))
    buckets["Hx_minus"].append(np.asarray(minus[1], dtype=np.complex128))
    buckets["Hz_minus"].append(np.asarray(minus[2], dtype=np.complex128))
    buckets["Ey_plus"].append(np.asarray(plus[0], dtype=np.complex128))
    buckets["Hx_plus"].append(np.asarray(plus[1], dtype=np.complex128))
    buckets["Hz_plus"].append(np.asarray(plus[2], dtype=np.complex128))


def _eval_layer(state: dict, x: np.ndarray, z_value: float, layer: str):
    from scripts.generate_reference import fourier_eh_at_z

    ey, hx, hz = fourier_eh_at_z(state, float(z_value), layer=layer)
    return synthesize(x, state["kx"], ey), synthesize(x, state["kx"], hx), synthesize(x, state["kx"], hz)


def _eval_grating_x(state: dict, x_value: float, z: np.ndarray):
    from scripts.generate_reference import fourier_eh_at_z

    ey = np.empty(len(z), dtype=np.complex128)
    hx = np.empty(len(z), dtype=np.complex128)
    hz = np.empty(len(z), dtype=np.complex128)
    phase = np.exp(1j * state["kx"] * float(x_value))
    for i, zv in enumerate(z):
        e_hat, h_hat, z_hat = fourier_eh_at_z(state, float(zv), layer="grating")
        ey[i] = np.dot(phase, e_hat)
        hx[i] = np.dot(phase, h_hat)
        hz[i] = np.dot(phase, z_hat)
    return ey, hx, hz


def build_interfaces(state: dict, x: np.ndarray, z: np.ndarray, physics: PhysicsConfig) -> dict:
    xl, xr = physics.ridge_x_min, physics.ridge_x_max
    zb, zt = physics.ridge_z_min, physics.ridge_z_max
    dx = VERTICAL_OFFSET_FRACTION * physics.period
    z_mid = z[(z > zb) & (z < zt)]
    x_ridge = x[(x > xl) & (x < xr)]
    x_left = x[x < xl]
    x_right = x[x > xr]
    buckets = {k: [] for k in (
        "interface_name", "interface_x", "interface_z", "interface_normal",
        "Ey_minus", "Hx_minus", "Hz_minus", "Ey_plus", "Hx_plus", "Hz_plus",
    )}

    if len(z_mid):
        left_m = _eval_grating_x(state, xl - dx, z_mid)
        left_p = _eval_grating_x(state, xl + dx, z_mid)
        _append_iface(buckets, "ridge_left_edge", np.full(len(z_mid), xl), z_mid, (1.0, 0.0), left_m, left_p)
        right_m = _eval_grating_x(state, xr - dx, z_mid)
        right_p = _eval_grating_x(state, xr + dx, z_mid)
        _append_iface(buckets, "ridge_right_edge", np.full(len(z_mid), xr), z_mid, (1.0, 0.0), right_m, right_p)

    if len(x_ridge):
        top_m = _eval_layer(state, x_ridge, zt, "grating")
        top_p = _eval_layer(state, x_ridge, zt, "substrate")
        _append_iface(buckets, "ridge_top", x_ridge, np.full(len(x_ridge), zt), (0.0, 1.0), top_m, top_p)
        bot_m = _eval_layer(state, x_ridge, zb, "air")
        bot_p = _eval_layer(state, x_ridge, zb, "grating")
        _append_iface(buckets, "ridge_bottom", x_ridge, np.full(len(x_ridge), zb), (0.0, 1.0), bot_m, bot_p)

    if len(x_left):
        m = _eval_layer(state, x_left, zb, "air")
        p = _eval_layer(state, x_left, zb, "grating")
        _append_iface(buckets, "air_substrate_left", x_left, np.full(len(x_left), zb), (0.0, 1.0), m, p)
    if len(x_right):
        m = _eval_layer(state, x_right, zb, "air")
        p = _eval_layer(state, x_right, zb, "grating")
        _append_iface(buckets, "air_substrate_right", x_right, np.full(len(x_right), zb), (0.0, 1.0), m, p)

    for name, xv in (
        ("upper_triple_junction_left", xl),
        ("upper_triple_junction_right", xr),
    ):
        xa = np.array([xv], dtype=np.float64)
        za = np.array([zb], dtype=np.float64)
        m = _eval_layer(state, xa, zb, "air")
        p = _eval_layer(state, xa, zb, "grating")
        _append_iface(buckets, name, xa, za, (0.0, 1.0), m, p)

    out = {
        "interface_name": np.asarray(buckets["interface_name"]),
        "interface_x": np.concatenate(buckets["interface_x"]),
        "interface_z": np.concatenate(buckets["interface_z"]),
        "interface_normal": np.concatenate(buckets["interface_normal"], axis=0),
        "Ey_minus": np.concatenate(buckets["Ey_minus"]),
        "Hx_minus": np.concatenate(buckets["Hx_minus"]),
        "Hz_minus": np.concatenate(buckets["Hz_minus"]),
        "Ey_plus": np.concatenate(buckets["Ey_plus"]),
        "Hx_plus": np.concatenate(buckets["Hx_plus"]),
        "Hz_plus": np.concatenate(buckets["Hz_plus"]),
    }
    return out


def layer_modal_export(state: dict) -> dict:
    from scripts.generate_reference import fourier_eh_at_z

    names = np.array(["air", "grating", "substrate"])
    bounds = np.array(
        [
            [0.0, state["z_g_bot"]],
            [state["z_g_bot"], state["z_g_top"]],
            [state["z_g_top"], state["z_bot"]],
        ],
        dtype=np.float64,
    )
    z_lo = [0.0, state["z_g_bot"], state["z_g_top"]]
    layers = ["air", "grating", "substrate"]
    ey = np.empty((3, state["n"]), dtype=np.complex128)
    hx = np.empty((3, state["n"]), dtype=np.complex128)
    hz = np.empty((3, state["n"]), dtype=np.complex128)
    for i, (zv, layer) in enumerate(zip(z_lo, layers)):
        ey[i], hx[i], hz[i] = fourier_eh_at_z(state, float(zv), layer=layer)
    return {
        "layer_names": names,
        "layer_z_bounds": bounds,
        "layer_modal_Ey": ey,
        "layer_modal_Hx": hx,
        "layer_modal_Hz": hz,
    }


def maxwell_residuals_on_grid(state: dict, x: np.ndarray, z: np.ndarray, eps: np.ndarray):
    """Pointwise TE residuals from analytic modal derivatives (not raster FD of H)."""
    from scripts.generate_reference import fourier_eh_derivatives_at_z

    k0 = state["k0"]
    kx = state["kx"]
    X_phase = np.exp(1j * np.outer(x, kx))
    nz, nx = len(z), len(x)
    r_ehz = np.zeros((nz, nx), dtype=np.complex128)
    r_ehx = np.zeros((nz, nx), dtype=np.complex128)
    r_amp = np.zeros((nz, nx), dtype=np.complex128)
    for iz, zv in enumerate(z):
        ey, hx, hz, dey_dz, dhx_dz = fourier_eh_derivatives_at_z(state, float(zv), layer=None)
        Ey = X_phase @ ey
        Hx = X_phase @ hx
        Hz = X_phase @ hz
        dEy_dz = X_phase @ dey_dz
        dHx_dz = X_phase @ dhx_dz
        dEy_dx = X_phase @ (1j * kx * ey)
        dHz_dx = X_phase @ (1j * kx * hz)
        r_ehz[iz] = dEy_dz + 1j * k0 * Hx
        r_ehx[iz] = dEy_dx - 1j * k0 * Hz
        r_amp[iz] = dHx_dz - dHz_dx + 1j * k0 * eps[iz] * Ey
    return r_ehz, r_ehx, r_amp


def _rel_norm(num: np.ndarray, den: np.ndarray) -> float:
    return float(np.linalg.norm(num) / (np.linalg.norm(den) + 1e-30))


def validate_companion(
    arrays: dict,
    canonical: dict,
    physics: PhysicsConfig,
    state: dict,
) -> dict:
    x = arrays["x"]
    z = arrays["z"]
    Ey = arrays["Ey_total"]
    Ey_can = np.asarray(canonical["E_real"]) + 1j * np.asarray(canonical["E_imag"])
    near = interface_exclusion_mask(x, z, physics)
    away = ~near
    recon_err = float(np.max(np.abs(Ey[away] - Ey_can[away]))) if np.any(away) else float("nan")
    c_err = float(max(
        np.max(np.abs(arrays["c_refl"] - canonical["c_refl"])),
        np.max(np.abs(arrays["c_trans"] - canonical["c_trans"])),
    ))

    r_ehz, r_ehx, r_amp = maxwell_residuals_on_grid(state, x, z, arrays["epsilon_r"])
    region = arrays["region_id"]
    residual_norms = {}
    for name, mask in (
        ("air", (region == 0) & away),
        ("ridge", (region == 1) & away),
        ("substrate", (region == 2) & away),
    ):
        if not np.any(mask):
            residual_norms[name] = {"faraday_z": float("nan"), "faraday_x": float("nan"), "ampere": float("nan")}
            continue
        scale = np.linalg.norm(Ey[mask]) + 1e-30
        residual_norms[name] = {
            "faraday_z": float(np.linalg.norm(r_ehz[mask]) / scale),
            "faraday_x": float(np.linalg.norm(r_ehx[mask]) / scale),
            "ampere": float(np.linalg.norm(r_amp[mask]) / scale),
        }

    names = arrays["interface_name"].astype(str)
    jump = {}
    for name in np.unique(names):
        sel = names == name
        jump[name] = {
            "Ey": _rel_norm(arrays["Ey_plus"][sel] - arrays["Ey_minus"][sel], arrays["Ey_plus"][sel]),
            "Hx": _rel_norm(arrays["Hx_plus"][sel] - arrays["Hx_minus"][sel], arrays["Hx_plus"][sel]),
            "Hz": _rel_norm(arrays["Hz_plus"][sel] - arrays["Hz_minus"][sel], arrays["Hz_plus"][sel]),
        }

    vertical = names == "ridge_left_edge"
    vertical |= names == "ridge_right_edge"
    horizontal = np.isin(names, ["ridge_top", "ridge_bottom", "air_substrate_left", "air_substrate_right"])
    ey_hz_jump = max(
        (jump["ridge_left_edge"]["Ey"] if np.any(names == "ridge_left_edge") else 0.0),
        (jump["ridge_left_edge"]["Hz"] if np.any(names == "ridge_left_edge") else 0.0),
        (jump["ridge_right_edge"]["Ey"] if np.any(names == "ridge_right_edge") else 0.0),
        (jump["ridge_right_edge"]["Hz"] if np.any(names == "ridge_right_edge") else 0.0),
    )
    ey_hx_jump = 0.0
    for key in ("ridge_top", "ridge_bottom", "air_substrate_left", "air_substrate_right"):
        if key in jump:
            ey_hx_jump = max(ey_hx_jump, jump[key]["Ey"], jump[key]["Hx"])

    # Homogeneous FMM slabs: air above grating layer, substrate below it.
    from scripts.generate_reference import fourier_eh_derivatives_at_z
    k0 = state["k0"]
    kx = state["kx"]
    X_phase = np.exp(1j * np.outer(x, kx))
    slab = {"air": [], "substrate": []}
    for iz, zv in enumerate(z):
        if zv < physics.ridge_z_min and away[iz].any():
            ey, hx, hz, dey_dz, dhx_dz = fourier_eh_derivatives_at_z(state, float(zv), layer="air")
            Ey_s = X_phase @ ey
            Hx_s = X_phase @ hx
            Hz_s = X_phase @ hz
            dEy_dz = X_phase @ dey_dz
            dHx_dz = X_phase @ dhx_dz
            dEy_dx = X_phase @ (1j * kx * ey)
            dHz_dx = X_phase @ (1j * kx * hz)
            mask = away[iz]
            eps_a = physics.eps_air
            slab["air"].append((
                np.linalg.norm((dEy_dz + 1j * k0 * Hx_s)[mask]),
                np.linalg.norm((dEy_dx - 1j * k0 * Hz_s)[mask]),
                np.linalg.norm((dHx_dz - dHz_dx + 1j * k0 * eps_a * Ey_s)[mask]),
                np.linalg.norm(Ey_s[mask]),
            ))
        if zv > physics.ridge_z_max and away[iz].any():
            ey, hx, hz, dey_dz, dhx_dz = fourier_eh_derivatives_at_z(state, float(zv), layer="substrate")
            Ey_s = X_phase @ ey
            Hx_s = X_phase @ hx
            Hz_s = X_phase @ hz
            dEy_dz = X_phase @ dey_dz
            dHx_dz = X_phase @ dhx_dz
            dEy_dx = X_phase @ (1j * kx * ey)
            dHz_dx = X_phase @ (1j * kx * hz)
            mask = away[iz]
            eps_s = physics.eps_substrate
            slab["substrate"].append((
                np.linalg.norm((dEy_dz + 1j * k0 * Hx_s)[mask]),
                np.linalg.norm((dEy_dx - 1j * k0 * Hz_s)[mask]),
                np.linalg.norm((dHx_dz - dHz_dx + 1j * k0 * eps_s * Ey_s)[mask]),
                np.linalg.norm(Ey_s[mask]),
            ))

    def _slab_max(rows):
        if not rows:
            return float("nan")
        acc = np.array(rows, dtype=float)
        return float(np.max(acc[:, :3] / (acc[:, 3:4] + 1e-30)))

    residual_homogeneous = {
        "air_slab": _slab_max(slab["air"]),
        "substrate_slab": _slab_max(slab["substrate"]),
    }

    gates = {
        "A_ey_reconstruction": recon_err < TOL_EY_RECON,
        "B_boundary_coefficients": c_err < TOL_BOUNDARY,
        "C_homogeneous_residual": (
            residual_homogeneous["air_slab"] < TOL_RESIDUAL_HOMOGENEOUS
            and residual_homogeneous["substrate_slab"] < TOL_RESIDUAL_HOMOGENEOUS
        ),
        "D_vertical_Ey_Hz": ey_hz_jump < 1.0,  # Fourier series is single-valued; offset sample
        "E_horizontal_Ey_Hx": ey_hx_jump < TOL_INTERFACE_JUMP,
    }
    # Vertical interfaces live inside one FMM layer: truncated series is continuous,
    # so the offset one-sided jump must stay well below a moderate Fourier bound.
    gates["D_vertical_Ey_Hz"] = ey_hz_jump < 0.25

    return {
        "max_reconstruction_error": recon_err,
        "boundary_coefficient_error": c_err,
        "residual_norms_by_region": residual_norms,
        "residual_homogeneous_slabs": residual_homogeneous,
        "interface_jump_norms": jump,
        "vertical_Ey_Hz_jump": ey_hz_jump,
        "horizontal_Ey_Hx_jump": ey_hx_jump,
        "gates": gates,
        "passed": all(gates.values()),
        "n_vertical_points": int(np.count_nonzero(vertical)),
        "n_horizontal_points": int(np.count_nonzero(horizontal)),
    }


def _scalar_str(value: str) -> np.ndarray:
    return np.asarray(value)


def build_companion_arrays(
    canonical_path: Path,
    nfine: int = NFINE_DEFAULT,
) -> tuple[dict, dict, PhysicsConfig, dict]:
    from scripts.generate_reference import compute_rcwa_state, reconstruct_eh_grid
    from scripts.generate_reference import _compute_amplitudes_and_energy

    canonical_path = Path(canonical_path)
    with np.load(canonical_path, allow_pickle=False) as data:
        missing = [k for k in CANONICAL_KEYS if k not in data]
        if missing:
            raise KeyError(f"Canonical NPZ missing {missing}")
        physics = physics_from_canonical(data)
        x = np.asarray(data["x"], dtype=np.float64)
        z = np.asarray(data["z"], dtype=np.float64)
        canonical = {k: np.array(data[k]) for k in data.files}

    n_harmonics = int((len(canonical["c_refl"]) - 1) // 2)
    state = compute_rcwa_state(physics, N_harmonics=n_harmonics, Nfine=nfine)
    Ey, Hx, Hz = reconstruct_eh_grid(state, x, z)
    ey_bg, hx_bg, hz_bg = layered_background_eh(z, physics)
    Ey_s = Ey - ey_bg[:, None]
    Hx_s = Hx - hx_bg[:, None]
    Hz_s = Hz - hz_bg[:, None]
    region, eps = region_id_and_epsilon(x, z, physics)
    amps = _compute_amplitudes_and_energy(
        state["c_bwd_air_top"], state["c_fwd_sub_top"],
        state["kz_air"], state["kz_sub"], state["kx"], n_harmonics,
    )
    interfaces = build_interfaces(state, x, z, physics)
    layers = layer_modal_export(state)
    cfg = generator_config_payload(physics, n_harmonics, nfine)
    arrays = {
        "schema_version": _scalar_str(SCHEMA_VERSION),
        "canonical_npz_sha256": _scalar_str(sha256_file(canonical_path)),
        "generator_commit": _scalar_str(git_commit()),
        "generator_config_hash": _scalar_str(config_hash(cfg)),
        "dtype": _scalar_str("complex128"),
        "time_convention": _scalar_str(TIME_CONVENTION),
        "polarization_convention": _scalar_str(POLARIZATION_CONVENTION),
        "x": x,
        "z": z,
        "region_id": region,
        "epsilon_r": eps,
        "Ey_total": np.asarray(Ey, dtype=np.complex128),
        "Hx_total": np.asarray(Hx, dtype=np.complex128),
        "Hz_total": np.asarray(Hz, dtype=np.complex128),
        "Ey_scattered": np.asarray(Ey_s, dtype=np.complex128),
        "Hx_scattered": np.asarray(Hx_s, dtype=np.complex128),
        "Hz_scattered": np.asarray(Hz_s, dtype=np.complex128),
        "kx": np.asarray(state["kx"], dtype=np.float64),
        "kz_air": np.asarray(state["kz_air"], dtype=np.complex128),
        "kz_substrate": np.asarray(state["kz_sub"], dtype=np.complex128),
        "c_refl": np.asarray(state["c_bwd_air_top"], dtype=np.complex128),
        "c_trans": np.asarray(state["c_fwd_sub_top"], dtype=np.complex128),
        "powers": np.stack([amps["R_m"], amps["T_m"]]).astype(np.float64),
        **interfaces,
        **layers,
    }
    return arrays, canonical, physics, state


def array_inventory(arrays: dict) -> tuple[dict, dict, dict]:
    shapes = {}
    dtypes = {}
    checksums = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        shapes[key] = list(arr.shape)
        dtypes[key] = str(arr.dtype)
        checksums[key] = sha256_array(arr)
    return shapes, dtypes, checksums


def _refuse_overwrite(dest: Path, canonical_hash: str) -> None:
    if not dest.exists():
        return
    with np.load(dest, allow_pickle=False) as old:
        if "canonical_npz_sha256" not in old:
            raise FileExistsError(
                f"Companion {dest} exists without a canonical hash; refusing to overwrite"
            )
        old_hash = str(old["canonical_npz_sha256"].item())
    if old_hash != canonical_hash:
        raise FileExistsError(
            f"Companion {dest} is bound to canonical SHA-256 {old_hash}, "
            f"not {canonical_hash}; refusing to overwrite"
        )


def write_companion(
    canonical_path: Path,
    output_dir: Path,
    nfine: int = NFINE_DEFAULT,
    filename: str = COMPANION_FILENAME,
) -> dict:
    canonical_path = Path(canonical_path).resolve()
    output_dir = Path(output_dir)
    dest = output_dir / filename
    arrays, canonical, physics, state = build_companion_arrays(canonical_path, nfine=nfine)
    canonical_hash = str(arrays["canonical_npz_sha256"].item())
    _refuse_overwrite(dest, canonical_hash)

    report = validate_companion(arrays, canonical, physics, state)
    if not report["passed"]:
        failed = [k for k, v in report["gates"].items() if not v]
        raise RuntimeError(
            "Companion failed validation gates "
            + ", ".join(failed)
            + f": { {k: report[k] for k in ('max_reconstruction_error','boundary_coefficient_error','residual_homogeneous_slabs','vertical_Ey_Hz_jump','horizontal_Ey_Hx_jump') } }"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **arrays)
    companion_hash = sha256_file(dest)
    shapes, dtypes, checksums = array_inventory(arrays)
    manifest = {
        "companion_path": str(dest),
        "canonical_path": str(canonical_path),
        "canonical_sha256": canonical_hash,
        "companion_sha256": companion_hash,
        "generator_commit": str(arrays["generator_commit"].item()),
        "generator_config_hash": str(arrays["generator_config_hash"].item()),
        "schema_version": SCHEMA_VERSION,
        "dtype": "complex128",
        "time_convention": TIME_CONVENTION,
        "polarization_convention": POLARIZATION_CONVENTION,
        "scattered_definition": SCATTERED_DEFINITION,
        "array_shapes": shapes,
        "array_dtypes": dtypes,
        "array_sha256": checksums,
        "powers_R_total": float(np.asarray(canonical["R_total"])),
        "powers_T_total": float(np.asarray(canonical["T_total"])),
        "max_reconstruction_error": report["max_reconstruction_error"],
        "boundary_coefficient_error": report["boundary_coefficient_error"],
        "residual_norms_by_region": report["residual_norms_by_region"],
        "residual_homogeneous_slabs": report["residual_homogeneous_slabs"],
        "interface_jump_norms": report["interface_jump_norms"],
        "gates": report["gates"],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def write_canonical_snapshot_for_tests(
    path: Path,
    physics: PhysicsConfig,
    n_harmonics: int = 7,
    nfine: int = 256,
) -> Path:
    """Write a canonical-format NPZ without H fields (tests only)."""
    from scripts.generate_reference import solve_rcwa

    x, z, e_r, e_i, amps = solve_rcwa(physics, N_harmonics=n_harmonics, Nfine=nfine)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=x, z=z, E_real=e_r, E_imag=e_i,
        field_representation="total",
        wavelength=physics.wavelength,
        period=physics.period,
        ridge_width=physics.ridge_width,
        ridge_height=physics.ridge_height,
        ridge_base_fraction=physics.ridge_base_fraction,
        n_air=physics.n_air,
        n_ridge=physics.n_ridge,
        n_substrate=physics.n_substrate,
        k0=physics.k0,
        domain_height=physics.domain_height,
        coordinate_convention="z=0 top, z increases downward, E_inc=exp(-ik0*z)",
        geometry_convention="dielectric ridge on substrate; substrate outside ridge footprint",
        z_top_monitor=0.08 * physics.domain_height,
        z_bot_monitor=0.92 * physics.domain_height,
        c_refl=amps["c_refl"], c_trans=amps["c_trans"],
        kz_air=amps["kz_air"], kz_sub=amps["kz_sub"], kx=amps["kx"],
        R_m=amps["R_m"], T_m=amps["T_m"],
        R_total=amps["R_total"], T_total=amps["T_total"],
    )
    return path
