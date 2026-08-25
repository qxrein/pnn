#!/usr/bin/env python3
"""Frozen final-head SVD least-squares equation-closure audit.

Diagnostic only.  Hidden z-MLP layers, constant modal coefficients, and all
non-final-head parameters stay frozen.  RCWA modal amplitudes are withheld
from the objective and used only after the solve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.field_comparison import compare_modal_with_rcwa, extract_modal_amplitudes
from src.geometry import epsilon_r
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import (
    compute_background_coefficients,
    maxwell_2d_lbg_pde_residual,
)
from src.modal_dtn import modal_dtn_pointwise
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase5_conditioning import physical_scales
from scripts.run_phase6_explicit_modal import jsonable, metrics
from scripts.train_lbg import make_lambda_0p8

TARGET_T1 = 0.049410786
N_DTN_ORDERS = 8
MODAL_ORDER_MAX = 3
HEAD_PARAM_COUNT = 1386
SEED = 42
RCOND = 1e-12
PHASE_VALIDITY_THRESHOLD = 1e-8
N_COLLOCATION = (64, 32, 32, 32)
CANONICAL_REF = "outputs/reference_lambda_0p8_geometry_consistent_20260824.npz"
COMPANION_PATH = f"outputs/reference_companion/{COMPANION_FILENAME}"
VALIDATED_SCALES = {
    "E_star": 1.18281,
    "H_star": 1.18367,
    "pde_ridge_4": 2.66131,
    "pde_ridge_5": 2.66131,
    "pde_substrate_4": 2.48685,
    "pde_substrate_5": 2.48685,
    "top_DtN": 2.36648,
    "bottom_DtN": 2.89874,
}
PDE_COMPONENT = ("Ar", "Ai", "Br", "Bi", "Cr", "Ci")
PDE_PART = ("real", "imag", "real", "imag", "real", "imag")
BLOCK_ORDER = (
    *(f"pde_air_{i}" for i in range(6)),
    *(f"pde_ridge_{i}" for i in range(6)),
    *(f"pde_substrate_{i}" for i in range(6)),
    "vertical_Ey",
    "vertical_Hz",
    "horizontal_Ey",
    "horizontal_Hx",
    "top_DtN",
    "bottom_DtN",
)
REGION_OF_BLOCK = {
    **{f"pde_air_{i}": "air" for i in range(6)},
    **{f"pde_ridge_{i}": "ridge" for i in range(6)},
    **{f"pde_substrate_{i}": "substrate" for i in range(6)},
    "vertical_Ey": "ridge",
    "vertical_Hz": "ridge",
    "horizontal_Ey": "horizontal_interfaces",
    "horizontal_Hx": "horizontal_interfaces",
    "top_DtN": "air",
    "bottom_DtN": "substrate",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def head_parameters(model: ExplicitFourierModalDD) -> list[torch.nn.Parameter]:
    params = []
    for net in (model.net_air, model.net_grat, model.net_sub):
        last = net.coefficient_mlp[-1]
        params.extend([last.weight, last.bias])
    return params


def head_parameter_records(model: ExplicitFourierModalDD) -> list[dict]:
    records = []
    offset = 0
    for subnet, net in (("air", model.net_air), ("ridge", model.net_grat), ("substrate", model.net_sub)):
        last = net.coefficient_mlp[-1]
        for kind, parameter in (("weight", last.weight), ("bias", last.bias)):
            name = f"net_{subnet}.coefficient_mlp.2.{kind}"
            records.append({
                "name": name,
                "subnet": subnet,
                "kind": kind,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "offset": offset,
                "requires_grad": bool(parameter.requires_grad),
            })
            offset += int(parameter.numel())
    return records


def freeze_except_head(model: ExplicitFourierModalDD) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = head_parameters(model)
    for parameter in heads:
        parameter.requires_grad_(True)
    n = sum(p.numel() for p in heads)
    if n != HEAD_PARAM_COUNT:
        raise RuntimeError(f"expected {HEAD_PARAM_COUNT} head parameters, found {n}")
    return heads


def pack_head(heads: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in heads])


def unpack_head(heads: list[torch.nn.Parameter], theta: np.ndarray | torch.Tensor) -> None:
    vec = torch.as_tensor(theta, dtype=heads[0].dtype, device=heads[0].device)
    offset = 0
    for parameter in heads:
        n = parameter.numel()
        parameter.data.copy_(vec[offset:offset + n].reshape(parameter.shape))
        offset += n
    if offset != vec.numel():
        raise RuntimeError("head parameter packing mismatch")


def _cat_real_imag(*parts: torch.Tensor) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in parts])


def residual_blocks(model, pts, physics, coeff, n_dtn_orders: int = N_DTN_ORDERS) -> dict[str, torch.Tensor]:
    """Pointwise production residuals; complex fields split into real/imag rows."""
    blocks: dict[str, torch.Tensor] = {}
    p = physics
    for name, net, key, eps in (
        ("air", model.net_air, "air", p.eps_air),
        ("ridge", model.net_grat, "grat", None),
        ("substrate", model.net_sub, "sub", p.eps_substrate),
    ):
        x, z = pts[f"x_{key}"], pts[f"z_{key}"]
        eps_val = epsilon_r(x, z, p) if name == "ridge" else eps
        residuals = maxwell_2d_lbg_pde_residual(net, x, z, p, eps_val, coeff)
        for i, residual in enumerate(residuals):
            blocks[f"pde_{name}_{i}"] = residual.reshape(-1)

    xl, xr = p.ridge_x_min, p.ridge_x_max
    offset = 1e-5
    z_left, z_right = pts["z_vleft"], pts["z_vright"]
    ey_v, hz_v = [], []
    for x_iface, z_pts in ((xl, z_left), (xr, z_right)):
        xa = torch.full_like(z_pts, x_iface - offset)
        xb = torch.full_like(z_pts, x_iface + offset)
        elr, eli, _, _, hzlr, hzli = model.net_grat.field_components(xa, z_pts)
        err, eri, _, _, hzrr, hzri = model.net_grat.field_components(xb, z_pts)
        ey_v.extend([elr - err, eli - eri])
        hz_v.extend([hzlr - hzrr, hzli - hzri])
    blocks["vertical_Ey"] = _cat_real_imag(*ey_v)
    blocks["vertical_Hz"] = _cat_real_imag(*hz_v)

    ey_h, hx_h = [], []
    for net_a, net_b, z_int, x_key in (
        (model.net_air, model.net_grat, p.ridge_z_min, "x_int1"),
        (model.net_grat, model.net_sub, p.ridge_z_max, "x_int2"),
    ):
        x_pts = pts[x_key]
        z = torch.full_like(x_pts, z_int)
        era, eia, hra, hia, _, _ = net_a.field_components(x_pts, z)
        erb, eib, hrb, hib, _, _ = net_b.field_components(x_pts, z)
        ey_h.extend([era - erb, eia - eib])
        hx_h.extend([hra - hrb, hia - hib])
    blocks["horizontal_Ey"] = _cat_real_imag(*ey_h)
    blocks["horizontal_Hx"] = _cat_real_imag(*hx_h)

    dhr_t, dhi_t = modal_dtn_pointwise(
        model.net_air, pts["x_top"], 0.0, p, "top", p.n_air, n_dtn_orders,
    )
    dhr_b, dhi_b = modal_dtn_pointwise(
        model.net_sub, pts["x_bot"], p.domain_height, p, "bottom", p.n_substrate, n_dtn_orders,
    )
    blocks["top_DtN"] = _cat_real_imag(dhr_t, dhi_t)
    blocks["bottom_DtN"] = _cat_real_imag(dhr_b, dhi_b)
    return blocks


def pack_residual(blocks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str], np.ndarray]:
    missing = [k for k in BLOCK_ORDER if k not in blocks]
    if missing:
        raise KeyError(f"missing residual blocks: {missing}")
    pieces = [blocks[k].reshape(-1) for k in BLOCK_ORDER]
    lengths = np.array([int(p.numel()) for p in pieces], dtype=np.int64)
    return torch.cat(pieces), list(BLOCK_ORDER), lengths


def _chunk_metadata(name, length, n_chunk, labels, field, region, interface):
    if length % n_chunk != 0:
        raise RuntimeError(f"{name} length {length} is not divisible by {n_chunk}")
    n = length // n_chunk
    rows = []
    for chunk, (part, iface) in enumerate(labels):
        for sample in range(n):
            rows.append({
                "block": name,
                "region": region,
                "interface": iface,
                "field_component": field,
                "real_or_imag": part,
                "sample_index": sample,
                "modal_order": None,
            })
    return rows


def row_metadata(lengths: np.ndarray, pts) -> list[dict]:
    rows: list[dict] = []
    for name, length in zip(BLOCK_ORDER, lengths):
        n = int(length)
        if name.startswith("pde_"):
            _, region, idx_s = name.split("_")
            idx = int(idx_s)
            for sample in range(n):
                rows.append({
                    "block": name,
                    "region": region,
                    "interface": None,
                    "field_component": PDE_COMPONENT[idx],
                    "real_or_imag": PDE_PART[idx],
                    "sample_index": sample,
                    "modal_order": None,
                })
        elif name in ("vertical_Ey", "vertical_Hz"):
            field = "Ey" if name.endswith("Ey") else "Hz"
            labels = (
                ("real", "ridge_left"),
                ("imag", "ridge_left"),
                ("real", "ridge_right"),
                ("imag", "ridge_right"),
            )
            rows.extend(_chunk_metadata(name, n, 4, labels, field, "ridge", None))
        elif name in ("horizontal_Ey", "horizontal_Hx"):
            field = "Ey" if name.endswith("Ey") else "Hx"
            labels = (
                ("real", "ridge_top"),
                ("imag", "ridge_top"),
                ("real", "ridge_bottom"),
                ("imag", "ridge_bottom"),
            )
            rows.extend(_chunk_metadata(name, n, 4, labels, field, "horizontal_interfaces", None))
        elif name in ("top_DtN", "bottom_DtN"):
            field = "Hx"
            region = "air" if name.startswith("top") else "substrate"
            interface = "top" if name.startswith("top") else "bottom"
            labels = (("real", interface), ("imag", interface))
            rows.extend(_chunk_metadata(name, n, 2, labels, field, region, None))
            for row in rows[-n:]:
                row["interface"] = interface
        else:
            raise KeyError(name)
    if len(rows) != int(lengths.sum()):
        raise RuntimeError("row metadata length mismatch")
    return rows


def row_scales(lengths: np.ndarray, scales: dict[str, float]) -> np.ndarray:
    values = []
    for name, length in zip(BLOCK_ORDER, lengths):
        if name.startswith("pde_"):
            scale = scales[name]
        elif name == "vertical_Ey":
            scale = scales["vertical_E"]
        elif name == "vertical_Hz":
            scale = scales["vertical_Hz"]
        elif name == "horizontal_Ey":
            scale = scales["horizontal_E"]
        elif name == "horizontal_Hx":
            scale = scales["horizontal_Hx"]
        elif name == "top_DtN":
            scale = scales["top_DtN"]
        elif name == "bottom_DtN":
            scale = scales["bottom_DtN"]
        else:
            raise KeyError(name)
        values.append(np.full(length, float(scale), dtype=np.float64))
    return np.concatenate(values)


def assemble_system(model, heads, pts, physics, coeff) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unpack_head(heads, np.zeros(HEAD_PARAM_COUNT))
    r0, _, lengths = pack_residual(residual_blocks(model, pts, physics, coeff))
    r0_np = r0.detach().cpu().numpy().astype(np.float64)
    n_rows = r0_np.size
    jacobian = np.zeros((n_rows, HEAD_PARAM_COUNT), dtype=np.float64)
    theta = np.zeros(HEAD_PARAM_COUNT, dtype=np.float64)
    for j in range(HEAD_PARAM_COUNT):
        theta[j] = 1.0
        unpack_head(heads, theta)
        residual, _, _ = pack_residual(residual_blocks(model, pts, physics, coeff))
        jacobian[:, j] = residual.detach().cpu().numpy() - r0_np
        theta[j] = 0.0
        if (j + 1) % 50 == 0 or j + 1 == HEAD_PARAM_COUNT:
            print(f"  assembled column {j + 1}/{HEAD_PARAM_COUNT}", flush=True)
    unpack_head(heads, np.zeros(HEAD_PARAM_COUNT))
    return jacobian, r0_np, lengths


def affine_check(jacobian, r0, model, heads, pts, physics, coeff, rng) -> dict:
    thetas = [rng.standard_normal(HEAD_PARAM_COUNT) for _ in range(4)]
    thetas.append(rng.standard_normal(HEAD_PARAM_COUNT) * 1e-3)
    max_abs = 0.0
    rels = []
    sq = []
    for theta in thetas:
        unpack_head(heads, theta)
        residual, _, _ = pack_residual(residual_blocks(model, pts, physics, coeff))
        direct = residual.detach().cpu().numpy().astype(np.float64)
        predicted = jacobian @ theta + r0
        err = direct - predicted
        max_abs = max(max_abs, float(np.max(np.abs(err))))
        rels.append(float(np.linalg.norm(err) / (np.linalg.norm(direct) + 1e-30)))
        sq.append(float(np.sqrt(np.mean(err ** 2))))
    unpack_head(heads, np.zeros(HEAD_PARAM_COUNT))
    return {
        "n_test_vectors": len(thetas),
        "max_absolute_affine_error": max_abs,
        "relative_affine_error": float(max(rels)),
        "rms_affine_error": float(np.sqrt(np.mean(np.square(sq)))),
        "expected_max_absolute_error": 3.55e-15,
    }


def svd_least_squares(matrix: np.ndarray, rhs_plus_b: np.ndarray, rcond: float = RCOND) -> dict:
    """Minimize ||A @ theta + b||_2 with column scaling and an explicit SVD cutoff."""
    col_norm = np.linalg.norm(matrix, axis=0)
    positive = col_norm[col_norm > 0]
    col_scale_floor = 1e-12 * float(max(np.max(col_norm), 1.0))
    scale = np.maximum(col_norm, col_scale_floor)
    scaled = matrix / scale
    rhs = -rhs_plus_b
    _, sigma_unscaled, _ = np.linalg.svd(matrix, full_matrices=False)
    u, sigma, vt = np.linalg.svd(scaled, full_matrices=False)
    if sigma.size == 0 or sigma[0] == 0.0:
        raise RuntimeError("SVD produced an empty or zero spectrum")
    diagnostic_ranks = {
        str(tol): int(np.sum(sigma > float(tol) * sigma[0]))
        for tol in (1e-8, 1e-10, 1e-12, 1e-14)
    }
    rank = int(np.sum(sigma > rcond * sigma[0]))
    inv = np.zeros_like(sigma)
    inv[:rank] = 1.0 / sigma[:rank]
    q = vt.T @ (inv * (u.T @ rhs))
    theta = q / scale
    residual = matrix @ theta + rhs_plus_b
    cond = float(sigma[0] / sigma[rank - 1]) if rank else float("inf")
    near_zero = np.where(col_norm <= col_scale_floor)[0]
    return {
        "theta": theta,
        "sigma": sigma,
        "sigma_unscaled": sigma_unscaled,
        "rank": rank,
        "effective_rank": diagnostic_ranks["1e-08"],
        "diagnostic_ranks": diagnostic_ranks,
        "rcond": rcond,
        "condition_number": cond,
        "col_norm": col_norm,
        "col_scale": scale,
        "column_scale_floor": col_scale_floor,
        "near_zero_columns": near_zero.tolist(),
        "n_near_zero_columns": int(near_zero.size),
        "column_scale_min": float(np.min(scale)),
        "column_scale_max": float(np.max(scale)),
        "residual": residual,
        "n_rows": int(matrix.shape[0]),
        "n_cols": int(matrix.shape[1]),
        "q": q,
    }


def slice_by_block(vector: np.ndarray, lengths: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    offset = 0
    for name, length in zip(BLOCK_ORDER, lengths):
        out[name] = vector[offset:offset + int(length)]
        offset += int(length)
    return out


def block_norms(vector: np.ndarray, lengths: np.ndarray) -> dict[str, float]:
    pieces = slice_by_block(vector, lengths)
    out = {name: float(np.linalg.norm(piece)) for name, piece in pieces.items()}
    out["all"] = float(np.linalg.norm(vector))
    return out


def grouped_norms(vector: np.ndarray, lengths: np.ndarray, metadata: list[dict]) -> dict:
    by_region: dict[str, list[float]] = {}
    by_interface: dict[str, list[float]] = {}
    by_part: dict[str, list[float]] = {}
    for value, row in zip(vector, metadata):
        by_region.setdefault(row["region"], []).append(value)
        if row["interface"] is not None:
            by_interface.setdefault(row["interface"], []).append(value)
        by_part.setdefault(row["real_or_imag"], []).append(value)
    return {
        "by_block": block_norms(vector, lengths),
        "by_region": {k: float(np.linalg.norm(v)) for k, v in by_region.items()},
        "by_interface": {k: float(np.linalg.norm(v)) for k, v in by_interface.items()},
        "by_real_imag": {k: float(np.linalg.norm(v)) for k, v in by_part.items()},
    }


def cx(value: complex) -> dict:
    return {"real": float(np.real(value)), "imag": float(np.imag(value)), "abs": float(abs(value))}


def phase_error_deg(pred: complex, ref: complex) -> float | None:
    if abs(pred) < PHASE_VALIDITY_THRESHOLD:
        return None
    return float(abs(np.angle(pred / (ref + 1e-30))) * 180.0 / np.pi)


def spatial_fourier_t(model, physics, z_bot: float, n_quad: int = 512) -> dict[int, complex]:
    x = torch.linspace(0.0, physics.period, n_quad + 1, dtype=torch.float64)[:-1].requires_grad_(True)
    z = torch.full_like(x.detach(), z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    e = (er + 1j * ei).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    g0 = 2.0 * np.pi / physics.period
    out = {}
    for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1):
        out[m] = complex(np.mean(e * np.exp(-1j * m * g0 * x_np)))
    return out


def production_coefficient_t(model, physics, z_bot: float) -> dict[int, complex]:
    z = torch.tensor([z_bot], dtype=torch.float64)
    coeff = model.net_sub.modal_coefficients_at(z)[0].detach().cpu().numpy()
    return {int(m): complex(coeff[i]) for i, m in enumerate(model.net_sub.orders)}


def companion_t(physics, z_bot: float, path: Path) -> dict[int, complex]:
    data = np.load(path, allow_pickle=False)
    n = int((len(data["c_trans"]) - 1) // 2)
    out = {}
    for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1):
        idx = n + m
        t = complex(data["c_trans"][idx])
        t *= np.exp(-1j * data["kz_substrate"][idx] * (z_bot - physics.ridge_z_max))
        out[m] = t
    return out


def modal_report(model, physics, coeff, device, dtype, ref, ref_path, companion_path: Path) -> dict:
    fields, dual, modal, cmp = metrics(model, physics, coeff, device, dtype, ref, ref_path)
    x, z = fields["x"][0], fields["z"][:, 0]
    modal3 = extract_modal_amplitudes(
        fields["E_real"] + 1j * fields["E_imag"], x, z, physics,
        n_orders=3, formulation="layered_bg", field_representation="total",
    )
    cmp3 = compare_modal_with_rcwa(
        modal3, str(ref_path), int((len(np.load(ref_path)["c_refl"]) - 1) // 2), physics=physics,
    )
    z_bot = float(modal3["z_bot_monitor"])
    t_prod = {int(m): complex(np.asarray(modal3["t_m_complex"])[i])
              for i, m in enumerate(range(-3, 4))}
    t_coeff = production_coefficient_t(model, physics, z_bot)
    t_quad = spatial_fourier_t(model, physics, z_bot)
    t_rcwa = companion_t(physics, z_bot, companion_path)
    orders = list(range(-3, 4))
    per_m = {}
    for m in orders:
        pred = t_prod[m]
        ref_t = t_rcwa[m]
        err = pred - ref_t
        mag_err = abs(pred) - abs(ref_t)
        per_m[str(m)] = {
            "t_production": cx(pred),
            "t_coefficient": cx(t_coeff[m]),
            "t_spatial_fourier": cx(t_quad[m]),
            "t_rcwa_companion": cx(ref_t),
            "abs": abs(pred),
            "complex_error": cx(err),
            "magnitude_error": float(mag_err),
            "phase_error_deg": phase_error_deg(pred, ref_t),
            "phase_valid": abs(pred) >= PHASE_VALIDITY_THRESHOLD,
        }
    return {
        "fields_meta": {k: dual[k] for k in dual if not isinstance(dual[k], np.ndarray)},
        "t_m": per_m,
        "t_minus1_abs": abs(t_prod[-1]),
        "t_plus1_abs": abs(t_prod[1]),
        "t_minus1_complex_error": abs(t_prod[-1] - t_rcwa[-1]),
        "t_plus1_complex_error": abs(t_prod[1] - t_rcwa[1]),
        "m0_phase_error_deg": per_m["0"]["phase_error_deg"],
        "R": float(modal3["R_total"]),
        "T": float(modal3["T_total"]),
        "R_plus_T": float(modal3["energy_check"]),
        "total_complex_l2": float(dual["total/complex_l2"]),
        "scattered_complex_l2": float(dual["scattered/complex_l2"]),
        "phase_validity_threshold": PHASE_VALIDITY_THRESHOLD,
        "z_bot_monitor": z_bot,
        "modal_comparison": cmp3,
        "modal_n1": {k: v for k, v in modal.items() if not isinstance(v, np.ndarray)},
        "cmp_n1": cmp,
        "dual": {k: v for k, v in dual.items() if not isinstance(v, np.ndarray)},
    }


def model_config_hash(physics) -> str:
    payload = {
        "architecture": "ExplicitFourierModalDD",
        "modal_order_max": MODAL_ORDER_MAX,
        "use_coefficient_mlp": True,
        "n_dtn_orders": N_DTN_ORDERS,
        "seed": SEED,
        "n_collocation": N_COLLOCATION,
        "wavelength": physics.wavelength,
        "period": physics.period,
        "ridge_width": physics.ridge_width,
        "ridge_height": physics.ridge_height,
        "domain_height": physics.domain_height,
        "ridge_base_fraction": physics.ridge_base_fraction,
        "n_air": physics.n_air,
        "n_ridge": physics.n_ridge,
        "n_substrate": physics.n_substrate,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def confirm_validated_scales(scales: dict[str, float], meta: dict) -> None:
    checks = {
        "E_star": meta["E_star"],
        "H_star": meta["H_star"],
        "pde_ridge_4": scales["pde_ridge_4"],
        "pde_ridge_5": scales["pde_ridge_5"],
        "pde_substrate_4": scales["pde_substrate_4"],
        "pde_substrate_5": scales["pde_substrate_5"],
        "top_DtN": scales["top_DtN"],
        "bottom_DtN": scales["bottom_DtN"],
    }
    for key, actual in checks.items():
        expected = VALIDATED_SCALES[key]
        if abs(actual - expected) / expected > 5e-4:
            raise RuntimeError(f"physical scale {key}={actual} disagrees with validated {expected}")


def load_points(physics, device, dtype, shared_path: Path | None):
    if shared_path is not None and shared_path.exists():
        data = np.load(shared_path)
        return {k: torch.as_tensor(data[k], dtype=dtype, device=device) for k in data.files}
    return sample_nd_points(physics, *N_COLLOCATION, device, dtype, seed=SEED)


def load_baseline_500() -> dict:
    history = ROOT / "outputs/phase5_conditioning/variant_a_baseline/history.csv"
    if not history.exists():
        return {"available": False}
    with history.open() as fh:
        rows = list(csv.DictReader(fh))
    row = next((r for r in rows if int(float(r["epoch"])) == 500), None)
    if row is None:
        return {"available": False}
    return {
        "available": True,
        "epoch": 500,
        "source": str(history),
        "t_minus1_abs": float(row["t_minus1"]),
        "t_plus1_abs": float(row["t_plus1"]),
        "total_complex_l2": float(row["total_complex_l2"]),
        "scattered_complex_l2": float(row["scattered_complex_l2"]),
        "R_plus_T": float(row["R_plus_T"]),
        "numerical_rank": None,
        "condition_number": None,
        "residual_before": None,
        "residual_after": None,
        "residual_reduction": None,
        "t_minus1_complex_error": None,
        "t_plus1_complex_error": None,
        "final_head_norm": None,
    }


def write_variant_dir(directory: Path, kind: str, solved: dict, jacobian: np.ndarray,
                      r0: np.ndarray, lengths: np.ndarray, metadata: list[dict],
                      scales: dict | None, affine: dict, freeze_info: dict,
                      before_groups: dict, after_groups: dict, after_modal: dict,
                      reeval: dict, extra: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    before_norm = float(np.linalg.norm(r0 if kind == "raw" else extra["b_used"]))
    after_lin = float(np.linalg.norm(solved["residual"]))
    raw_before = float(np.linalg.norm(r0))
    raw_after = float(np.linalg.norm(jacobian @ solved["theta"] + r0))
    system_metadata = {
        "variant": kind,
        "matrix_shape": [solved["n_rows"], solved["n_cols"]],
        "n_rows": solved["n_rows"],
        "n_cols": solved["n_cols"],
        "rcond": solved["rcond"],
        "column_scale_floor": solved["column_scale_floor"],
        "n_near_zero_columns": solved["n_near_zero_columns"],
        "near_zero_columns": solved["near_zero_columns"],
        "column_scale_min": solved["column_scale_min"],
        "column_scale_max": solved["column_scale_max"],
        "numerical_rank": solved["rank"],
        "effective_rank": solved["effective_rank"],
        "diagnostic_ranks": solved["diagnostic_ranks"],
        "condition_number": solved["condition_number"],
        "residual_norm_before": before_norm,
        "residual_norm_after": after_lin,
        "residual_reduction_factor": before_norm / max(after_lin, 1e-30),
        "raw_norm_before": raw_before,
        "raw_norm_after": raw_after,
        "final_head_norm": float(np.linalg.norm(solved["theta"])),
        "final_head_max_abs": float(np.max(np.abs(solved["theta"]))),
        "physical_scales": scales,
        "block_averaging": False,
        "loss_weights": None,
        "affine_verification": affine,
        **freeze_info,
        **extra,
    }
    matrix_meta = {
        "blocks": list(BLOCK_ORDER),
        "lengths": [int(v) for v in lengths],
        "rows": metadata,
        "parameter_records": freeze_info["parameter_records"],
        "parameter_ordering": "concat(net_air, net_grat, net_sub) last Linear weight then bias, C-order",
    }
    report = {
        **system_metadata,
        "residual_groups_before": before_groups,
        "residual_groups_after_linear": grouped_norms(solved["residual"], lengths, metadata)
        if kind != "physical" else extra.get("physical_groups_after_linear"),
        "residual_groups_after_eval": after_groups,
        "modal": after_modal,
        "reevaluation": reeval,
        "t_minus1_abs": after_modal["t_minus1_abs"],
        "t_plus1_abs": after_modal["t_plus1_abs"],
        "t_minus1_complex_error": after_modal["t_minus1_complex_error"],
        "t_plus1_complex_error": after_modal["t_plus1_complex_error"],
        "m0_phase_error_deg": after_modal["m0_phase_error_deg"],
        "R_plus_T": after_modal["R_plus_T"],
        "total_complex_l2": after_modal["total_complex_l2"],
        "scattered_complex_l2": after_modal["scattered_complex_l2"],
        "target_t_pm1": TARGET_T1,
        "modal_loss_weight": 0.0,
        "optical_coupling": False,
    }
    (directory / "system_metadata.json").write_text(json.dumps(jsonable(system_metadata), indent=2) + "\n")
    (directory / "residual_matrix_metadata.json").write_text(json.dumps(jsonable(matrix_meta), indent=2) + "\n")
    report_name = "raw_solution_report.json" if kind == "raw" else "physical_solution_report.json"
    (directory / report_name).write_text(json.dumps(jsonable(report), indent=2) + "\n")
    np.save(directory / "singular_values.npy", solved["sigma"])
    np.save(directory / "theta_solution.npy", solved["theta"])
    np.save(directory / "singular_values_unscaled.npy", solved["sigma_unscaled"])
    return report


def reevaluate(model, heads, theta, jacobian, r0, pts, physics, coeff, lengths, metadata,
               physics_eval, device, dtype, ref, ref_path, companion_path, checkpoint_path,
               freeze_info) -> dict:
    unpack_head(heads, theta)
    r_eval, _, _ = pack_residual(residual_blocks(model, pts, physics, coeff))
    direct = r_eval.detach().cpu().numpy().astype(np.float64)
    predicted = jacobian @ theta + r0
    affine_err = direct - predicted
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "theta": np.asarray(theta),
        "architecture": "ExplicitFourierModalDD",
        "modal_order_max": MODAL_ORDER_MAX,
        **{k: freeze_info[k] for k in (
            "random_seed", "model_configuration_hash", "reference_hash", "companion_hash",
        ) if k in freeze_info},
    }, checkpoint_path)
    after_modal = modal_report(model, physics, coeff, device, dtype, ref, ref_path, companion_path)
    after_groups = grouped_norms(direct, lengths, metadata)
    return {
        "direct_residual_norm": float(np.linalg.norm(direct)),
        "affine_predicted_residual_norm": float(np.linalg.norm(predicted)),
        "direct_minus_affine_max_abs": float(np.max(np.abs(affine_err))),
        "direct_minus_affine_rms": float(np.sqrt(np.mean(affine_err ** 2))),
        "direct": direct,
        "modal": after_modal,
        "groups": after_groups,
        "checkpoint": str(checkpoint_path),
    }


def reload_in_subprocess(checkpoint: Path, points: Path, output: Path) -> dict:
    cmd = [
        sys.executable, str(ROOT / "scripts/run_frozen_head_least_squares.py"),
        "--verify-checkpoint", str(checkpoint),
        "--shared-points", str(points),
        "--verify-output", str(output),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    return json.loads(output.read_text())


def decide(raw_report: dict, physical_report: dict, affine: dict, reeval_ok: bool) -> dict:
    if not reeval_ok:
        return {
            "case": "D",
            "decision": (
                "Production-model re-evaluation disagrees with A @ theta + b. "
                "Stop: fix parameter assignment, output ordering, or residual assembly. "
                "Do not start training."
            ),
            "start_10000_epoch_run": False,
            "modal_data_loss": False,
            "optical_coupling": False,
            "authorize_nonlinear_from_ls_head": False,
        }

    def recovered(report):
        t_m = report["t_minus1_abs"]
        t_p = report["t_plus1_abs"]
        rel_m = abs(t_m - TARGET_T1) / TARGET_T1
        rel_p = abs(t_p - TARGET_T1) / TARGET_T1
        complex_ok = (report["t_minus1_complex_error"] < 0.25 * TARGET_T1 and
                      report["t_plus1_complex_error"] < 0.25 * TARGET_T1)
        ok = (rel_m < 0.25) and (rel_p < 0.25) and complex_ok
        return ok, rel_m, rel_p, t_m, t_p

    ok_a, rel_am, rel_ap, raw_m, raw_p = recovered(raw_report)
    ok_b, rel_bm, rel_bp, phys_m, phys_p = recovered(physical_report)
    near_old = all(0.003 <= v <= 0.008 for v in (raw_m, raw_p, phys_m, phys_p))

    raw_frac = raw_report["residual_groups_after_eval"]["by_block"]
    phys_frac = physical_report["residual_groups_after_eval"]["by_block"]
    differing = []
    for name in BLOCK_ORDER:
        a = raw_frac[name] / max(raw_frac["all"], 1e-30)
        b = phys_frac[name] / max(phys_frac["all"], 1e-30)
        if abs(a - b) > 0.05:
            differing.append({"block": name, "raw_fraction": a, "physical_fraction": b})

    if ok_a or ok_b:
        case = "A"
        decision = (
            "The current frozen final-head basis can represent the target ±1 response "
            "under the residual system. Do not automatically start long training. "
            "The least-squares head is a candidate initialization for a later optimizer experiment."
        )
        failure_mode = None
    elif near_old:
        ranks = {"raw": raw_report["numerical_rank"], "physical": physical_report["numerical_rank"]}
        n_cols = raw_report["n_cols"]
        rank_def = min(ranks.values()) < n_cols - 1
        if rank_def:
            failure_mode = "rank_deficiency"
        elif min(raw_report["condition_number"], physical_report["condition_number"]) > 1e8:
            failure_mode = "weak_pm1_sensitive_singular_directions_or_ill_conditioning"
        else:
            failure_mode = (
                "insufficient_z_basis_or_missing_interface_localized_basis_"
                "or_incompatible_pointwise_vs_truncated_FMM_objective"
            )
        case = "B"
        decision = (
            "Both least-squares solutions remain near the previous 0.004–0.006 |t_±1| range. "
            "The frozen final-head basis cannot produce the required transmitted ±1 response "
            "under the current residual system. Do not start 10,000 epochs, modal-data loss, "
            f"or optical coupling. Diagnosed cause: {failure_mode}."
        )
    else:
        case = "B"
        failure_mode = "did_not_recover_target_amplitudes"
        decision = (
            "Neither least-squares solution recovered |t_±1| ≈ 0.04941. "
            "Do not start 10,000 epochs, modal-data loss, or optical coupling."
        )

    if differing and case != "D":
        extra = (
            " Raw and physical least squares produce different residual distributions "
            f"in blocks {', '.join(d['block'] for d in differing)}. "
            "Do not declare either physically superior based only on a lower norm."
        )
        if case != "A":
            case = "C" if case == "B" and differing else case
        decision = decision + extra

    return {
        "case": case,
        "recovered_t_pm1": ok_a or ok_b,
        "raw_recovered": ok_a,
        "physical_recovered": ok_b,
        "raw_relative_t_error": {"m=-1": rel_am, "m=+1": rel_ap},
        "physical_relative_t_error": {"m=-1": rel_bm, "m=+1": rel_bp},
        "differing_blocks": differing,
        "authorize_nonlinear_from_ls_head": ok_a or ok_b,
        "start_10000_epoch_run": False,
        "modal_data_loss": False,
        "optical_coupling": False,
        "failure_mode": None if (ok_a or ok_b) else failure_mode,
        "decision": decision,
        "affine_check": affine,
    }


def write_plots(root: Path, raw: dict, physical: dict, baseline: dict, companion_tm: dict[int, complex]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for label, sigma in (("raw scaled", raw["sigma"]), ("physical scaled", physical["sigma"])):
        ax.semilogy(np.arange(1, len(sigma) + 1), np.maximum(sigma, np.finfo(float).tiny), label=label)
    ax.set(xlabel="singular value index", ylabel="σ", title="Frozen-head SVD spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "singular_value_spectrum.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    names = list(BLOCK_ORDER)
    x = np.arange(len(names))
    raw_b = [raw["before"]["by_block"][n] for n in names]
    raw_a = [raw["after"]["by_block"][n] for n in names]
    phys_b = [physical["before"]["by_block"][n] for n in names]
    phys_a = [physical["after"]["by_block"][n] for n in names]
    axes[0].bar(x - 0.2, raw_b, 0.4, label="raw before")
    axes[0].bar(x + 0.2, raw_a, 0.4, label="raw after")
    axes[0].set_xticks(x, names, rotation=90, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Residual by block")
    regions = sorted(raw["before"]["by_region"])
    xr = np.arange(len(regions))
    axes[1].bar(xr - 0.2, [raw["before"]["by_region"][n] for n in regions], 0.4, label="raw before")
    axes[1].bar(xr + 0.2, [raw["after"]["by_region"][n] for n in regions], 0.4, label="raw after")
    axes[1].set_xticks(xr, regions, rotation=30, fontsize=8)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Residual by region")
    fig.tight_layout()
    fig.savefig(root / "residual_block_comparison.png", dpi=160)
    plt.close(fig)

    orders = list(range(-3, 4))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(orders, [companion_tm[m].__abs__() for m in orders], "k--", label="RCWA companion")
    ax.plot(orders, [raw["modal"]["t_m"][str(m)]["abs"] for m in orders], "o-", label="raw LS")
    ax.plot(orders, [physical["modal"]["t_m"][str(m)]["abs"] for m in orders], "s-", label="physical LS")
    if baseline.get("available"):
        ax.scatter([-1, 1], [baseline["t_minus1_abs"], baseline["t_plus1_abs"]],
                   marker="x", label="500-epoch baseline")
    ax.set(xlabel="m", ylabel="|t_m|", title="Transmitted modal amplitudes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "modal_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for m in orders:
        ref = companion_tm[m]
        ax.plot(ref.real, ref.imag, "k+")
        ax.annotate(f"c{m}", (ref.real, ref.imag), fontsize=7)
        for key, marker in (("raw", "o"), ("physical", "s")):
            val = (raw if key == "raw" else physical)["modal"]["t_m"][str(m)]["t_production"]
            ax.plot(val["real"], val["imag"], marker, label=key if m == 0 else None)
    ax.set(xlabel="Re t_m", ylabel="Im t_m", title="Complex modal plane")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "complex_modal_plane.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x - 0.2, raw_a, 0.4, label="raw after")
    ax.bar(x + 0.2, phys_a, 0.4, label="physical after (raw units)")
    ax.set_xticks(x, names, rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.set_title("Raw versus physical residual contributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "raw_versus_physical_residual_contributions.png", dpi=160)
    plt.close(fig)


def verify_checkpoint(args) -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    device = torch.device("cpu")
    dtype = torch.float64
    set_seed(SEED)
    coeff = compute_background_coefficients(physics)
    pts = load_points(physics, device, dtype, Path(args.shared_points))
    payload = torch.load(args.verify_checkpoint, map_location="cpu", weights_only=False)
    model = ExplicitFourierModalDD(physics, MODAL_ORDER_MAX, True).double()
    model.load_state_dict(payload["state_dict"])
    heads = freeze_except_head(model)
    residual, _, _ = pack_residual(residual_blocks(model, pts, physics, coeff))
    theta = pack_head(heads).cpu().numpy()
    report = {
        "checkpoint": str(args.verify_checkpoint),
        "residual_norm": float(np.linalg.norm(residual.detach().cpu().numpy())),
        "head_norm": float(np.linalg.norm(theta)),
        "head_max_abs": float(np.max(np.abs(theta))),
        "theta": theta.tolist(),
    }
    Path(args.verify_output).write_text(json.dumps(report) + "\n")


def comparison_row(name, report, extra=None) -> dict:
    extra = extra or {}
    return {
        "method": name,
        "numerical_rank": report.get("numerical_rank"),
        "condition_number": report.get("condition_number"),
        "residual_before": report.get("raw_norm_before", report.get("residual_before")),
        "residual_after": report.get("raw_norm_after", report.get("residual_after")),
        "residual_reduction": report.get("residual_reduction_factor", report.get("residual_reduction")),
        "t_minus1_abs": report.get("t_minus1_abs"),
        "t_plus1_abs": report.get("t_plus1_abs"),
        "t_minus1_complex_error": report.get("t_minus1_complex_error"),
        "t_plus1_complex_error": report.get("t_plus1_complex_error"),
        "total_complex_l2": report.get("total_complex_l2"),
        "scattered_complex_l2": report.get("scattered_complex_l2"),
        "R_plus_T": report.get("R_plus_T"),
        "final_head_norm": report.get("final_head_norm", extra.get("final_head_norm")),
    }


def make_model(physics):
    set_seed(SEED)
    return ExplicitFourierModalDD(physics, MODAL_ORDER_MAX, True).double()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-head SVD least-squares equation closure")
    parser.add_argument("--output-root", default="outputs/phase5_equation_closure")
    parser.add_argument(
        "--shared-points",
        default="outputs/phase5_physical_conditioning/shared_points.npz",
    )
    parser.add_argument("--verify-checkpoint", default="")
    parser.add_argument("--verify-output", default="")
    args = parser.parse_args()
    if args.verify_checkpoint:
        verify_checkpoint(args)
        return

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    raw_dir = root / "frozen_head_raw"
    phys_dir = root / "frozen_head_physical"

    cfg = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    ref_path = ROOT / CANONICAL_REF
    companion_path = ROOT / COMPANION_PATH
    ref_meta = validate_reference(ref_path, physics)
    ref = normalize_reference_orientation(load_reference_npz(ref_path))
    device = torch.device("cpu")
    dtype = torch.float64
    set_seed(SEED)
    coeff = compute_background_coefficients(physics)
    pts = load_points(physics, device, dtype, ROOT / args.shared_points)
    scales, scale_meta = physical_scales(pts, physics, coeff)
    confirm_validated_scales(scales, scale_meta)

    model = make_model(physics)
    theta_init = pack_head(head_parameters(model)).cpu().numpy()
    heads = freeze_except_head(model)
    n_train = sum(p.numel() for p in heads)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    records = head_parameter_records(model)
    freeze_info = {
        "n_trainable_final_head_parameters": n_train,
        "n_frozen_parameters": n_frozen,
        "parameter_records": records,
        "parameter_ordering": "concat(net_air, net_grat, net_sub) last Linear weight then bias, C-order",
        "random_seed": SEED,
        "model_configuration_hash": model_config_hash(physics),
        "reference_hash": sha256_file(ref_path),
        "companion_hash": sha256_file(companion_path),
        "generator_commit": git_commit(),
        "initial_final_head_norm": float(np.linalg.norm(theta_init)),
        "initial_final_head_max_abs": float(np.max(np.abs(theta_init))),
        "initial_head_matches_affine_audit": bool(np.max(np.abs(theta_init)) == 0.0),
        "modal_loss_weight": 0.0,
        "optical_coupling": False,
        "canonical_modified": False,
        "companion_used_in_loss": False,
    }
    print(json.dumps({k: freeze_info[k] for k in freeze_info if k != "parameter_records"}, indent=2), flush=True)

    root.mkdir(parents=True)
    np.savez(root / "shared_points.npz", **{k: v.detach().cpu().numpy() for k, v in pts.items()})
    np.save(root / "theta_init.npy", theta_init)

    print("Assembling frozen-head Jacobian (1,386 exact unit columns)...", flush=True)
    jacobian, r0, lengths = assemble_system(model, heads, pts, physics, coeff)
    metadata = row_metadata(lengths, pts)
    print("Verifying affine residual model...", flush=True)
    affine = affine_check(jacobian, r0, model, heads, pts, physics, coeff, np.random.default_rng(12345))
    print(json.dumps(affine, indent=2), flush=True)

    row_scale = row_scales(lengths, scales)
    before_groups = grouped_norms(r0, lengths, metadata)
    before_modal = modal_report(model, physics, coeff, device, dtype, ref, ref_path, companion_path)

    reports = {}
    plot_bundle = {}
    for name, matrix, rhs, directory, scale_dump in (
        ("raw", jacobian, r0, raw_dir, None),
        ("physical", jacobian / row_scale[:, None], r0 / row_scale, phys_dir, scales),
    ):
        print(f"Solving variant {name}...", flush=True)
        solved = svd_least_squares(matrix, rhs, rcond=RCOND)
        fresh = make_model(physics)
        fresh_heads = freeze_except_head(fresh)
        unpack_head(fresh_heads, solved["theta"])
        reeval = reevaluate(
            fresh, fresh_heads, solved["theta"], jacobian, r0, pts, physics, coeff,
            lengths, metadata, physics, device, dtype, ref, ref_path, companion_path,
            directory / "solved_checkpoint.pt", freeze_info,
        )
        extra = {
            "b_used": rhs,
            "physical_norm_before": float(np.linalg.norm(rhs)) if name == "physical" else None,
            "physical_norm_after": float(np.linalg.norm(matrix @ solved["theta"] + rhs))
            if name == "physical" else None,
        }
        if name == "physical":
            extra["physical_groups_after_linear"] = grouped_norms(solved["residual"], lengths, metadata)
            extra["raw_norm_after_physical_solve"] = float(np.linalg.norm(jacobian @ solved["theta"] + r0))
        reports[name] = write_variant_dir(
            directory, name, solved, jacobian, r0, lengths, metadata, scale_dump,
            affine, freeze_info, before_groups, reeval["groups"], reeval["modal"], reeval, extra,
        )
        reports[name]["sigma"] = solved["sigma"]
        plot_bundle[name] = {
            "sigma": solved["sigma"],
            "before": before_groups,
            "after": reeval["groups"],
            "modal": reeval["modal"],
        }
        reload = reload_in_subprocess(
            directory / "solved_checkpoint.pt",
            root / "shared_points.npz",
            directory / "reload_report.json",
        )
        reports[name]["reevaluation"]["reload_residual_norm"] = reload["residual_norm"]
        reports[name]["reevaluation"]["reload_minus_original"] = abs(
            reload["residual_norm"] - reeval["direct_residual_norm"]
        )
        reports[name]["reload"] = reload
        unpack_head(heads, np.zeros(HEAD_PARAM_COUNT))

    reeval_ok = all(
        reports[k]["reevaluation"]["direct_minus_affine_max_abs"] < 1e-10
        and reports[k]["reevaluation"]["reload_minus_original"] < 1e-10
        for k in ("raw", "physical")
    )
    decision = decide(reports["raw"], reports["physical"], affine, reeval_ok)
    baseline = load_baseline_500()
    z_bot = reports["raw"]["modal"]["z_bot_monitor"]
    companion_tm = companion_t(physics, z_bot, companion_path)
    write_plots(root, plot_bundle["raw"], plot_bundle["physical"], baseline, companion_tm)

    rows = [
        comparison_row("raw least squares", reports["raw"]),
        comparison_row("physical-scale least squares", reports["physical"]),
        comparison_row("previous 500-epoch baseline", baseline),
    ]
    with (root / "frozen_head_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "reference": ref_meta,
        "companion_path": str(companion_path),
        "physical_scale_metadata": scale_meta,
        "physical_scales": scales,
        "freeze": freeze_info,
        "affine_verification": affine,
        "before_solve": {"residual": before_groups, "modal": before_modal},
        "raw": {k: v for k, v in reports["raw"].items() if k != "sigma"},
        "physical": {k: v for k, v in reports["physical"].items() if k != "sigma"},
        "comparison": rows,
        "baseline_500_epoch": baseline,
        **decision,
    }
    (root / "frozen_head_summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps(jsonable({
        "case": decision["case"],
        "decision": decision["decision"],
        "affine_max_abs": affine["max_absolute_affine_error"],
        "raw_|t-1|": reports["raw"]["t_minus1_abs"],
        "raw_|t+1|": reports["raw"]["t_plus1_abs"],
        "physical_|t-1|": reports["physical"]["t_minus1_abs"],
        "physical_|t+1|": reports["physical"]["t_plus1_abs"],
        "raw_rank": reports["raw"]["numerical_rank"],
        "physical_rank": reports["physical"]["numerical_rank"],
        "raw_cond": reports["raw"]["condition_number"],
        "physical_cond": reports["physical"]["condition_number"],
        "raw_residual_before": reports["raw"]["raw_norm_before"],
        "raw_residual_after": reports["raw"]["raw_norm_after"],
        "physical_residual_before": reports["physical"].get("physical_norm_before"),
        "physical_residual_after": reports["physical"].get("physical_norm_after"),
        "reeval_ok": reeval_ok,
    }), indent=2))


if __name__ == "__main__":
    main()
