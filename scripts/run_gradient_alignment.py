#!/usr/bin/env python3
"""Gradient-alignment audit.  Phases 0–8.  Diagnostic only.

No training objective is altered.  RCWA values stay outside every loss.
The script quantifies whether the existing PDE gradient points in a
direction that can increase the transmitted ±1 amplitudes.

Structure
---------
Phase 0  Safe preflight
Phase 1  Build feature-scaled Jacobian A_phys; compute SVD
Phase 2  Modal sensitivity of every right singular vector
Phase 3  Project each loss-component gradient onto singular subspaces
         at five checkpoints (init, ep1, ep100, ep500, ep1000)
Phase 4  Gradient conflict analysis (cosine similarities + directional
         derivatives of t_±1 along every loss-component gradient)
Phase 5  1000-epoch training-trajectory with projection logging
Phase 6  Decision (A/B/C/D/E)
Phase 7  Follow-up proposal JSON if warranted
Phase 8  Verification (compileall + pytest)
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
from src.geometry import epsilon_r as epsilon_r_fn
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
from scripts.train_lbg import layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    BLOCK_ORDER, CANONICAL_REF, COMPANION_PATH, HEAD_PARAM_COUNT,
    MODAL_ORDER_MAX, N_DTN_ORDERS, RCOND, SEED, TARGET_T1,
    assemble_system, confirm_validated_scales, freeze_except_head,
    head_parameter_records, head_parameters, load_points,
    model_config_hash, pack_head, pack_residual, residual_blocks,
    row_metadata, row_scales, unpack_head,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import (
    git_commit, git_status, make_fresh_model, parameter_ordering_hash,
    run_compileall, run_pytest,
)
from scripts.run_feature_scaling import (
    ScaledHead, apply_feature_scaling, compute_feature_scales,
    freeze_scaled_except_head, make_scaled_model, scaled_head_parameters,
    FEATURE_SCALE_FLOOR,
)


# ── Patch spatial_fourier_t so requires_grad propagates through ──────────────
def _sfq_fixed(model, physics, z_bot: float, n_quad: int = 512) -> dict[int, complex]:
    x    = torch.linspace(0.0, physics.period, n_quad + 1,
                          dtype=torch.float64)[:-1].requires_grad_(True)
    z    = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    e    = (er + 1j * ei).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    g0   = 2.0 * np.pi / physics.period
    return {m: complex(np.mean(e * np.exp(-1j * m * g0 * x_np)))
            for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1)}

_fhls.spatial_fourier_t = _sfq_fixed


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_COLLOCATION = (64, 32, 32, 32)
PHASE_VALIDITY_THRESHOLD = 1e-8
TARGET_T1 = 0.049410786
BASELINE_T1 = 0.00638    # Phase 6 seed-42 best


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = _vec_norm(a); nb = _vec_norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _proj_norm(v: np.ndarray, basis: np.ndarray) -> float:
    """||P_basis v||  where basis is (n, k) orthonormal columns."""
    if basis.shape[1] == 0:
        return 0.0
    coords = basis.T @ v          # (k,)
    return float(np.linalg.norm(coords))


def _feature_scale_hash(scales: dict) -> str:
    payload = json.dumps(
        {k: v["scales"] for k, v in scales.items()}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Modal observable Jacobian (head-parameter space)
# ─────────────────────────────────────────────────────────────────────────────

def build_modal_jacobian(
    model: ExplicitFourierModalDD,
    heads: list[torch.nn.Parameter],
    physics,
    z_bot: float,
    n_quad: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """J_t: (2*n_orders, HEAD_PARAM_COUNT) Jacobian of Re/Im t_m w.r.t. head params.

    Returns (J_t, t0_complex, orders_list).
    Uses the spatial-Fourier extraction at z_bot (no RCWA).
    """
    orders = list(range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1))
    n_m    = len(orders)
    G0     = 2.0 * np.pi / physics.period

    def _eval_t(theta_vec: np.ndarray) -> np.ndarray:
        unpack_head(heads, theta_vec)
        x  = torch.linspace(0.0, physics.period, n_quad + 1,
                             dtype=torch.float64)[:-1].requires_grad_(True)
        z  = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
        er, ei, *_ = model.net_sub.field_components(x, z)
        E  = (er + 1j * ei).detach().numpy()
        xn = x.detach().numpy()
        return np.array([complex(np.mean(E * np.exp(-1j * m * G0 * xn)))
                          for m in orders])

    t0    = _eval_t(np.zeros(HEAD_PARAM_COUNT))
    J_t   = np.zeros((2 * n_m, HEAD_PARAM_COUNT), dtype=np.float64)
    e_j   = np.zeros(HEAD_PARAM_COUNT)
    for j in range(HEAD_PARAM_COUNT):
        e_j[j] = 1.0
        tj     = _eval_t(e_j)
        delta  = tj - t0
        J_t[0::2, j] = delta.real
        J_t[1::2, j] = delta.imag
        e_j[j] = 0.0
        if (j + 1) % 200 == 0:
            print(f"    modal Jacobian {j+1}/{HEAD_PARAM_COUNT}", flush=True)

    unpack_head(heads, np.zeros(HEAD_PARAM_COUNT))
    return J_t, t0, orders


# ─────────────────────────────────────────────────────────────────────────────
# Loss-component gradients w.r.t. head parameters
# ─────────────────────────────────────────────────────────────────────────────

LOSS_COMPONENTS = (
    "pde_air", "pde_grat", "pde_sub",
    "vertical_E", "vertical_H",
    "horizontal_E", "horizontal_H",
    "top_DtN", "bottom_DtN",
)

def _loss_component_gradients(
    model, pts, physics, coeff, heads_flat: list[torch.nn.Parameter],
) -> dict[str, np.ndarray]:
    """Gradient of each loss component w.r.t. head parameters (frozen hidden).

    Returns dict: component → (HEAD_PARAM_COUNT,) float64 array.
    Also returns 'total' gradient.
    """
    from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss
    from src.maxwell_layered_bg import (
        lbg_bottom_bc, lbg_top_bc, lbg_vertical_interface_loss,
    )

    zero  = torch.zeros(1, dtype=torch.float64)
    p     = physics

    def _pde(key, net, eps_val):
        xk = pts.get(f"x_{key}"); zk = pts.get(f"z_{key}")
        if xk is None or len(xk) == 0:
            return zero
        if key == "grat":
            eps_val = epsilon_r_fn(xk, zk, physics)
        res = maxwell_2d_lbg_pde_residual(net, xk, zk, physics, eps_val, coeff)
        return sum(torch.mean(r**2) for r in res) / len(res)

    def _grad(loss_tensor: torch.Tensor) -> np.ndarray:
        for param in heads_flat:
            if param.grad is not None:
                param.grad.zero_()
        loss_tensor.backward(retain_graph=True)
        parts = [param.grad.detach().reshape(-1).clone()
                 if param.grad is not None
                 else torch.zeros_like(param.reshape(-1))
                 for param in heads_flat]
        return torch.cat(parts).numpy()

    # Build each component loss
    La  = _pde("air",  model.net_air,  p.n_air**2)
    Lg  = _pde("grat", model.net_grat, p.n_ridge**2)
    Ls  = _pde("sub",  model.net_sub,  p.n_substrate**2)

    LEv_l, LHv_l = lbg_vertical_interface_loss(
        model.net_grat, p.ridge_x_min, pts["z_vleft"])
    LEv_r, LHv_r = lbg_vertical_interface_loss(
        model.net_grat, p.ridge_x_max, pts["z_vright"])
    LEv = LEv_l + LEv_r; LHv = LHv_l + LHv_r

    LE1, LH1 = maxwell_2d_nd_interface_loss(
        model.net_air, model.net_grat, p.ridge_z_min, pts["x_int1"])
    LE2, LH2 = maxwell_2d_nd_interface_loss(
        model.net_grat, model.net_sub,  p.ridge_z_max, pts["x_int2"])
    LEh = LE1 + LE2; LHh = LH1 + LH2

    Lt  = lbg_top_bc(
        model.net_air, pts["x_top"], physics, use_dtn=True, n_dtn_orders=N_DTN_ORDERS)
    Lb  = lbg_bottom_bc(
        model.net_sub, pts["x_bot"], physics, coeff, use_dtn=True, n_dtn_orders=N_DTN_ORDERS)

    components_map = {
        "pde_air":       La,
        "pde_grat":      Lg,
        "pde_sub":       Ls,
        "vertical_E":    LEv,
        "vertical_H":    LHv,
        "horizontal_E":  LEh,
        "horizontal_H":  LHh,
        "top_DtN":       Lt,
        "bottom_DtN":    Lb,
    }

    grads = {}
    for name, loss in components_map.items():
        if loss.requires_grad:
            grads[name] = _grad(loss)
        else:
            grads[name] = np.zeros(HEAD_PARAM_COUNT)

    # Total gradient
    total = sum(v for v in components_map.values())
    grads["total"] = _grad(total)

    # Zero gradients after use
    for param in heads_flat:
        if param.grad is not None:
            param.grad.zero_()

    return grads


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — SVD of Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def phase1_svd(
    jacobian_s: np.ndarray,
    r0_s: np.ndarray,
    scales_row: np.ndarray,
) -> dict:
    print("\n[Phase 1] SVD of feature-scaled Jacobian", flush=True)

    A_phys   = jacobian_s / scales_row[:, None]
    b_phys   = r0_s       / scales_row
    col_norm = np.linalg.norm(A_phys, axis=0)

    U, sigma, Vt = np.linalg.svd(A_phys, full_matrices=True)
    rank_rcond   = int(np.sum(sigma > RCOND * sigma[0]))
    cond         = float(sigma[0] / sigma[rank_rcond - 1]) if rank_rcond else float("inf")

    V_active = Vt[:rank_rcond, :].T          # (1386, rank)
    V_null   = Vt[rank_rcond:, :].T          # (1386, n_null)

    diagnostic_ranks = {
        str(tol): int(np.sum(sigma > float(tol) * sigma[0]))
        for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
    }

    print(f"  rank={rank_rcond}  cond={cond:.3e}  "
          f"n_null={1386 - rank_rcond}  col_norm=[{col_norm.min():.3e},{col_norm.max():.3e}]",
          flush=True)
    print(f"  diagnostic_ranks: {diagnostic_ranks}", flush=True)

    return {
        "n_rows":            int(A_phys.shape[0]),
        "n_cols":            int(A_phys.shape[1]),
        "rank":              rank_rcond,
        "n_null":            int(1386 - rank_rcond),
        "condition_number":  cond,
        "rcond":             RCOND,
        "diagnostic_ranks":  diagnostic_ranks,
        "sigma":             sigma,
        "U":                 U,
        "Vt":                Vt,
        "V_active":          V_active,
        "V_null":            V_null,
        "col_norm":          col_norm,
        "A_phys":            A_phys,
        "b_phys":            b_phys,
        "sigma_max":         float(sigma[0]),
        "sigma_at_rank":     float(sigma[rank_rcond - 1]) if rank_rcond else 0.0,
        "sigma_first_null":  float(sigma[rank_rcond]) if rank_rcond < len(sigma) else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Modal sensitivity of singular directions
# ─────────────────────────────────────────────────────────────────────────────

def phase2_modal_sensitivity(
    J_t: np.ndarray,
    t0: np.ndarray,
    orders: list[int],
    svd: dict,
) -> dict:
    print("\n[Phase 2] Modal sensitivity of singular directions", flush=True)

    Vt      = svd["Vt"]           # (1386, 1386)
    sigma   = svd["sigma"]
    rank    = svd["rank"]

    idx_m1  = orders.index(-1)
    idx_p1  = orders.index(+1)
    idx_0   = orders.index(0)

    # Real/imag row indices for each mode in J_t  (shape 2*n_modes x 1386)
    ri_m1   = 2 * idx_m1;  ii_m1 = ri_m1 + 1
    ri_p1   = 2 * idx_p1;  ii_p1 = ri_p1 + 1
    ri_0    = 2 * idx_0;   ii_0  = ri_0  + 1

    J_t_m1  = J_t[[ri_m1, ii_m1], :]   # (2, 1386)
    J_t_p1  = J_t[[ri_p1, ii_p1], :]
    J_t_0   = J_t[[ri_0,  ii_0 ], :]

    rows = []
    for i in range(min(len(sigma), 1386)):
        vi   = Vt[i, :]                        # i-th right singular vector
        s_i  = float(sigma[i]) if i < len(sigma) else 0.0

        dt_m1 = J_t_m1 @ vi                    # (2,)
        dt_p1 = J_t_p1 @ vi
        dt_0  = J_t_0  @ vi

        s_pm1 = float(np.sqrt(np.sum(dt_m1**2) + np.sum(dt_p1**2)))
        s_m1  = float(np.linalg.norm(dt_m1))
        s_p1  = float(np.linalg.norm(dt_p1))
        s_0   = float(np.linalg.norm(dt_0))

        rows.append({
            "singular_index":       i,
            "singular_value":       s_i,
            "active":               bool(i < rank),
            "t_minus1_sensitivity": s_m1,
            "t_plus1_sensitivity":  s_p1,
            "combined_pm1_sensitivity": s_pm1,
            "t0_sensitivity":       s_0,
            "vector_norm":          float(np.linalg.norm(vi)),
        })

    # Sort by combined ±1 sensitivity descending
    rows_sorted = sorted(rows, key=lambda r: -r["combined_pm1_sensitivity"])
    print(f"  Top 10 singular directions by ±1 sensitivity:", flush=True)
    for r in rows_sorted[:10]:
        label = "active" if r["active"] else "null"
        print(f"    i={r['singular_index']:4d}  σ={r['singular_value']:.3e}  "
              f"[{label}]  s_±1={r['combined_pm1_sensitivity']:.4e}  "
              f"s_0={r['t0_sensitivity']:.4e}", flush=True)

    # Build ±1-sensitive subspace: top singular directions of J_pm1 = [J_t_m1; J_t_p1]
    J_pm1    = np.vstack([J_t_m1, J_t_p1])   # (4, 1386)
    _, sigma_pm1_vals, Vt_pm1 = np.linalg.svd(J_pm1, full_matrices=False)
    # Retain directions with σ > 0.01 * max
    rank_pm1 = int(np.sum(sigma_pm1_vals > 0.01 * sigma_pm1_vals[0]))
    rank_pm1 = max(rank_pm1, 1)
    V_pm1    = Vt_pm1[:rank_pm1, :].T        # (1386, rank_pm1)

    # Summary statistics
    n_active_with_pm1 = sum(
        1 for r in rows if r["active"] and r["combined_pm1_sensitivity"] > 1e-12)
    n_null_with_pm1   = sum(
        1 for r in rows if not r["active"] and r["combined_pm1_sensitivity"] > 1e-12)
    max_active_pm1    = max(
        (r["combined_pm1_sensitivity"] for r in rows if r["active"]), default=0.0)
    max_null_pm1      = max(
        (r["combined_pm1_sensitivity"] for r in rows if not r["active"]), default=0.0)

    print(f"  Active dirs with ±1 sensitivity > 1e-12: {n_active_with_pm1}/{rank}",
          flush=True)
    print(f"  Null   dirs with ±1 sensitivity > 1e-12: {n_null_with_pm1}/{1386-rank}",
          flush=True)
    print(f"  Max active ±1 sensitivity: {max_active_pm1:.4e}", flush=True)
    print(f"  ±1-sensitive subspace rank (via J_pm1 SVD): {rank_pm1}", flush=True)
    print(f"  σ_pm1_vals: {sigma_pm1_vals}", flush=True)

    return {
        "rows":                     rows,
        "rows_sorted_by_pm1":       rows_sorted,
        "n_active_with_pm1":        n_active_with_pm1,
        "n_null_with_pm1":          n_null_with_pm1,
        "max_active_pm1_sensitivity": max_active_pm1,
        "max_null_pm1_sensitivity": max_null_pm1,
        "pm1_subspace_rank":        rank_pm1,
        "pm1_sigma_vals":           sigma_pm1_vals.tolist(),
        "V_pm1":                    V_pm1,
        "t0_at_zero_head": {m: {"re": float(t0[i].real), "im": float(t0[i].imag)}
                             for i, m in enumerate(orders)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — PDE gradient projection
# ─────────────────────────────────────────────────────────────────────────────

def phase3_gradient_projection(
    model, heads_s, pts, physics, coeff, svd: dict, p2: dict,
    checkpoints: list[dict],
) -> list[dict]:
    """Compute gradient projections at each checkpoint."""
    print("\n[Phase 3] PDE gradient projection", flush=True)

    V_active = svd["V_active"]
    V_null   = svd["V_null"]
    V_pm1    = p2["V_pm1"]
    rank     = svd["rank"]

    all_rows = []
    for ckpt in checkpoints:
        ep    = ckpt["epoch"]
        theta = ckpt["theta"]                  # (HEAD_PARAM_COUNT,)
        print(f"  checkpoint epoch={ep}", flush=True)

        unpack_head(heads_s, theta)
        grads = _loss_component_gradients(model, pts, physics, coeff, heads_s)
        unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

        row = {"epoch": ep}
        for name, g in grads.items():
            norm_g     = _vec_norm(g)
            norm_act   = _proj_norm(g, V_active)
            norm_pm1   = _proj_norm(g, V_pm1)
            norm_null  = _proj_norm(g, V_null)
            frac_pm1   = norm_pm1 / (norm_g + 1e-30)
            frac_act   = norm_act / (norm_g + 1e-30)
            frac_null  = norm_null / (norm_g + 1e-30)
            row[f"{name}_norm"]       = norm_g
            row[f"{name}_P_active"]   = norm_act
            row[f"{name}_P_pm1"]      = norm_pm1
            row[f"{name}_P_null"]     = norm_null
            row[f"{name}_frac_pm1"]   = frac_pm1
            row[f"{name}_frac_active"] = frac_act
            row[f"{name}_frac_null"]  = frac_null

        all_rows.append(row)

        # Print summary for key components
        for c in ("pde_grat", "total"):
            if c in grads:
                print(f"    {c}: ||g||={row[c+'_norm']:.3e}  "
                      f"||P_active||/||g||={row[c+'_frac_active']:.3f}  "
                      f"||P_pm1||/||g||={row[c+'_frac_pm1']:.3f}  "
                      f"||P_null||/||g||={row[c+'_frac_null']:.3f}", flush=True)

    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Gradient conflict analysis
# ─────────────────────────────────────────────────────────────────────────────

def phase4_gradient_conflict(
    model, heads_s, pts, physics, coeff,
    J_t: np.ndarray, orders: list[int],
) -> dict:
    """Cosine similarities and directional modal derivatives at initialization."""
    print("\n[Phase 4] Gradient conflict analysis", flush=True)

    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))
    grads = _loss_component_gradients(model, pts, physics, coeff, heads_s)
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    idx_m1 = orders.index(-1); idx_p1 = orders.index(+1)
    ri_m1  = 2 * idx_m1; ii_m1 = ri_m1 + 1
    ri_p1  = 2 * idx_p1; ii_p1 = ri_p1 + 1

    J_m1   = J_t[[ri_m1, ii_m1], :]     # (2, 1386)
    J_p1   = J_t[[ri_p1, ii_p1], :]

    def _dt(g, J_row):
        """Directional derivative of complex t_m along gradient g (normalised)."""
        g_hat = g / (_vec_norm(g) + 1e-30)
        return float(np.linalg.norm(J_row @ g_hat))

    # Cosine similarities between all component pairs
    names  = [c for c in LOSS_COMPONENTS if c in grads] + ["total"]
    cosines = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j > i:
                c = _cos(grads[ni], grads[nj])
                cosines[f"{ni}_vs_{nj}"] = c

    # Directional derivatives of t_±1 along each component gradient
    dir_derivs = {}
    for name in names:
        g = grads[name]
        dir_derivs[name] = {
            "d_t_minus1": _dt(g, J_m1),
            "d_t_plus1":  _dt(g, J_p1),
            "d_t_pm1_combined": float(
                np.sqrt(_dt(g, J_m1)**2 + _dt(g, J_p1)**2)),
            "gradient_norm": _vec_norm(g),
        }

    print("  Directional t_±1 derivatives along each gradient:", flush=True)
    for name, v in dir_derivs.items():
        print(f"    {name:20s}  ||g||={v['gradient_norm']:.3e}  "
              f"d|t-1|={v['d_t_minus1']:.4e}  "
              f"d|t+1|={v['d_t_plus1']:.4e}  "
              f"combined={v['d_t_pm1_combined']:.4e}", flush=True)

    # Cosine between ridge PDE and ±1-sensitive subspace
    print("\n  Key cosines:", flush=True)
    key_pairs = [
        ("pde_grat", "total"), ("pde_grat", "top_DtN"),
        ("pde_grat", "bottom_DtN"), ("pde_grat", "horizontal_E"),
        ("pde_air", "total"), ("horizontal_E", "total"),
    ]
    for a, b in key_pairs:
        if a in grads and b in grads:
            c = _cos(grads[a], grads[b])
            print(f"    cos({a}, {b}) = {c:.4f}", flush=True)

    return {
        "cosines":           cosines,
        "directional_derivs": dir_derivs,
        "gradient_norms":    {n: _vec_norm(grads[n]) for n in names if n in grads},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Training-trajectory logging
# ─────────────────────────────────────────────────────────────────────────────

def phase5_trajectory(
    physics, pts, coeff, device, dtype, ref, ref_path, companion_p,
    svd: dict, p2: dict, J_t: np.ndarray, orders: list[int],
    out_dir: Path, epochs: int = 1000, log_every: int = 100,
) -> list[dict]:
    print("\n[Phase 5] Training trajectory (1000 epochs)", flush=True)

    from scripts.run_frozen_head_least_squares import modal_report

    V_active = svd["V_active"]
    V_pm1    = p2["V_pm1"]
    V_null   = svd["V_null"]

    idx_m1 = orders.index(-1); idx_p1 = orders.index(+1)
    ri_m1  = 2 * idx_m1;       ii_m1  = ri_m1 + 1
    ri_p1  = 2 * idx_p1;       ii_p1  = ri_p1 + 1
    J_m1   = J_t[[ri_m1, ii_m1], :]
    J_p1   = J_t[[ri_p1, ii_p1], :]

    set_seed(SEED)
    model_s, _sc = make_scaled_model(physics, pts)
    # Full model, all params trainable
    for p in model_s.parameters():
        p.requires_grad_(True)

    heads_s = scaled_head_parameters(model_s)
    opt     = torch.optim.Adam(model_s.parameters(), lr=5e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    traj = []

    for ep in range(1, epochs + 1):
        model_s.train()
        opt.zero_grad(set_to_none=True)

        losses = layered_bg_loss(
            model_s, pts, physics, coeff,
            w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
            use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
            rcwa_amps=None, w_modal=0.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % log_every == 0 or ep == 1:
            # Gradient projection (head params only)
            head_grad = torch.cat([
                p.grad.detach().reshape(-1) if p.grad is not None
                else torch.zeros_like(p.reshape(-1))
                for p in heads_s
            ]).numpy()

            norm_hg    = _vec_norm(head_grad)
            proj_act   = _proj_norm(head_grad, V_active)
            proj_pm1   = _proj_norm(head_grad, V_pm1)
            proj_null  = _proj_norm(head_grad, V_null)

            # Directional derivative of |t_±1| along head gradient
            g_hat = head_grad / (norm_hg + 1e-30)
            d_m1  = float(np.linalg.norm(J_m1 @ g_hat))
            d_p1  = float(np.linalg.norm(J_p1 @ g_hat))

            head_norm = float(pack_head(heads_s).norm())

            # Modal
            modal_m = modal_report(model_s, physics, coeff, device, dtype,
                                   ref, ref_path, companion_p)
            t1   = modal_m["t_minus1_abs"]
            tp   = modal_m["t_plus1_abs"]
            RT   = modal_m["R_plus_T"]
            L2   = modal_m["total_complex_l2"]
            scat = modal_m["scattered_complex_l2"]

            # Hidden feature norms
            h_norms = {}
            for net_name, net, zkey in (
                ("net_air",  model_s.net_air,  "z_air"),
                ("net_grat", model_s.net_grat, "z_grat"),
                ("net_sub",  model_s.net_sub,  "z_sub"),
            ):
                z = pts[zkey]
                sh = net.coefficient_mlp
                with torch.no_grad():
                    zn = (2*(z - net.z_lo)/(net.z_hi - net.z_lo) - 1)[:, None]
                    h  = sh.tanh(sh.lin0(zn))
                h_norms[net_name] = float(h.norm())

            row = {
                "epoch":           ep,
                "pde_air":         float(losses["pde_air"].detach()),
                "pde_grat":        float(losses["pde_grat"].detach()),
                "pde_sub":         float(losses["pde_sub"].detach()),
                "top_DtN":         float(losses["top"].detach()),
                "bottom_DtN":      float(losses["bottom"].detach()),
                "E_int":           float((losses["E_int1"]+losses["E_int2"]).detach()),
                "H_int":           float((losses["H_int1"]+losses["H_int2"]).detach()),
                "total_loss":      float(losses["total"].detach()),
                "head_norm":       head_norm,
                "hgrad_norm":      norm_hg,
                "hgrad_P_active":  proj_act,
                "hgrad_P_pm1":     proj_pm1,
                "hgrad_P_null":    proj_null,
                "hgrad_frac_pm1":  proj_pm1 / (norm_hg + 1e-30),
                "d_t_minus1_along_grad": d_m1,
                "d_t_plus1_along_grad":  d_p1,
                "t_minus1": t1, "t_plus1": tp,
                "R_plus_T": RT, "total_complex_l2": L2,
                "scattered_complex_l2": scat,
                **{f"h_norm_{k}": v for k, v in h_norms.items()},
            }
            traj.append(row)
            print(f"  ep={ep:4d}  pde_grat={row['pde_grat']:.2e}  "
                  f"|t-1|={t1:.4f}  R+T={RT:.4f}  "
                  f"hgrad_frac_pm1={row['hgrad_frac_pm1']:.4f}  "
                  f"d_t-1={d_m1:.3e}  |head|={head_norm:.3e}", flush=True)

    # Save CSV
    if traj:
        csv_path = out_dir / "trajectory" / "training_trajectory.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader(); writer.writerows(traj)

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Decision
# ─────────────────────────────────────────────────────────────────────────────

def phase6_decision(p2: dict, p3_rows: list[dict], p4: dict,
                    traj: list[dict]) -> dict:
    print("\n[Phase 6] Decision", flush=True)

    # Key metrics
    max_active_pm1   = p2["max_active_pm1_sensitivity"]
    pm1_rank         = p2["pm1_subspace_rank"]

    # At initialization (first checkpoint = epoch 0 or closest)
    init_row = p3_rows[0] if p3_rows else {}
    total_frac_pm1   = init_row.get("total_frac_pm1", 0.0)
    ridge_frac_pm1   = init_row.get("pde_grat_frac_pm1", 0.0)
    d_t1_total       = p4["directional_derivs"].get("total", {}).get("d_t_pm1_combined", 0.0)
    d_t1_ridge       = p4["directional_derivs"].get("pde_grat", {}).get("d_t_pm1_combined", 0.0)

    # Trajectory: does |t_±1| increase?
    t1_start         = traj[0]["t_minus1"] if traj else BASELINE_T1
    t1_end           = traj[-1]["t_minus1"] if traj else BASELINE_T1
    t1_increasing    = t1_end > 1.05 * t1_start

    # avg gradient projection fraction over trajectory
    avg_frac_pm1 = float(np.mean([r["hgrad_frac_pm1"] for r in traj])) if traj else 0.0

    print(f"  max_active_pm1_sensitivity:  {max_active_pm1:.4e}", flush=True)
    print(f"  pm1_subspace_rank:           {pm1_rank}", flush=True)
    print(f"  total_gradient_frac_pm1:     {total_frac_pm1:.4f}", flush=True)
    print(f"  ridge_gradient_frac_pm1:     {ridge_frac_pm1:.4f}", flush=True)
    print(f"  d_t±1_along_total_gradient:  {d_t1_total:.4e}", flush=True)
    print(f"  d_t±1_along_ridge_gradient:  {d_t1_ridge:.4e}", flush=True)
    print(f"  avg_hgrad_frac_pm1 (traj):   {avg_frac_pm1:.4f}", flush=True)
    print(f"  t1_start={t1_start:.5f}  t1_end={t1_end:.5f}  "
          f"increasing={t1_increasing}", flush=True)

    FRAC_THRESHOLD   = 0.05    # < 5% of gradient is in ±1 direction → "negligible"

    if max_active_pm1 < 1e-12:
        case = "D"
        diagnosis = (
            "The ±1-sensitive subspace has negligible output sensitivity. "
            "The modal output J_t has near-zero projection onto the active "
            "singular vectors. Revisit modal output basis and DFT extraction path."
        )
    elif total_frac_pm1 < FRAC_THRESHOLD and d_t1_total < 1e-8:
        case = "A"
        diagnosis = (
            f"Total PDE gradient has negligible ±1-sensitive projection "
            f"({total_frac_pm1:.4f} < {FRAC_THRESHOLD} threshold). "
            f"Directional derivative of t_±1 along total gradient is {d_t1_total:.3e}. "
            "The current loss is misaligned with the desired scattering direction. "
            "Propose a preconditioned-gradient experiment."
        )
    elif d_t1_ridge < d_t1_total * 0.1 and not t1_increasing:
        case = "C"
        diagnosis = (
            f"Ridge PDE gradient directional derivative for t_±1 "
            f"({d_t1_ridge:.3e}) is much smaller than total ({d_t1_total:.3e}). "
            "Ridge-source-cancellation gradient conflicts with transmitted-order "
            "formation. The ridge PDE dominates the total gradient without "
            "contributing to ±1."
        )
    elif total_frac_pm1 >= FRAC_THRESHOLD and not t1_increasing:
        case = "B"
        diagnosis = (
            f"Gradient has substantial ±1-sensitive projection "
            f"({total_frac_pm1:.4f} >= {FRAC_THRESHOLD}), but |t_±1| does not "
            f"increase (start={t1_start:.5f}, end={t1_end:.5f}). "
            "Inspect curvature, step size, or gradient cancellation between "
            "competing loss blocks."
        )
    elif avg_frac_pm1 > FRAC_THRESHOLD and t1_increasing:
        case = "E"
        diagnosis = (
            f"Gradient projection onto ±1 is nonzero (avg={avg_frac_pm1:.4f}) "
            f"and directional derivative is positive ({d_t1_total:.3e}), "
            f"but |t_±1| increase is modest "
            f"(from {t1_start:.5f} to {t1_end:.5f}). "
            "Inspect optimizer state, parameter scaling, and competing losses."
        )
    else:
        case = "A"
        diagnosis = (
            f"Gradient ±1 projection fraction is low ({total_frac_pm1:.4f}) "
            f"and t_±1 does not increase. "
            "Current loss is misaligned with required scattering direction."
        )

    print(f"\n  CASE {case}: {diagnosis[:150]}...", flush=True)
    return {
        "case":                     case,
        "diagnosis":                diagnosis,
        "max_active_pm1_sensitivity": max_active_pm1,
        "pm1_subspace_rank":        pm1_rank,
        "total_gradient_frac_pm1":  total_frac_pm1,
        "ridge_gradient_frac_pm1":  ridge_frac_pm1,
        "d_t_pm1_along_total_grad": d_t1_total,
        "d_t_pm1_along_ridge_grad": d_t1_ridge,
        "avg_hgrad_frac_pm1":       avg_frac_pm1,
        "t1_start":                 t1_start,
        "t1_end":                   t1_end,
        "t1_increasing":            t1_increasing,
        "modal_loss_disabled":      True,
        "optical_coupling_disabled": True,
        "start_10000_epoch_run":    False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Follow-up proposal
# ─────────────────────────────────────────────────────────────────────────────

def phase7_proposal(decision: dict, p2: dict, p4: dict) -> dict | None:
    case = decision["case"]
    if case not in ("A", "B", "C", "E"):
        return None

    proposals = {
        "A": {
            "method": "PDE-Jacobian preconditioned gradient",
            "rationale": (
                "The PDE gradient has negligible projection onto the ±1-sensitive "
                "subspace. A preconditioned update that projects the gradient onto "
                "the ±1-active singular directions before the optimizer step would "
                "steer learning toward the target modes without changing the objective."
            ),
            "algorithm": (
                "Compute the projection P_pm1 @ g (where P_pm1 is the projector onto "
                "the ±1-sensitive active subspace), then take a gradient step with "
                "g_precond = P_pm1 @ g + alpha * P_null @ g where alpha < 1 "
                "downweights null-space components."
            ),
            "constraints": [
                "PDE-only: no RCWA in objective",
                "No modal-data loss",
                "No optical coupling",
                "Recompute P_pm1 from J_t every N epochs (frozen RCWA not used)",
            ],
        },
        "B": {
            "method": "Loss-block gradient scaling / reweighting",
            "rationale": (
                "Gradient has ±1 projection but t_±1 does not increase. "
                "Competing blocks may cancel the ±1-relevant update. "
                "Diagnose per-component directional derivatives and rescale."
            ),
            "constraints": ["No RCWA in objective", "No modal-data loss"],
        },
        "C": {
            "method": "Ridge-to-boundary coupling investigation",
            "rationale": (
                "Ridge PDE gradient conflicts with ±1 formation. "
                "The ridge source term drives learning away from boundary-observable "
                "modes. Investigate whether the DtN boundary provides a coupling "
                "channel from ridge to ±1."
            ),
            "constraints": ["No RCWA in objective", "No modal-data loss"],
        },
        "E": {
            "method": "Continuation with monitoring",
            "rationale": (
                "Directional derivative is positive but progress is slow. "
                "Controlled extension beyond 1000 epochs with monitoring."
            ),
            "constraints": ["No modal-data loss", "No optical coupling"],
        },
    }

    proposal = {
        "authorized_by_this_task": False,
        "case": case,
        "proposed_experiment": proposals.get(case, {}),
        "diagnostic_evidence": {
            "max_active_pm1_sensitivity": decision["max_active_pm1_sensitivity"],
            "total_gradient_frac_pm1":    decision["total_gradient_frac_pm1"],
            "d_t_pm1_along_total_grad":   decision["d_t_pm1_along_total_grad"],
            "pm1_subspace_rank":          decision["pm1_subspace_rank"],
        },
        "note": (
            "This proposal is a design document only.  No experiment starts "
            "automatically.  Authorization required before any implementation."
        ),
    }
    return proposal


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def write_plots(root: Path, svd: dict, p2: dict, p4: dict, traj: list[dict]) -> None:
    sigma = svd["sigma"]

    # Singular-value spectrum + ±1 sensitivity overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(np.arange(1, len(sigma) + 1),
                     np.maximum(sigma, np.finfo(float).tiny), lw=0.8)
    axes[0].axvline(svd["rank"], color="r", ls="--", lw=0.8, label=f"rank={svd['rank']}")
    axes[0].set(xlabel="index", ylabel="σ", title="Singular value spectrum")
    axes[0].legend(fontsize=8)

    rows = p2["rows"]
    pm1_vals = [r["combined_pm1_sensitivity"] for r in rows]
    axes[1].semilogy(np.arange(len(pm1_vals)), pm1_vals, lw=0.8, label="±1 sensitivity")
    axes[1].axvline(svd["rank"], color="r", ls="--", lw=0.8, label=f"rank={svd['rank']}")
    axes[1].set(xlabel="singular direction index", ylabel="s_±1",
                title="±1 sensitivity per singular direction")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "singular_direction_sensitivity.png", dpi=160)
    plt.close(fig)

    # Gradient projection bar chart at initialization
    if traj:
        names   = [c for c in LOSS_COMPONENTS] + ["total"]
        fracs   = [traj[0].get(f"{c}_frac_pm1", 0.0) for c in names]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x       = np.arange(len(names))
        ax.bar(x, fracs)
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="||P_pm1 g|| / ||g||",
               title="Gradient fraction in ±1-sensitive subspace (epoch=1)")
        fig.tight_layout()
        fig.savefig(root / "gradient_projection.png", dpi=160)
        plt.close(fig)

    # Cosine heatmap
    cos_dict  = p4["cosines"]
    all_names = [c for c in LOSS_COMPONENTS if c in p4["gradient_norms"]] + ["total"]
    n         = len(all_names)
    C         = np.zeros((n, n))
    for i, ni in enumerate(all_names):
        C[i, i] = 1.0
        for j, nj in enumerate(all_names):
            key1 = f"{ni}_vs_{nj}"; key2 = f"{nj}_vs_{ni}"
            if key1 in cos_dict:
                C[i, j] = C[j, i] = cos_dict[key1]
    fig, ax   = plt.subplots(figsize=(8, 6.5))
    im        = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(n), all_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n), all_names, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Gradient cosine similarities (initialization)")
    fig.tight_layout()
    fig.savefig(root / "gradient_cosines.png", dpi=160)
    plt.close(fig)

    # Directional modal derivatives
    dd        = p4["directional_derivs"]
    d_names   = list(dd.keys())
    d_m1      = [dd[n]["d_t_minus1"] for n in d_names]
    d_p1      = [dd[n]["d_t_plus1"]  for n in d_names]
    fig, ax   = plt.subplots(figsize=(10, 4.5))
    x         = np.arange(len(d_names))
    ax.bar(x - 0.2, d_m1, 0.4, label="|dt_{-1}|")
    ax.bar(x + 0.2, d_p1, 0.4, label="|dt_{+1}|")
    ax.set_xticks(x, d_names, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="directional derivative", title="Directional t_±1 derivatives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "directional_modal_derivatives.png", dpi=160)
    plt.close(fig)

    # Training trajectory
    if traj:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        ep        = [r["epoch"] for r in traj]
        axes[0].plot(ep, [r["t_minus1"] for r in traj], label="|t_{-1}|")
        axes[0].plot(ep, [r["t_plus1"]  for r in traj], label="|t_{+1}|")
        axes[0].axhline(TARGET_T1, color="k", ls="--", label="target")
        axes[0].set(xlabel="epoch", ylabel="|t_m|", title="Modal amplitudes")
        axes[0].legend(fontsize=8)

        axes[1].semilogy(ep, [r["pde_grat"] for r in traj], label="ridge PDE")
        axes[1].semilogy(ep, [r["total_loss"] for r in traj], label="total")
        axes[1].set(xlabel="epoch", ylabel="loss", title="Loss components")
        axes[1].legend(fontsize=8)

        axes[2].plot(ep, [r["hgrad_frac_pm1"] for r in traj], label="frac_pm1")
        axes[2].plot(ep, [r["hgrad_frac_active"] if "hgrad_frac_active" in r
                          else r["hgrad_P_active"] / (r["hgrad_norm"] + 1e-30)
                          for r in traj], label="frac_active", ls="--")
        axes[2].set(xlabel="epoch", ylabel="fraction",
                    title="Gradient projection fractions")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "trajectory" / "training_trajectory.png", dpi=160)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Gradient-alignment audit")
    ap.add_argument("--output-root",   default="outputs/phase5_gradient_alignment")
    ap.add_argument("--shared-points", default="outputs/phase5_feature_scaling/shared_points.npz")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    (root / "trajectory").mkdir()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg         = load_config(ROOT / "configs/default.yaml")
    physics     = make_lambda_0p8(cfg.physics)
    ref_path    = ROOT / CANONICAL_REF
    companion_p = ROOT / COMPANION_PATH
    validate_reference(ref_path, physics)
    ref         = normalize_reference_orientation(load_reference_npz(ref_path))
    device      = torch.device("cpu")
    dtype       = torch.float64
    coeff       = compute_background_coefficients(physics)
    pts         = load_points(physics, device, dtype, ROOT / args.shared_points)

    # ── Phase 0 record ────────────────────────────────────────────────────────
    can_sha = sha256_file(ref_path)
    com_sha = sha256_file(companion_p)
    man     = json.loads((ROOT / "outputs/reference_companion/manifest.json").read_text())

    # Feature scales (from prior audit)
    fs_rep  = json.loads(
        (ROOT / "outputs/phase5_feature_scaling/feature_scale_report.json").read_text())
    fs_scales = fs_rep["phase1_feature_scales"]
    fs_hash   = hashlib.sha256(
        json.dumps({k: v["scales"] for k, v in fs_scales.items()},
                   sort_keys=True).encode()
    ).hexdigest()

    phase0_record = {
        "git_commit":             git_commit(),
        "working_tree_status":    git_status(),
        "canonical_sha256":       can_sha,
        "canonical_sha_ok":       can_sha == man["canonical_sha256"],
        "companion_sha256":       com_sha,
        "companion_sha_ok":       com_sha == man["companion_sha256"],
        "model_config_hash":      model_config_hash(physics),
        "parameter_order_hash":   parameter_ordering_hash(
            head_parameter_records(make_fresh_model(physics))),
        "feature_scale_hash":     fs_hash,
        "seed":                   SEED,
        "head_param_count":       HEAD_PARAM_COUNT,
        "modal_loss_weight":      0.0,
        "optical_coupling":       False,
    }
    (root / "phase0_preflight.json").write_text(
        json.dumps(phase0_record, indent=2) + "\n")
    print("[Phase 0] Preflight recorded", flush=True)
    for k in ("canonical_sha_ok", "companion_sha_ok", "git_commit",
              "model_config_hash", "feature_scale_hash"):
        print(f"  {k}: {phase0_record[k]}", flush=True)

    # ── Build feature-scaled model + Jacobian ────────────────────────────────
    print("\n[Building scaled model + Jacobian]", flush=True)
    scales_dict, scale_meta = physical_scales(pts, physics, coeff)
    confirm_validated_scales(scales_dict, scale_meta)

    model_s, _ = make_scaled_model(physics, pts)
    heads_s     = freeze_scaled_except_head(model_s)
    jacobian_s, r0_s, lengths = assemble_system(
        model_s, heads_s, pts, physics, coeff)
    scales_row  = row_scales(lengths, scales_dict)

    # ── Phase 1 — SVD ────────────────────────────────────────────────────────
    svd = phase1_svd(jacobian_s, r0_s, scales_row)
    np.save(root / "singular_values.npy", svd["sigma"])

    # ── Build modal Jacobian ─────────────────────────────────────────────────
    print("\n[Building modal Jacobian]", flush=True)
    z_bot = 0.92 * physics.domain_height
    J_t, t0_arr, orders = build_modal_jacobian(model_s, heads_s, physics, z_bot)
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    p2 = phase2_modal_sensitivity(J_t, t0_arr, orders, svd)

    # ── Phase 3: gradient projection at checkpoints ──────────────────────────
    # Checkpoints: init (theta=0), ep1, ep100, ep500, ep1000 from prior smoke
    # We reconstruct epoch snapshots by running a mini training loop
    print("\n[Collecting gradient checkpoints]", flush=True)
    checkpoints = []
    for label, n_ep in (("init", 0), ("ep1", 1), ("ep100", 100),
                         ("ep500", 500), ("ep1000", 1000)):
        set_seed(SEED)
        m_ckpt, _ = make_scaled_model(physics, pts)
        for p in m_ckpt.parameters(): p.requires_grad_(True)
        heads_ckpt = scaled_head_parameters(m_ckpt)
        opt_c = torch.optim.Adam(m_ckpt.parameters(), lr=5e-4)
        for _ in range(n_ep):
            opt_c.zero_grad(set_to_none=True)
            L = layered_bg_loss(m_ckpt, pts, physics, coeff,
                                w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
                                use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
                                rcwa_amps=None, w_modal=0.0)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(m_ckpt.parameters(), 1.0)
            opt_c.step()
        theta_c = pack_head(heads_ckpt).detach().numpy()
        checkpoints.append({"epoch": n_ep, "theta": theta_c, "label": label})
        print(f"  checkpoint {label}: |theta|={float(np.linalg.norm(theta_c)):.4e}",
              flush=True)

    # Phase 3 uses a separate fresh model for gradient evaluation at each theta
    set_seed(SEED)
    model_g, _ = make_scaled_model(physics, pts)
    for p in model_g.parameters(): p.requires_grad_(True)
    heads_g = scaled_head_parameters(model_g)

    p3_rows = phase3_gradient_projection(
        model_g, heads_g, pts, physics, coeff, svd, p2, checkpoints)

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    p4 = phase4_gradient_conflict(model_g, heads_g, pts, physics, coeff, J_t, orders)

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    traj = phase5_trajectory(
        physics, pts, coeff, device, dtype, ref, ref_path, companion_p,
        svd, p2, J_t, orders, root,
    )

    # ── Phase 6 ──────────────────────────────────────────────────────────────
    decision = phase6_decision(p2, p3_rows, p4, traj)

    # ── Phase 7 ──────────────────────────────────────────────────────────────
    proposal = phase7_proposal(decision, p2, p4)

    # ── Phase 8: verification ────────────────────────────────────────────────
    print("\n[Phase 8] Verification", flush=True)
    compile_r = run_compileall()
    pytest_r  = run_pytest()
    print(f"  compileall: {'OK' if compile_r['passed'] else 'FAIL'}")
    print(f"  pytest: {pytest_r['summary_line']}")

    # ── Write all outputs ─────────────────────────────────────────────────────
    write_plots(root, svd, p2, p4, traj)

    # SVD metadata JSON
    svd_meta = {k: v for k, v in svd.items()
                if k not in ("sigma", "U", "Vt", "V_active", "V_null",
                              "A_phys", "b_phys", "col_norm")}
    (root / "svd_metadata.json").write_text(json.dumps(jsonable(svd_meta), indent=2) + "\n")

    # Modal sensitivity CSV
    with (root / "modal_singular_direction_sensitivity.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(p2["rows"][0]))
        writer.writeheader(); writer.writerows(p2["rows"])
    (root / "modal_singular_direction_sensitivity.json").write_text(
        json.dumps(jsonable({
            k: v for k, v in p2.items()
            if k not in ("V_pm1",)
        }), indent=2) + "\n")

    # Gradient projection CSV
    if p3_rows:
        with (root / "gradient_projection.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(p3_rows[0]))
            writer.writeheader(); writer.writerows(p3_rows)

    # Gradient conflict JSON
    (root / "gradient_conflict.json").write_text(
        json.dumps(jsonable(p4), indent=2) + "\n")

    # Summary
    summary = {
        **phase0_record,
        "svd": svd_meta,
        "phase2": {k: v for k, v in p2.items()
                   if k not in ("rows", "rows_sorted_by_pm1", "V_pm1")},
        "phase3_gradient_projection": p3_rows,
        "phase4_gradient_conflict": {
            "cosines": p4["cosines"],
            "directional_derivs": p4["directional_derivs"],
            "gradient_norms": p4["gradient_norms"],
        },
        "phase5_trajectory_summary": {
            "n_epochs":   len(traj) * 100 if traj else 0,
            "t1_start":   traj[0]["t_minus1"] if traj else None,
            "t1_end":     traj[-1]["t_minus1"] if traj else None,
            "avg_frac_pm1": float(np.mean([r["hgrad_frac_pm1"] for r in traj]))
                             if traj else None,
        },
        "decision": decision,
        "proposal": proposal,
        "compileall": compile_r,
        "pytest":     pytest_r,
        "phase5_passed": False,  # Phase 5 (validation gate) has not been reached
        "pm1_sensitive_gradient_present": (
            decision.get("total_gradient_frac_pm1", 0.0) >= 0.05
            or decision.get("d_t_pm1_along_total_grad", 0.0) > 1e-8
        ),
        "follow_up_preconditioned_experiment_authorized": False,
        "modal_data_loss_disabled": True,
        "optical_coupling_disabled": True,
    }
    (root / "gradient_alignment_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n")

    # Summary CSV
    rows_csv = [{"label": "gradient_alignment_audit",
                 "case": decision["case"],
                 "rank": svd["rank"],
                 "condition_number": svd["condition_number"],
                 "pm1_subspace_rank": p2["pm1_subspace_rank"],
                 "max_active_pm1_sensitivity": p2["max_active_pm1_sensitivity"],
                 "total_grad_frac_pm1": decision["total_gradient_frac_pm1"],
                 "ridge_grad_frac_pm1": decision["ridge_gradient_frac_pm1"],
                 "d_t_pm1_total": decision["d_t_pm1_along_total_grad"],
                 "t1_start": decision["t1_start"],
                 "t1_end": decision["t1_end"],
                 "t1_increasing": decision["t1_increasing"],
                 "avg_hgrad_frac_pm1": decision["avg_hgrad_frac_pm1"],
                 "pytest_passed": pytest_r["passed"],
                 "canonical_sha_ok": phase0_record["canonical_sha_ok"],
                 "companion_sha_ok": phase0_record["companion_sha_ok"],
                 }]
    with (root / "gradient_alignment_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        writer.writeheader(); writer.writerows(rows_csv)

    if proposal:
        (root / "follow_up_proposal.json").write_text(
            json.dumps(jsonable(proposal), indent=2) + "\n")

    print("\n" + "=" * 70, flush=True)
    print(json.dumps(jsonable({
        "case":            decision["case"],
        "diagnosis":       decision["diagnosis"][:200],
        "pm1_frac_total":  decision["total_gradient_frac_pm1"],
        "d_t_pm1_total":   decision["d_t_pm1_along_total_grad"],
        "t1_start":        decision["t1_start"],
        "t1_end":          decision["t1_end"],
        "avg_frac_pm1":    decision["avg_hgrad_frac_pm1"],
        "pytest":          pytest_r["summary_line"],
        "output_root":     str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
