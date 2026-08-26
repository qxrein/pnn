#!/usr/bin/env python3
"""Curvature-aware damped Gauss-Newton trust-region experiment.  Phases 0–6.

Motivation
----------
The preconditioned-gradient experiment (Case B) confirmed that the gradient
direction was correct (pm1_frac_after reached 97.5%) but |t_±1| did not
increase.  Case B diagnosis: curvature or competing loss blocks prevent
progress at the current step scale.

Approach
--------
The pure Gauss-Newton method requires a nonzero gradient J^T @ r at the
linearization point.  At theta=0, J^T @ r = 0 because the dominant residual
rows (pde_grat_4 and pde_grat_5, the contrast-source curl-H equations) are
driven by delta_eps * E_bg which is independent of the head parameters when
E_scat=0.  A warm-start is therefore essential.

Strategy: Hybrid warm-start + damped GN.
  1. Run N_WARMUP Adam steps to move theta away from zero (get nonzero J^T r).
  2. Switch to damped GN trust-region steps using the current J.
  3. Reassemble J every REASSEMBLE_EVERY GN iterations.

The warm-start uses the SAME PDE-only loss — no RCWA, no modal targets.

Safety invariants
-----------------
- PDE-only; RCWA withheld from the objective
- modal-data loss = 0
- optical coupling disabled
- canonical reference and companion: read-only
- all previous outputs preserved
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
from src.maxwell_layered_bg import (
    compute_background_coefficients,
    maxwell_2d_lbg_pde_residual,
    lbg_bottom_bc, lbg_top_bc, lbg_vertical_interface_loss,
)
from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable
from scripts.train_lbg import make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    BLOCK_ORDER, CANONICAL_REF, COMPANION_PATH, HEAD_PARAM_COUNT,
    MODAL_ORDER_MAX, N_DTN_ORDERS, SEED, TARGET_T1,
    head_parameter_records, load_points, model_config_hash, modal_report,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import (
    git_commit, git_status, make_fresh_model, parameter_ordering_hash,
    run_compileall, run_pytest,
)
from scripts.run_feature_scaling import (
    freeze_scaled_except_head, make_scaled_model, scaled_head_parameters,
)


# ── Patch spatial_fourier_t (requires_grad for ExplicitFourierModalNetwork) ──
def _sfq(model, physics, z_bot: float, n_quad: int = 512) -> dict[int, complex]:
    x = torch.linspace(0.0, physics.period, n_quad + 1,
                       dtype=torch.float64)[:-1].requires_grad_(True)
    z = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    e    = (er + 1j * ei).detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    g0   = 2.0 * np.pi / physics.period
    return {m: complex(np.mean(e * np.exp(-1j * m * g0 * x_np)))
            for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1)}

_fhls.spatial_fourier_t = _sfq


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_T1   = 0.00638
LAMBDA_INIT   = 1e-4   # initial damping (relative to max GN diagonal)
LAMBDA_MIN    = 1e-10
LAMBDA_MAX    = 1e6
RHO_DECREASE  = 0.25
RHO_INCREASE  = 0.75
LAMBDA_GROW   = 4.0
LAMBDA_SHRINK = 3.0
SMOKE_ITERS   = 500
LOG_EVERY     = 50
N_WARMUP      = 200   # Adam warm-start steps before switching to GN
WARMUP_LR     = 5e-4  # same as previous smoke runs
REASSEMBLE_EVERY = 50 # reassemble J every N GN iterations


# ─────────────────────────────────────────────────────────────────────────────
# Residual vector (all blocks stacked as real scalars)
# ─────────────────────────────────────────────────────────────────────────────

def compute_residual_vector(model, pts, physics, coeff) -> torch.Tensor:
    """Stack all PDE + interface + DtN residuals into a single 1D vector.

    Each complex PDE residual component is split into real and imaginary
    parts and then squared into a mean-MSE scalar — but for Gauss-Newton
    we need the *pre-squared* residual rows.  We return pointwise residuals
    (not squared), shape (n_rows,), so that J = d r / d theta correctly
    gives the Jacobian for the least-squares problem min ||r||^2.
    """
    p   = physics
    rows: list[torch.Tensor] = []

    def _pde_rows(key, net, eps_val):
        xk = pts.get(f"x_{key}")
        zk = pts.get(f"z_{key}")
        if xk is None or len(xk) == 0:
            return
        if key == "grat":
            eps_val = epsilon_r_fn(xk, zk, physics)
        res = maxwell_2d_lbg_pde_residual(net, xk, zk, physics, eps_val, coeff)
        for r in res:
            rows.append(r.reshape(-1))

    _pde_rows("air",  model.net_air,  p.n_air**2)
    _pde_rows("grat", model.net_grat, p.n_ridge**2)
    _pde_rows("sub",  model.net_sub,  p.n_substrate**2)

    # Vertical interfaces
    for x_iface, z_pts in (
        (p.ridge_x_min, pts["z_vleft"]),
        (p.ridge_x_max, pts["z_vright"]),
    ):
        LEv, LHv = lbg_vertical_interface_loss(model.net_grat, x_iface, z_pts)
        # lbg_vertical_interface_loss returns scalar MSE losses; we need raw residuals
        # Recompute raw:
        offset = 1e-5
        xa = torch.full_like(z_pts, x_iface - offset)
        xb = torch.full_like(z_pts, x_iface + offset)
        ela, eli, _, _, hzla, hzli = model.net_grat.field_components(xa, z_pts)
        erb, eib, _, _, hzrb, hzib = model.net_grat.field_components(xb, z_pts)
        rows.extend([ela - erb, eli - eib, hzla - hzrb, hzli - hzib])

    # Horizontal interfaces
    for net_a, net_b, z_int, x_key in (
        (model.net_air,  model.net_grat, p.ridge_z_min, "x_int1"),
        (model.net_grat, model.net_sub,  p.ridge_z_max, "x_int2"),
    ):
        x_pts = pts[x_key]
        z_t   = torch.full_like(x_pts, z_int)
        era, eia, hra, hia, _, _ = net_a.field_components(x_pts, z_t)
        erb, eib, hrb, hib, _, _ = net_b.field_components(x_pts, z_t)
        rows.extend([era - erb, eia - eib, hra - hrb, hia - hib])

    # DtN boundaries
    from src.modal_dtn import modal_dtn_pointwise
    dhr_t, dhi_t = modal_dtn_pointwise(
        model.net_air, pts["x_top"], 0.0, p, "top", p.n_air, N_DTN_ORDERS)
    dhr_b, dhi_b = modal_dtn_pointwise(
        model.net_sub, pts["x_bot"], p.domain_height, p, "bottom", p.n_substrate, N_DTN_ORDERS)
    rows.extend([dhr_t, dhi_t, dhr_b, dhi_b])

    return torch.cat([r.reshape(-1) for r in rows])


def residual_loss(r: torch.Tensor) -> torch.Tensor:
    """Mean squared residual: scalar loss from residual vector."""
    return torch.mean(r ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Full Jacobian assembly (all parameters)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_full_jacobian(
    model, params: list[torch.nn.Parameter],
    pts, physics, coeff,
    theta_linpoint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build J (n_rows × n_params) and r0 (n_rows,) via unit-basis evaluation.

    Evaluates around theta_linpoint (defaults to all-zeros if None).
    r0 = r(theta_linpoint).
    Column j = r(theta_linpoint + e_j) - r0.
    """
    n_params = sum(p.numel() for p in params)

    # Set to linearization point
    if theta_linpoint is not None:
        set_param_vec(params, theta_linpoint)
    else:
        with torch.no_grad():
            for p in params:
                p.zero_()

    r0 = compute_residual_vector(model, pts, physics, coeff).detach().numpy()
    n_rows   = len(r0)
    jacobian = np.zeros((n_rows, n_params), dtype=np.float64)

    offset = 0
    for pi, p in enumerate(params):
        n = p.numel()
        flat = p.data.reshape(-1)
        for j in range(n):
            flat[j] += 1.0   # add perturbation to current value
            r_j = compute_residual_vector(
                model, pts, physics, coeff).detach().numpy()
            jacobian[:, offset + j] = r_j - r0
            flat[j] -= 1.0   # restore
        offset += n
        if (pi + 1) % 10 == 0 or pi + 1 == len(params):
            print(f"    J col {offset}/{n_params}", flush=True)

    # Restore to linearization point
    if theta_linpoint is not None:
        set_param_vec(params, theta_linpoint)
    else:
        with torch.no_grad():
            for p in params:
                p.zero_()

    return jacobian, r0


# ─────────────────────────────────────────────────────────────────────────────
# Damped Gauss-Newton step
# ─────────────────────────────────────────────────────────────────────────────

def gn_step(
    J: np.ndarray,
    r: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float, float]:
    """Solve (J^T J + lambda * I) p = -J^T r.

    Returns (p, predicted_reduction, cond_number).
    """
    g        = J.T @ r                    # (n_params,)  gradient
    H        = J.T @ J                    # (n_params, n_params)  GN matrix
    n        = H.shape[0]
    A        = H + lam * np.eye(n, dtype=np.float64)

    # Cholesky solve (fast for positive-definite A)
    try:
        L = np.linalg.cholesky(A)
        p = np.linalg.solve(L.T, np.linalg.solve(L, -g))
    except np.linalg.LinAlgError:
        # Fall back to lstsq if not positive-definite
        p, *_ = np.linalg.lstsq(A, -g, rcond=None)

    # Predicted reduction: -g^T p - 0.5 p^T H p
    pred = -float(g @ p) - 0.5 * float(p @ (H @ p))

    # Condition number of A
    try:
        sv   = np.linalg.svd(A, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    except Exception:
        cond = float("inf")

    return p, pred, cond


# ─────────────────────────────────────────────────────────────────────────────
# Apply / retrieve parameter vector
# ─────────────────────────────────────────────────────────────────────────────

def get_param_vec(params: list[torch.nn.Parameter]) -> np.ndarray:
    return np.concatenate([p.data.detach().reshape(-1).numpy() for p in params])


def set_param_vec(params: list[torch.nn.Parameter], vec: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.data.copy_(torch.as_tensor(
                vec[offset:offset + n].reshape(p.shape),
                dtype=p.dtype, device=p.device))
            offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Smoke run
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(
    seed: int,
    physics,
    pts: dict,
    coeff: dict,
    device,
    dtype,
    ref,
    ref_path: Path,
    companion_p: Path,
    out_dir: Path,
    n_iters: int = SMOKE_ITERS,
    log_every: int = LOG_EVERY,
    lambda_init: float = LAMBDA_INIT,
    n_warmup: int = N_WARMUP,
) -> list[dict]:
    """Hybrid warm-start Adam + damped Gauss-Newton trust-region smoke run.

    Phase A (warm-start): N_WARMUP Adam steps to move theta away from zero,
    so that J^T @ r becomes nonzero before GN begins.  At theta=0 the dominant
    residual rows (pde_grat_4/5 contrast source) are independent of parameters,
    giving J^T @ r = 0 and making GN ineffective.

    Phase B (GN): n_iters Levenberg-Marquardt steps starting from the warm
    theta.  The Jacobian is reassembled every REASSEMBLE_EVERY steps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [GN smoke seed={seed}]  warmup={n_warmup} Adam  "
          f"+ {n_iters} GN  alpha={lambda_init:.1e}", flush=True)

    set_seed(seed)
    model_s, _ = make_scaled_model(physics, pts)
    for p in model_s.parameters():
        p.requires_grad_(True)

    all_params = list(model_s.parameters())
    n_params   = sum(p.numel() for p in all_params)

    from scripts.train_lbg import layered_bg_loss

    traj = []

    # ── Phase A: Adam warm-start ──────────────────────────────────────────────
    opt_adam = torch.optim.Adam(model_s.parameters(), lr=WARMUP_LR)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam, T_max=n_warmup)
    print(f"  Phase A: {n_warmup} Adam warm-start steps...", flush=True)
    for ws in range(1, n_warmup + 1):
        opt_adam.zero_grad(set_to_none=True)
        losses = layered_bg_loss(
            model_s, pts, physics, coeff,
            w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
            use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
            rcwa_amps=None, w_modal=0.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        opt_adam.step()
        sched.step()
        if ws % 50 == 0:
            r_ws = compute_residual_vector(model_s, pts, physics, coeff)
            g_ws = float(r_ws.detach().norm())
            pnorm = float(sum(p.data.norm()**2 for p in all_params)**0.5)
            print(f"    warmup ep={ws:3d}  ||r||={g_ws:.4e}  "
                  f"pde_grat={float(losses['pde_grat'].detach()):.3e}  "
                  f"||theta||={pnorm:.4e}", flush=True)

    # Check J^T @ r after warm-start
    r_warm = compute_residual_vector(model_s, pts, physics, coeff).detach().numpy()
    print(f"  Post-warmup ||r||={np.linalg.norm(r_warm):.4e}", flush=True)
    # Quick gradient check via autograd
    opt_adam.zero_grad(set_to_none=True)
    losses_check = layered_bg_loss(
        model_s, pts, physics, coeff,
        w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
        use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
        rcwa_amps=None, w_modal=0.0,
    )
    losses_check["total"].backward()
    g_autograd = float(sum(
        p.grad.norm()**2 for p in all_params if p.grad is not None
    )**0.5)
    print(f"  Post-warmup autograd ||g||={g_autograd:.4e}  "
          f"(should be >> 1e-10 for GN to work)", flush=True)
    opt_adam.zero_grad(set_to_none=True)

    if g_autograd < 1e-10:
        print("  WARNING: autograd gradient still near zero after warm-start. "
              "GN will likely fail.", flush=True)

    # Log warm-start final state
    modal_warm = modal_report(model_s, physics, coeff, device, dtype,
                              ref, ref_path, companion_p)
    print(f"  Post-warmup |t-1|={modal_warm['t_minus1_abs']:.4f}  "
          f"R+T={modal_warm['R_plus_T']:.4f}", flush=True)

    # ── Phase B: Damped GN trust-region ──────────────────────────────────────
    lam          = lambda_init
    theta_curr   = get_param_vec(all_params)
    J            = None   # assembled on first GN iteration
    r_for_gn     = None

    print(f"\n  Phase B: {n_iters} GN iterations...", flush=True)

    for it in range(1, n_iters + 1):
        # Assemble / reassemble Jacobian
        if J is None or (it > 1 and (it - 1) % REASSEMBLE_EVERY == 0):
            print(f"  it={it}: assembling J at current theta ({n_params} params)...",
                  flush=True)
            J, r_for_gn = assemble_full_jacobian(
                model_s, all_params, pts, physics, coeff,
                theta_linpoint=theta_curr)
            g_check = np.linalg.norm(J.T @ r_for_gn)
            print(f"    ||J^T r||={g_check:.4e}  ||r||={np.linalg.norm(r_for_gn):.4e}",
                  flush=True)
            if it == 1:
                # Scale lambda relative to GN diagonal at current point
                gn_diag = np.sum(J**2, axis=0)
                gn_diag_mean = float(np.mean(gn_diag))
                gn_diag_max  = float(np.max(gn_diag))
                # Use a stronger damping: 0.1 × max diagonal (not mean)
                lam = max(lambda_init * gn_diag_max, 1e-8)
                print(f"    GN diag mean={gn_diag_mean:.4e}  max={gn_diag_max:.4e}  "
                      f"lam reset to {lam:.4e}", flush=True)
        else:
            # Recompute r at current theta (J unchanged between reassembly)
            set_param_vec(all_params, theta_curr)
            r_for_gn = compute_residual_vector(
                model_s, pts, physics, coeff).detach().numpy()

        r_norm = float(np.linalg.norm(r_for_gn))
        L_curr = 0.5 * float(np.dot(r_for_gn, r_for_gn))

        # GN step
        p_step, pred, cond = gn_step(J, r_for_gn, lam)
        p_norm = float(np.linalg.norm(p_step))

        # Trial step
        theta_trial = theta_curr + p_step
        set_param_vec(all_params, theta_trial)
        r_trial = compute_residual_vector(
            model_s, pts, physics, coeff).detach().numpy()
        L_trial  = 0.5 * float(np.dot(r_trial, r_trial))
        actual   = L_curr - L_trial
        rho      = actual / (abs(pred) + 1e-30)

        if rho > 0:
            theta_curr = theta_trial
            r_for_gn   = r_trial
            r_norm     = float(np.linalg.norm(r_trial))
            accepted   = True
        else:
            set_param_vec(all_params, theta_curr)
            accepted   = False

        if rho < RHO_DECREASE:
            lam = min(lam * LAMBDA_GROW, LAMBDA_MAX)
        elif rho > RHO_INCREASE:
            lam = max(lam / LAMBDA_SHRINK, LAMBDA_MIN)

        if it % log_every == 0 or it == 1:
            modal_m   = modal_report(model_s, physics, coeff, device, dtype,
                                     ref, ref_path, companion_p)
            param_norm = float(np.linalg.norm(theta_curr))

            loss_d = layered_bg_loss(
                model_s, pts, physics, coeff,
                w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
                use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
                rcwa_amps=None, w_modal=0.0,
            )

            row = {
                "iter":            it,
                "phase":           "GN",
                "seed":            seed,
                "lambda":          lam,
                "rho":             rho,
                "accepted":        accepted,
                "r_norm":          r_norm,
                "L_curr":          L_curr,
                "pred_reduction":  pred,
                "actual_reduction": actual,
                "p_norm":          p_norm,
                "cond_number":     cond,
                "pde_grat":        float(loss_d["pde_grat"].detach()),
                "pde_air":         float(loss_d["pde_air"].detach()),
                "pde_sub":         float(loss_d["pde_sub"].detach()),
                "top_DtN":         float(loss_d["top"].detach()),
                "bottom_DtN":      float(loss_d["bottom"].detach()),
                "E_int":           float((loss_d["E_int1"]+loss_d["E_int2"]).detach()),
                "H_int":           float((loss_d["H_int1"]+loss_d["H_int2"]).detach()),
                "total_loss":      float(loss_d["total"].detach()),
                "param_norm":      param_norm,
                "t_minus1":        modal_m["t_minus1_abs"],
                "t_plus1":         modal_m["t_plus1_abs"],
                "R_plus_T":        modal_m["R_plus_T"],
                "total_complex_l2": modal_m["total_complex_l2"],
                "scattered_complex_l2": modal_m["scattered_complex_l2"],
            }
            traj.append(row)
            print(
                f"    it={it:4d}  r_norm={r_norm:.4e}  "
                f"|t-1|={row['t_minus1']:.4f}  R+T={row['R_plus_T']:.4f}  "
                f"lam={lam:.2e}  rho={rho:.3f}  "
                f"acc={'Y' if accepted else 'N'}  ||p||={p_norm:.3e}",
                flush=True)

    if traj:
        csv_path = out_dir / "trajectory.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader(); writer.writerows(traj)
        print(f"    Saved → {csv_path}", flush=True)

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(
    traj42: list[dict],
    traj43: list[dict] | None,
) -> dict:
    t1_vals  = [r["t_minus1"] for r in traj42]
    t1_start = t1_vals[0]
    t1_end   = t1_vals[-1]
    t1_max   = max(t1_vals)
    t1_max_it = traj42[t1_vals.index(t1_max)]["iter"]

    rt_max   = max(r["R_plus_T"] for r in traj42)
    stable   = rt_max < 1.1 and all(
        np.isfinite(r["t_minus1"]) for r in traj42)

    improving  = t1_max > 1.5 * BASELINE_T1
    recovering = t1_max > 0.25 * TARGET_T1

    accepted_count = sum(1 for r in traj42 if r["accepted"])
    all_rejected   = accepted_count == 0

    seed43_improving = False
    if traj43:
        t1_max43 = max(r["t_minus1"] for r in traj43)
        seed43_improving = t1_max43 > 1.5 * BASELINE_T1

    if all_rejected:
        case = "C"
        decision = (
            "All GN steps rejected (rho ≤ 0). Optimizer never accepted a step. "
            "Lambda may be too small (curvature estimate unreliable) or the "
            "residual vector is incompatible with the GN approximation. "
            "Try larger initial lambda or verify residual formulation."
        )
    elif not stable:
        case = "C"
        decision = (
            f"Optimizer became unstable (R+T max={rt_max:.3f} or NaN in outputs). "
            "Increase lambda or reduce step size."
        )
    elif recovering and stable:
        case = "A"
        decision = (
            f"GN optimization recovered |t_±1|={t1_max:.5f} "
            f"(peak iter={t1_max_it}, target={TARGET_T1:.5f}). "
            "Curvature-aware step breaks the m=0 basin. "
            + (f"Seed 43 also improving ({max(r['t_minus1'] for r in traj43):.5f})."
               if traj43 else "Two-seed confirmation run.")
        )
    elif improving and stable:
        case = "A"
        decision = (
            f"GN optimization shows material improvement: "
            f"|t_±1| reached {t1_max:.5f} (baseline={BASELINE_T1:.5f}). "
            "Directional progress confirmed. Full recovery not yet achieved."
        )
    elif accepted_count > 0 and not improving:
        case = "B"
        decision = (
            f"GN steps accepted ({accepted_count}/{len(traj42)}) but "
            f"|t_±1| ({t1_max:.5f}) did not materially exceed baseline ({BASELINE_T1:.5f}). "
            "Step norm may be too small or lambda too large. "
            "Inspect lambda trajectory and step norms."
        )
    else:
        case = "B"
        decision = (
            f"|t_±1| unchanged (max={t1_max:.5f}). GN provided no improvement. "
            "Inspect whether the parameterization prevents ±1 progress."
        )

    return {
        "case":                  case,
        "decision":              decision,
        "t1_start":              t1_start,
        "t1_end":                t1_end,
        "t1_max":                t1_max,
        "t1_max_iter":           t1_max_it,
        "baseline_t1":           BASELINE_T1,
        "target_t1":             TARGET_T1,
        "improving":             improving,
        "recovering":            recovering,
        "stable":                stable,
        "accepted_steps":        accepted_count,
        "total_steps":           len(traj42),
        "seed43_run":            traj43 is not None,
        "seed43_improving":      seed43_improving,
        "longer_run_authorized": False,
        "modal_data_loss":       False,
        "optical_coupling":      False,
        "start_10000_epoch_run": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def write_plots(
    root: Path,
    traj42: list[dict],
    traj43: list[dict] | None,
    baseline_traj: list[dict] | None,
) -> None:
    its42 = [r["iter"] for r in traj42]

    # Trust-region trajectory
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(its42, [r["lambda"] for r in traj42])
    axes[0].set_yscale("log")
    axes[0].set(xlabel="iter", ylabel="λ", title="Trust-region damping λ")

    axes[1].plot(its42, [r["rho"] for r in traj42])
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].axhline(RHO_DECREASE, color="r", ls=":", lw=0.8, label=f"ρ={RHO_DECREASE}")
    axes[1].axhline(RHO_INCREASE, color="g", ls=":", lw=0.8, label=f"ρ={RHO_INCREASE}")
    axes[1].set(xlabel="iter", ylabel="ρ", title="Actual/predicted ratio ρ")
    axes[1].legend(fontsize=8)

    axes[2].semilogy(its42, [r["p_norm"] for r in traj42], label="||p||")
    axes[2].semilogy(its42, [r["r_norm"] for r in traj42], ls="--", label="||r||")
    axes[2].set(xlabel="iter", ylabel="norm", title="Step / residual norms")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "trust_region_trajectory.png", dpi=160)
    plt.close(fig)

    # Modal trajectory comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(its42, [r["t_minus1"] for r in traj42], label="GN seed42")
    if traj43:
        its43 = [r["iter"] for r in traj43]
        axes[0].plot(its43, [r["t_minus1"] for r in traj43],
                     ls="--", label="GN seed43")
    if baseline_traj:
        ep_b = [int(float(r["epoch"])) for r in baseline_traj]
        axes[0].plot(ep_b, [float(r["t_minus1"]) for r in baseline_traj],
                     ls=":", color="gray", label="baseline (no precond)")
    axes[0].axhline(TARGET_T1, color="k", ls="--", lw=0.8, label="target")
    axes[0].axhline(BASELINE_T1, color="r", ls=":", lw=0.8, label="baseline")
    axes[0].set(xlabel="iter", ylabel="|t_{-1}|", title="Modal amplitude")
    axes[0].legend(fontsize=8)

    axes[1].plot(its42, [r["R_plus_T"] for r in traj42])
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="iter", ylabel="R+T", title="Energy balance")
    fig.tight_layout()
    fig.savefig(root / "modal_trajectory_comparison.png", dpi=160)
    plt.close(fig)

    # Loss components
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for comp in ("pde_grat", "pde_air", "top_DtN", "bottom_DtN", "E_int", "total_loss"):
        if comp in traj42[0]:
            ax.semilogy(its42, [r[comp] for r in traj42], label=comp)
    ax.legend(fontsize=8)
    ax.set(xlabel="iter", ylabel="loss", title="Loss components (seed 42)")
    fig.tight_layout()
    fig.savefig(root / "loss_component_comparison.png", dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Curvature-aware damped Gauss-Newton TR experiment")
    ap.add_argument("--output-root",   default="outputs/phase5_curvature_aware")
    ap.add_argument("--shared-points", default="outputs/phase5_feature_scaling/shared_points.npz")
    ap.add_argument("--iters",         type=int,   default=SMOKE_ITERS)
    ap.add_argument("--lambda-init",   type=float, default=LAMBDA_INIT)
    ap.add_argument("--skip-seed43",   action="store_true")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)

    # ── Phase 0: preflight ───────────────────────────────────────────────────
    can_sha = sha256_file(ROOT / CANONICAL_REF)
    com_sha = sha256_file(ROOT / COMPANION_PATH)
    man     = json.loads((ROOT / "outputs/reference_companion/manifest.json").read_text())
    assert can_sha == man["canonical_sha256"], "canonical SHA mismatch"
    assert com_sha == man["companion_sha256"], "companion SHA mismatch"

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

    fs_rep    = json.loads(
        (ROOT / "outputs/phase5_feature_scaling/feature_scale_report.json").read_text())
    fs_scales = fs_rep["phase1_feature_scales"]
    fs_hash   = hashlib.sha256(
        json.dumps({k: v["scales"] for k, v in fs_scales.items()},
                   sort_keys=True).encode()).hexdigest()

    phase0 = {
        "git_commit":             git_commit(),
        "working_tree_status":    git_status(),
        "canonical_sha256":       can_sha,
        "canonical_sha_ok":       True,
        "companion_sha256":       com_sha,
        "companion_sha_ok":       True,
        "model_config_hash":      model_config_hash(physics),
        "parameter_order_hash":   parameter_ordering_hash(
            head_parameter_records(make_fresh_model(physics))),
        "feature_scale_hash":     fs_hash,
        "seed":                   SEED,
        "optimizer":              "damped_gauss_newton_trust_region",
        "lambda_init":            args.lambda_init,
        "lambda_min":             LAMBDA_MIN,
        "lambda_max":             LAMBDA_MAX,
        "lambda_grow_factor":     LAMBDA_GROW,
        "lambda_shrink_factor":   LAMBDA_SHRINK,
        "rho_decrease_thresh":    RHO_DECREASE,
        "rho_increase_thresh":    RHO_INCREASE,
        "n_iters":                args.iters,
        "n_warmup_adam":          N_WARMUP,
        "warmup_lr":              WARMUP_LR,
        "log_every":              LOG_EVERY,
        "modal_data_loss_weight": 0.0,
        "optical_coupling":       False,
        "canonical_modified":     False,
        "companion_used_in_loss": False,
    }
    (root / "phase0_preflight.json").write_text(json.dumps(phase0, indent=2) + "\n")
    print("[Phase 0] Preflight OK", flush=True)

    # ── Phase 1: Jacobian feasibility ────────────────────────────────────────
    set_seed(SEED)
    m_tmp, _ = make_scaled_model(physics, pts)
    for p in m_tmp.parameters():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in m_tmp.parameters())
    n_head   = HEAD_PARAM_COUNT
    n_hidden = n_params - n_head

    # Estimate rows from a quick forward pass
    r_tmp = compute_residual_vector(m_tmp, pts, physics, coeff)
    n_rows = int(r_tmp.numel())
    J_mb   = n_rows * n_params * 8 / 1e6
    GN_mb  = n_params * n_params * 8 / 1e6
    del m_tmp, r_tmp

    jac_meta = {
        "n_rows":         n_rows,
        "n_params":       n_params,
        "n_head":         n_head,
        "n_hidden":       n_hidden,
        "J_memory_mb":    J_mb,
        "GN_memory_mb":   GN_mb,
        "method":         "warmstart_adam_then_damped_gauss_newton",
        "n_warmup_adam":  N_WARMUP,
        "warmup_lr":      WARMUP_LR,
        "assembly":       "column_by_column_unit_basis",
        "reassembly_every": REASSEMBLE_EVERY,
        "jacobian_evaluated_at": "theta_curr (not theta=0)",
        "root_cause_zero_gradient": (
            "At theta=0, pde_grat_4/5 rows dominate r but are driven by "
            "delta_eps*E_bg (independent of params when E_scat=0), so "
            "J^T@r=0 at theta=0.  Warm-start moves theta to a nonzero point "
            "where J^T@r is large enough for GN to take effective steps."
        ),
        "note": (f"J ({n_rows}×{n_params}) = {J_mb:.1f} MB, "
                 f"GN ({n_params}×{n_params}) = {GN_mb:.1f} MB. Tractable on CPU."),
    }
    (root / "jacobian_metadata.json").write_text(json.dumps(jac_meta, indent=2) + "\n")
    print(f"[Phase 1] J={n_rows}×{n_params} ({J_mb:.0f} MB)  "
          f"GN={n_params}×{n_params} ({GN_mb:.0f} MB)  → full GN feasible", flush=True)

    # ── Phase 2+3: Smoke run seed 42 ─────────────────────────────────────────
    traj42 = run_smoke(
        seed=SEED,
        physics=physics, pts=pts, coeff=coeff,
        device=device, dtype=dtype, ref=ref, ref_path=ref_path,
        companion_p=companion_p,
        out_dir=root / "smoke_500",
        n_iters=args.iters,
        lambda_init=args.lambda_init,
        n_warmup=N_WARMUP,
    )

    # ── Phase 4: Decision ────────────────────────────────────────────────────
    print("\n[Phase 4] Decision", flush=True)
    baseline_path = (ROOT / "outputs/phase5_gradient_alignment"
                     / "trajectory/training_trajectory.csv")
    baseline_traj = None
    if baseline_path.exists():
        baseline_traj = list(csv.DictReader(baseline_path.open()))

    dec_prelim = make_decision(traj42, None)
    print(f"  Preliminary case: {dec_prelim['case']}", flush=True)
    print(f"  t1_max={dec_prelim['t1_max']:.5f}  accepted={dec_prelim['accepted_steps']}"
          f"/{dec_prelim['total_steps']}", flush=True)

    # ── Phase 5: Two-seed confirmation if Case A ──────────────────────────────
    traj43 = None
    if dec_prelim["case"] == "A" and not args.skip_seed43:
        print("\n[Phase 5] Two-seed confirmation (seed=43)", flush=True)
        traj43 = run_smoke(
            seed=43,
            physics=physics, pts=pts, coeff=coeff,
            device=device, dtype=dtype, ref=ref, ref_path=ref_path,
            companion_p=companion_p,
            out_dir=root / "smoke_seed43",
            n_iters=args.iters,
            lambda_init=args.lambda_init,
            n_warmup=N_WARMUP,
        )
    else:
        reason = "Case A not met" if dec_prelim["case"] != "A" else "--skip-seed43"
        print(f"\n[Phase 5] Skipped ({reason})", flush=True)

    decision = make_decision(traj42, traj43)
    print(f"\n  Final case: {decision['case']}", flush=True)
    print(f"  {decision['decision'][:200]}", flush=True)

    # ── Phase 6: Verification + outputs ──────────────────────────────────────
    print("\n[Phase 6] Verification", flush=True)
    compile_r = run_compileall()
    pytest_r  = run_pytest()
    print(f"  compileall: {'OK' if compile_r['passed'] else 'FAIL'}")
    print(f"  pytest: {pytest_r['summary_line']}")

    write_plots(root, traj42, traj43, baseline_traj)

    summary = {
        **phase0,
        "jacobian_metadata": jac_meta,
        "decision":          decision,
        "seed42": {
            "t1_start":       traj42[0]["t_minus1"],
            "t1_end":         traj42[-1]["t_minus1"],
            "t1_max":         max(r["t_minus1"] for r in traj42),
            "t1_max_iter":    min(traj42, key=lambda r: -r["t_minus1"])["iter"],
            "R_plus_T_max":   max(r["R_plus_T"] for r in traj42),
            "r_norm_final":   traj42[-1]["r_norm"],
            "accepted_steps": decision["accepted_steps"],
            "total_steps":    decision["total_steps"],
            "lambda_final":   traj42[-1]["lambda"],
        },
        "seed43": {
            "t1_start": traj43[0]["t_minus1"],
            "t1_end":   traj43[-1]["t_minus1"],
            "t1_max":   max(r["t_minus1"] for r in traj43),
        } if traj43 else None,
        "baseline_t1":   BASELINE_T1,
        "target_t1":     TARGET_T1,
        "phase5_passed": False,
        "compileall":    compile_r,
        "pytest":        pytest_r,
    }
    (root / "curvature_aware_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n")

    # CSV
    all_rows = []
    for tr in ([traj42] + ([traj43] if traj43 else [])):
        for r in tr:
            all_rows.append({
                "seed":          r["seed"],
                "iter":          r["iter"],
                "lambda":        r["lambda"],
                "rho":           r["rho"],
                "accepted":      r["accepted"],
                "r_norm":        r["r_norm"],
                "p_norm":        r["p_norm"],
                "t_minus1":      r["t_minus1"],
                "t_plus1":       r["t_plus1"],
                "R_plus_T":      r["R_plus_T"],
                "total_complex_l2": r["total_complex_l2"],
                "scattered_complex_l2": r["scattered_complex_l2"],
                "pde_grat":      r["pde_grat"],
                "top_DtN":       r["top_DtN"],
                "bottom_DtN":    r["bottom_DtN"],
                "total_loss":    r["total_loss"],
                "param_norm":    r["param_norm"],
                "cond_number":   r["cond_number"],
            })
    if all_rows:
        with (root / "curvature_aware_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)

    print("\n" + "=" * 70, flush=True)
    print(json.dumps(jsonable({
        "case":            decision["case"],
        "decision":        decision["decision"][:200],
        "t1_max_seed42":   decision["t1_max"],
        "t1_end_seed42":   decision["t1_end"],
        "accepted_steps":  decision["accepted_steps"],
        "baseline_t1":     BASELINE_T1,
        "target_t1":       TARGET_T1,
        "seed43_run":      decision["seed43_run"],
        "longer_run_auth": decision["longer_run_authorized"],
        "pytest":          pytest_r["summary_line"],
        "output_root":     str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
