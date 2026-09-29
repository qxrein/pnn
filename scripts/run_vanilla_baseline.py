#!/usr/bin/env python3
"""Vanilla Raissi-style PINN baseline — gradient-alignment control experiment.

PURPOSE
-------
Tests whether the gradient-alignment obstruction identified in the modal
ExplicitFourierModalDD architecture (Proposition 1) is specific to that
architecture's z-only hidden features, or is a more general property of
PINNs on the binary grating problem.

WHAT THIS DOES
--------------
1. Builds a monolithic (x,z)->field MLP (VanillaPINN from src/vanilla_pinn.py)
   — 4 hidden layers × 32 units, tanh, input (x,z) jointly, no Fourier basis.
2. Trains with PURE PDE + interface + DtN boundary loss (same as our modal
   "PDE-only (Adam)" baseline — no auxiliary ±1 loss).
3. Measures:
   a. Final |t_{-1}|, |t_{+1}|, |t_0| via Fourier projection at z_bot monitor
   b. R+T energy balance
   c. frac_pm1 = ||P_{±1} g|| / ||g|| at zero-field initialization,
      adapted to the vanilla model's full parameter vector (not frozen head).
4. Reports a comparison table matching Table 1 in the paper.

ARCHITECTURE NOTE
-----------------
The vanilla model has 3462 trainable parameters vs 1620 for the modal model.
With MORE capacity, any failure to recover ±1 is strictly harder to explain
by insufficient expressiveness — it can only be explained by a training
dynamics failure, i.e. a gradient-alignment issue.

OUTPUT
------
outputs/vanilla_baseline/
    vanilla_summary.json       — all results + Table 1 comparison
    vanilla_summary.csv
    trajectory_seed42.csv      — epoch-by-epoch metrics
    trajectory_seed43.csv
    vanilla_field_plots/       — |E|, error map at best epoch
    obstruction_metric.json    — frac_pm1 at init, J_pm1 SVD data
"""
from __future__ import annotations

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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, load_config
from src.geometry import epsilon_r as epsilon_r_fn
from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss
from src.maxwell_layered_bg import (
    background_field_np,
    background_field_torch,
    compute_background_coefficients,
    delta_eps_tensor,
    lbg_bottom_bc,
    lbg_top_bc,
    lbg_vertical_interface_loss,
    maxwell_2d_lbg_pde_residual,
)
from src.field_comparison import extract_total_modal_amplitudes
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.reference_validation import validate_reference
from src.utils import set_seed
from src.vanilla_pinn import VanillaPINN, N_LAYERS, N_UNITS
from scripts.train_lbg import make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, N_DTN_ORDERS, SEED, TARGET_T1,
)

# ── SHA validation helpers ────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# Constants matching our existing baseline
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS        = 600
LR            = 5e-4
CONFIRM_SEED  = 43
LOG_EVERY     = 100
N_COLLOC      = 512     # points per region (matches phase5_pm1_aux shared points)
N_INTERFACE   = 256
N_BC          = 256
N_QUAD        = 512     # quadrature points for Fourier extraction
MODAL_ORDER_MAX = 3
Z_BOT_FRAC    = 0.92   # transmission monitor plane

# Reference baselines from Table 1 (paper)
BASELINE_MODAL_T1  = 0.006    # PDE-only modal, |t_{-1}|
BASELINE_MODAL_RT  = 1.000    # R+T
PREC_GRAD_T1       = 0.006    # preconditioned gradient, |t_{-1}|
PREC_GRAD_RT       = 1.003
AUX_LOSS_T1        = 0.019    # ±1 auxiliary loss, |t_{-1}|
AUX_LOSS_RT        = 1.005
RCWA_TARGET_T1     = 0.049    # RCWA reference

# ─────────────────────────────────────────────────────────────────────────────
# Collocation point sampling  (same regions as existing experiments)
# ─────────────────────────────────────────────────────────────────────────────

def sample_points(physics: PhysicsConfig, seed: int = SEED) -> dict:
    """Sample collocation points matching the layered_bg_loss expected keys."""
    rng    = np.random.default_rng(seed)
    p      = physics
    margin = 5e-3
    dtype  = torch.float64
    dev    = torch.device("cpu")
    def _t(a): return torch.as_tensor(a, dtype=dtype, device=dev)

    x_air  = rng.uniform(0, p.period, N_COLLOC)
    z_air  = rng.uniform(margin, p.ridge_z_min - margin, N_COLLOC)
    x_grat = rng.uniform(0, p.period, N_COLLOC)
    z_grat = rng.uniform(p.ridge_z_min + margin, p.ridge_z_max - margin, N_COLLOC)
    x_sub  = rng.uniform(0, p.period, N_COLLOC)
    z_sub  = rng.uniform(p.ridge_z_max + margin, p.domain_height - margin, N_COLLOC)
    x_int1 = rng.uniform(0, p.period, N_INTERFACE)
    x_int2 = rng.uniform(0, p.period, N_INTERFACE)
    x_top  = rng.uniform(0, p.period, N_BC)
    x_bot  = rng.uniform(0, p.period, N_BC)
    z_vl   = rng.uniform(p.ridge_z_min + margin, p.ridge_z_max - margin, N_INTERFACE)
    z_vr   = rng.uniform(p.ridge_z_min + margin, p.ridge_z_max - margin, N_INTERFACE)

    return {
        "x_air":   _t(x_air),  "z_air":   _t(z_air),
        "x_grat":  _t(x_grat), "z_grat":  _t(z_grat),
        "x_sub":   _t(x_sub),  "z_sub":   _t(z_sub),
        "x_int1":  _t(x_int1), "x_int2":  _t(x_int2),
        "x_top":   _t(x_top),  "x_bot":   _t(x_bot),
        "z_vleft": _t(z_vl),   "z_vright": _t(z_vr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDE + boundary loss (mirrors layered_bg_loss for vanilla model)
# ─────────────────────────────────────────────────────────────────────────────

def vanilla_loss(
    model: VanillaPINN,
    pts: dict,
    physics: PhysicsConfig,
    coeff: dict,
) -> dict:
    """Full PDE + interface + DtN loss, identical to layered_bg_loss call in
    the modal baseline (w_pde=w_E=w_H=w_top=w_bot=1.0, DtN enabled)."""
    zero = torch.zeros(1, dtype=torch.float64)
    p    = physics

    def _pde(key, net, eps_val):
        xk = pts.get(f"x_{key}"); zk = pts.get(f"z_{key}")
        if xk is None or not len(xk): return zero
        if key == "grat":
            eps_val = epsilon_r_fn(xk, zk, physics)
        res = maxwell_2d_lbg_pde_residual(net, xk, zk, physics, eps_val, coeff)
        return sum(torch.mean(r**2) for r in res) / len(res)

    La  = _pde("air",  model.net_air,  p.n_air**2)
    Lg  = _pde("grat", model.net_grat, p.n_ridge**2)
    Ls  = _pde("sub",  model.net_sub,  p.n_substrate**2)

    LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat, p.ridge_z_min, pts["x_int1"])
    LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,  p.ridge_z_max, pts["x_int2"])

    LEv_l, LHv_l = lbg_vertical_interface_loss(model.net_grat, p.ridge_x_min, pts["z_vleft"])
    LEv_r, LHv_r = lbg_vertical_interface_loss(model.net_grat, p.ridge_x_max, pts["z_vright"])

    Lt = lbg_top_bc(model.net_air, pts["x_top"], physics,
                    use_dtn=True, n_dtn_orders=N_DTN_ORDERS)
    Lb = lbg_bottom_bc(model.net_sub, pts["x_bot"], physics, coeff,
                       use_dtn=True, n_dtn_orders=N_DTN_ORDERS)

    total = La + Lg + Ls + LE1 + LH1 + LE2 + LH2 + LEv_l + LHv_l + LEv_r + LHv_r + Lt + Lb
    return {
        "pde_air": La, "pde_grat": Lg, "pde_sub": Ls,
        "E_int": LE1 + LE2, "H_int": LH1 + LH2,
        "E_vert": LEv_l + LEv_r, "H_vert": LHv_l + LHv_r,
        "top_DtN": Lt, "bottom_DtN": Lb,
        "total": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modal amplitude extraction at z_bot (scattered field DFT)
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_fourier_t(model: VanillaPINN, physics: PhysicsConfig,
                       z_bot: float, n_quad: int = N_QUAD) -> dict[int, complex]:
    """Extract scattered transmission amplitudes t_m via spatial DFT at z_bot.

    Uses model.net_sub (which routes to the backbone) — matches the existing
    spatial_fourier_t convention used throughout the codebase.
    """
    x  = torch.linspace(0.0, physics.period, n_quad + 1,
                        dtype=torch.float64)[:-1].requires_grad_(True)
    z  = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    E   = (er + 1j * ei).detach().numpy()
    xn  = x.detach().numpy()
    G0  = 2.0 * np.pi / physics.period
    return {
        m: complex(np.mean(E * np.exp(-1j * m * G0 * xn)))
        for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1)
    }


def _modal_report(model: VanillaPINN, physics: PhysicsConfig) -> dict:
    """Return |t_{-1}|, |t_{+1}|, |t_0| from scattered DFT at z_bot monitor."""
    z_bot = Z_BOT_FRAC * physics.domain_height
    t_m   = _spatial_fourier_t(model, physics, z_bot)
    t_m1  = abs(t_m[-1])
    t_p1  = abs(t_m[+1])
    t_0   = abs(t_m[ 0])
    return {"t_minus1_abs": t_m1, "t_plus1_abs": t_p1, "t_0_abs": t_0,
            "t_m_complex": {str(m): {"re": v.real, "im": v.imag}
                            for m, v in t_m.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# R+T energy balance from full field evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _energy_balance(model: VanillaPINN, physics: PhysicsConfig,
                    coeff: dict, n_x: int = 128, n_z: int = 256) -> float:
    """Compute R+T by evaluating the full field on a visualization grid."""
    p   = physics
    x1d = np.linspace(0, p.period, n_x, endpoint=False)
    z1d = np.linspace(0, p.domain_height, n_z)
    X, Z = np.meshgrid(x1d, z1d)
    xf   = torch.as_tensor(X.ravel(), dtype=torch.float64)
    zf   = torch.as_tensor(Z.ravel(), dtype=torch.float64)

    with torch.no_grad():
        er_s, ei_s, *_ = model.backbone.field_components(xf, zf)
    Er_s = er_s.numpy().reshape(n_z, n_x)
    Ei_s = ei_s.numpy().reshape(n_z, n_x)

    # Add background to get total field
    Ebg_r, Ebg_i, _, _ = background_field_np(z1d, coeff)
    Er_tot = Er_s + Ebg_r[:, None]
    Ei_tot = Ei_s + Ebg_i[:, None]
    E_tot  = Er_tot + 1j * Ei_tot

    result = extract_total_modal_amplitudes(
        E_tot, x1d, z1d, physics,
        n_orders=5, z_top_frac=0.08, z_bot_frac=0.92,
        formulation="layered_bg",
    )
    return float(result["energy_check"]), result


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-alignment obstruction metric (adapted for vanilla model)
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_all(params: list[nn.Parameter], vec: np.ndarray) -> None:
    """Write vec into the model's parameters (all, not just head)."""
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.data.copy_(
                torch.as_tensor(vec[offset:offset + n].reshape(p.shape),
                                 dtype=p.dtype))
            offset += n


def _pack_all(params: list[nn.Parameter]) -> np.ndarray:
    return np.concatenate([p.data.detach().reshape(-1).numpy() for p in params])


def compute_obstruction_metric(
    model: VanillaPINN,
    pts: dict,
    physics: PhysicsConfig,
    coeff: dict,
    n_quad: int = 256,
) -> dict:
    """Compute frac_pm1 for the vanilla model at zero-field initialization.

    Steps:
    1. Verify zero-field init (all params = 0 in backbone head).
    2. Build J_pm1 (4, N_params) via finite differences on t_{±1} DFT.
    3. Compute SVD of J_pm1 → V_pm1 (right singular vectors = ±1 subspace).
    4. Compute g = total PDE gradient w.r.t. ALL backbone parameters.
    5. frac_pm1 = ||V_pm1^T g|| / ||g||.

    Note: unlike the modal model where only the "head" params are active
    (the hidden layers are frozen), here ALL parameters are trainable — so
    g has dimension N_params = 3462.  The J_pm1 finite-difference loop is
    correspondingly slower (3462 columns vs 1386 for the modal model).
    """
    print("\n[Obstruction] Computing frac_pm1 for VanillaPINN...", flush=True)
    params  = list(model.backbone.parameters())
    n_p     = sum(p.numel() for p in params)
    z_bot   = Z_BOT_FRAC * physics.domain_height
    G0      = 2.0 * np.pi / physics.period
    print(f"  N_params = {n_p}", flush=True)

    # --- Step 0: save initial state, verify zero output ---
    theta0 = _pack_all(params).copy()
    max_E0 = model.zero_output_check()
    print(f"  Zero-output check: max|E_scat| at init = {max_E0:.2e}", flush=True)

    # Reset to zero (backbone head is zero by construction, but be explicit)
    # The hidden layers have random Xavier weights — we keep them as-is since
    # only the FINAL head is zeroed.  This matches the modal model: hidden
    # features are live, final head is zero, so E_scat=0 at init.
    # We do NOT zero the entire model — that would be a different init.
    # The current state IS the correct zero-field init: output is zero,
    # hidden features are nonzero.

    # --- Step 1: Build J_pm1 (4, n_p) via finite differences ---
    # t_{-1} and t_{+1} from DFT of E_scat at z_bot monitor
    def _eval_t_pm1(theta_vec: np.ndarray) -> np.ndarray:
        """Returns [Re t_{-1}, Im t_{-1}, Re t_{+1}, Im t_{+1}]."""
        _unpack_all(params, theta_vec)
        x  = torch.linspace(0.0, physics.period, n_quad + 1,
                             dtype=torch.float64)[:-1].requires_grad_(True)
        z  = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
        er, ei, *_ = model.net_sub.field_components(x, z)
        E   = (er + 1j * ei).detach().numpy()
        xn  = x.detach().numpy()
        t_m1 = complex(np.mean(E * np.exp(-1j * (-1) * G0 * xn)))
        t_p1 = complex(np.mean(E * np.exp(-1j * (+1) * G0 * xn)))
        return np.array([t_m1.real, t_m1.imag, t_p1.real, t_p1.imag])

    t_at_zero = _eval_t_pm1(theta0)
    print(f"  t_{{-1}} at init: {t_at_zero[0]:.4e}+{t_at_zero[1]:.4e}j", flush=True)
    print(f"  t_{{+1}} at init: {t_at_zero[2]:.4e}+{t_at_zero[3]:.4e}j", flush=True)

    J_pm1 = np.zeros((4, n_p), dtype=np.float64)
    e_j   = np.zeros(n_p)
    print(f"  Building J_pm1 ({4}×{n_p}) via FD... (may take a minute)", flush=True)
    for j in range(n_p):
        e_j[j] = 1.0
        t_j    = _eval_t_pm1(theta0 + e_j)
        J_pm1[:, j] = t_j - t_at_zero
        e_j[j] = 0.0
        if (j + 1) % 500 == 0:
            print(f"    J_pm1 column {j+1}/{n_p}", flush=True)
    # Restore init state
    _unpack_all(params, theta0)
    print("  J_pm1 done.", flush=True)

    # --- Step 2: SVD of J_pm1 → ±1 subspace ---
    _, sigma_pm1, Vt_pm1 = np.linalg.svd(J_pm1, full_matrices=False)
    rank_pm1 = max(int(np.sum(sigma_pm1 > 0.01 * sigma_pm1[0])), 1)
    V_pm1    = Vt_pm1[:rank_pm1, :].T       # (n_p, rank_pm1)
    print(f"  ±1 subspace rank = {rank_pm1}, sigma_max = {sigma_pm1[0]:.4f}", flush=True)

    # --- Step 3: PDE gradient g at zero-field init ---
    _unpack_all(params, theta0)
    for p in params:
        p.requires_grad_(True)
    for p in params:
        if p.grad is not None: p.grad.zero_()

    losses = vanilla_loss(model, pts, physics, coeff)
    losses["total"].backward()

    g = np.concatenate([
        p.grad.detach().reshape(-1).numpy() if p.grad is not None
        else np.zeros(p.numel())
        for p in params
    ])
    g_norm = float(np.linalg.norm(g))
    print(f"  ||g|| = {g_norm:.6f}", flush=True)

    # --- Step 4: Projection ---
    proj     = V_pm1.T @ g          # (rank_pm1,)
    proj_norm = float(np.linalg.norm(proj))
    frac_pm1 = proj_norm / (g_norm + 1e-30)
    print(f"  frac_pm1 = ||P_pm1 g||/||g|| = {frac_pm1:.6e}", flush=True)

    # Zero grads after use
    for p in params:
        if p.grad is not None: p.grad.zero_()
    # Restore to init state (all params)
    _unpack_all(params, theta0)

    return {
        "n_params":        n_p,
        "g_norm":          g_norm,
        "frac_pm1":        frac_pm1,
        "proj_norm_pm1":   proj_norm,
        "pm1_subspace_rank": rank_pm1,
        "pm1_sigma_max":   float(sigma_pm1[0]),
        "pm1_sigma_vals":  sigma_pm1.tolist(),
        "t_at_zero":       t_at_zero.tolist(),
        "max_E0_at_init":  max_E0,
        "interpretation": (
            "OBSTRUCTED (same as modal)" if frac_pm1 < 1e-4
            else "NOT obstructed (gradient reaches ±1 subspace)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training run
# ─────────────────────────────────────────────────────────────────────────────

def train_run(
    seed: int,
    physics: PhysicsConfig,
    pts: dict,
    coeff: dict,
    epochs: int,
    out_dir: Path,
    label: str,
) -> tuple[list[dict], dict, dict]:
    """Train vanilla PINN with PDE-only loss for `epochs` epochs.

    Returns (trajectory, best_state_dict, best_metrics).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Train] {label}  seed={seed}  epochs={epochs}", flush=True)

    set_seed(seed)
    model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    n_p   = model.n_params()
    print(f"  VanillaPINN: {n_p} parameters", flush=True)

    # Verify zero output
    max_e0 = model.zero_output_check()
    assert max_e0 < 1e-14, f"Zero-output init failed: max|E|={max_e0:.2e}"
    print(f"  Zero-output check passed: max|E|={max_e0:.2e}", flush=True)

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Compute initial modal report
    modal0 = _modal_report(model, physics)
    print(f"  Init: |t_{{-1}}|={modal0['t_minus1_abs']:.5f}", flush=True)

    traj: list[dict] = []
    best = {"t1": 0.0, "epoch": 0, "state": None, "RT": 0.0}
    RT_MAX_VALID = 1.05

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        losses_d = vanilla_loss(model, pts, physics, coeff)
        losses_d["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % LOG_EVERY == 0 or ep == 1:
            model.eval()
            modal_m = _modal_report(model, physics)
            t1      = modal_m["t_minus1_abs"]
            l_pde   = float((losses_d["pde_air"] + losses_d["pde_grat"] + losses_d["pde_sub"]).detach())
            l_tot   = float(losses_d["total"].detach())

            # Quick R+T (cheap: no full grid, just use t_0 from scattered DFT)
            # We use a fast approximation: report full R+T only at final epoch
            # to avoid the cost of grid evaluation at every checkpoint.
            row = {
                "epoch":      ep,
                "seed":       seed,
                "label":      label,
                "pde_air":    float(losses_d["pde_air"].detach()),
                "pde_grat":   float(losses_d["pde_grat"].detach()),
                "pde_sub":    float(losses_d["pde_sub"].detach()),
                "top_DtN":    float(losses_d["top_DtN"].detach()),
                "bottom_DtN": float(losses_d["bottom_DtN"].detach()),
                "L_total":    l_tot,
                "t_minus1":   t1,
                "t_plus1":    modal_m["t_plus1_abs"],
                "t_0":        modal_m["t_0_abs"],
            }
            traj.append(row)

            # Track best
            if t1 > best["t1"]:
                best["t1"]    = t1
                best["epoch"] = ep
                best["state"] = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

            if ep % (LOG_EVERY * 5) == 0 or ep == 1:
                print(f"  ep={ep:4d}  L_pde={l_pde:.3e}  |t_{{-1}}|={t1:.5f}",
                      flush=True)

    print(f"  Best |t_{{-1}}|={best['t1']:.5f}  @ epoch {best['epoch']}", flush=True)

    # --- Full R+T at final epoch ---
    model.eval()
    RT_final, modal_full = _energy_balance(model, physics, coeff)
    best_t1_full = modal_full["t_m_abs"][5 - 1]   # index for m=-1 with n_orders=5
    print(f"  Final R+T={RT_final:.5f}  |t_{{-1}}| (full grid)={best_t1_full:.5f}",
          flush=True)

    # Load best checkpoint for full R+T check
    if best["state"] is not None:
        set_seed(seed)
        model_best = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
        model_best.load_state_dict(best["state"])
        model_best.eval()
        RT_best, modal_best_full = _energy_balance(model_best, physics, coeff)
        t1_best_full = modal_best_full["t_m_abs"][5 - 1]
        print(f"  Best ckpt R+T={RT_best:.5f}  |t_{{-1}}|={t1_best_full:.5f}", flush=True)
        best["RT"]       = RT_best
        best["t1_full"]  = t1_best_full
        best["modal_full"] = modal_best_full
    else:
        RT_best     = RT_final
        t1_best_full = best_t1_full
        best["modal_full"] = modal_full
        best["RT"]       = RT_final
        best["t1_full"]  = t1_best_full

    # Save trajectory CSV
    if traj:
        with (out_dir / f"trajectory_{label}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader(); writer.writerows(traj)

    # Save best checkpoint
    if best["state"] is not None:
        torch.save({
            "state_dict": best["state"],
            "epoch":      best["epoch"],
            "t1_abs":     best["t1"],
            "seed":       seed,
        }, out_dir / f"best_checkpoint_{label}.pt")

    return traj, best, model


# ─────────────────────────────────────────────────────────────────────────────
# Field visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_field_plots(
    model: VanillaPINN,
    physics: PhysicsConfig,
    coeff: dict,
    ref,
    out_dir: Path,
    label: str,
) -> None:
    """Generate |E_y| comparison and error map."""
    p   = physics
    x1d = np.linspace(0, p.period, p.nx_visualization, endpoint=False)
    z1d = np.linspace(0, p.domain_height, p.nz_visualization)
    X, Z = np.meshgrid(x1d, z1d)
    xf   = torch.as_tensor(X.ravel(), dtype=torch.float64)
    zf   = torch.as_tensor(Z.ravel(), dtype=torch.float64)

    with torch.no_grad():
        er_s, ei_s, *_ = model.backbone.field_components(xf, zf)
    Er_s = er_s.numpy().reshape(p.nz_visualization, p.nx_visualization)
    Ei_s = ei_s.numpy().reshape(p.nz_visualization, p.nx_visualization)

    Ebg_r, Ebg_i, _, _ = background_field_np(z1d, coeff)
    Er_tot = Er_s + Ebg_r[:, None]
    Ei_tot = Ei_s + Ebg_i[:, None]
    E_pinn = Er_tot + 1j * Ei_tot

    rr, ri, _ = interpolate_reference_to_grid(ref, x1d, z1d)
    E_rcwa    = rr + 1j * ri

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].pcolormesh(X, Z, np.abs(E_pinn), cmap="viridis", shading="auto")
    axes[0].set(title=f"Vanilla PINN |E_y| ({label})", xlabel="x/λ", ylabel="z/λ")
    axes[0].invert_yaxis(); plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Z, np.abs(E_rcwa), cmap="viridis", shading="auto")
    axes[1].set(title="RCWA |E_y|", xlabel="x/λ"); axes[1].invert_yaxis()
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(E_pinn - E_rcwa)
    im2  = axes[2].pcolormesh(X, Z, diff, cmap="inferno", shading="auto")
    axes[2].set(title=r"$|E_{\rm PINN} - E_{\rm RCWA}|$", xlabel="x/λ")
    axes[2].invert_yaxis(); plt.colorbar(im2, ax=axes[2])

    # Mark grating layer
    for ax in axes:
        ax.axhline(p.ridge_z_min, color="white", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(p.ridge_z_max, color="white", lw=0.7, ls="--", alpha=0.6)

    fig.tight_layout()
    _savefig(fig, out_dir / f"field_comparison_{label}.png")

    # Modal bar chart
    z_bot = Z_BOT_FRAC * p.domain_height
    t_m   = _spatial_fourier_t(model, p, z_bot)
    orders = list(range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX + 1))
    t_pinn = [abs(t_m[m]) for m in orders]

    # RCWA reference from companion
    companion_path = ROOT / COMPANION_PATH
    if companion_path.exists():
        companion = np.load(companion_path, allow_pickle=False)
        n_harm = int((len(companion["c_trans"]) - 1) // 2)
        t_rcwa = []
        for m in orders:
            idx = n_harm + m
            t_c = abs(complex(companion["c_trans"][idx]))
            kz  = complex(companion["kz_substrate"][idx])
            t_c *= abs(np.exp(-1j * kz * (z_bot - p.ridge_z_max)))
            t_rcwa.append(t_c)
    else:
        t_rcwa = [0.0] * len(orders)

    x_pos = np.arange(len(orders))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_pos - 0.2, t_rcwa, 0.4, label="RCWA", color="steelblue")
    ax.bar(x_pos + 0.2, t_pinn, 0.4, label=f"Vanilla PINN ({label})",
           color="coral", hatch="////", edgecolor="black", lw=0.5)
    ax.set_xticks(x_pos); ax.set_xticklabels([str(m) for m in orders])
    ax.set(xlabel="Diffraction order m", ylabel=r"$|t_m|$",
           title=f"Modal amplitudes — Vanilla PINN vs RCWA ({label})")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / f"modal_comparison_{label}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = ROOT / "outputs/vanilla_baseline"
    if out_dir.exists():
        import shutil
        print(f"Removing existing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"Output: {out_dir}", flush=True)

    # ── Preflight ──────────────────────────────────────────────────────────
    ref_path    = ROOT / CANONICAL_REF
    companion_p = ROOT / COMPANION_PATH
    man_path    = ROOT / "outputs/reference_companion/manifest.json"
    man         = json.loads(man_path.read_text())
    assert _sha256(ref_path)    == man["canonical_sha256"], "canonical SHA mismatch"
    assert _sha256(companion_p) == man["companion_sha256"], "companion SHA mismatch"
    print("[Preflight] Reference hashes OK", flush=True)

    cfg     = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    coeff   = compute_background_coefficients(physics)
    ref     = normalize_reference_orientation(load_reference_npz(ref_path))

    # Validate reference against the lambda_0p8 physics (period=0.8), not default
    validate_reference(ref_path, physics)

    # Sanity print
    print(f"  Physics: Λ={physics.period:.3f}λ  dc={physics.ridge_width/physics.period:.2f}"
          f"  h={physics.ridge_height:.2f}  n_ridge={physics.n_ridge}"
          f"  n_sub={physics.n_substrate}  k0={physics.k0:.4f}", flush=True)

    # Sample collocation points (shared across seeds, same as existing runs)
    pts = sample_points(physics, seed=SEED)
    print(f"  Collocation: {N_COLLOC}/region, {N_INTERFACE} interface, {N_BC} BC",
          flush=True)

    # ── Step 1: Obstruction metric at init ─────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("STEP 1: Gradient-alignment obstruction metric (at init)", flush=True)
    print("="*60, flush=True)

    set_seed(SEED)
    model_init = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    obs_result = compute_obstruction_metric(model_init, pts, physics, coeff)
    (out_dir / "obstruction_metric.json").write_text(
        json.dumps(obs_result, indent=2) + "\n")
    print(f"\n  frac_pm1 = {obs_result['frac_pm1']:.4e}  → {obs_result['interpretation']}",
          flush=True)

    # ── Step 2: Training — seed 42 ─────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("STEP 2: PDE-only training  (seed 42, 600 epochs)", flush=True)
    print("="*60, flush=True)

    traj42, best42, model42 = train_run(
        seed=SEED, physics=physics, pts=pts, coeff=coeff,
        epochs=EPOCHS, out_dir=out_dir, label="seed42",
    )

    # ── Step 3: Training — seed 43 ─────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("STEP 3: PDE-only training  (seed 43, 600 epochs)", flush=True)
    print("="*60, flush=True)

    traj43, best43, model43 = train_run(
        seed=CONFIRM_SEED, physics=physics, pts=pts, coeff=coeff,
        epochs=EPOCHS, out_dir=out_dir, label="seed43",
    )

    # ── Step 4: Field plots ────────────────────────────────────────────────
    print("\n[Plots] Generating field plots...", flush=True)
    plot_dir = out_dir / "vanilla_field_plots"
    make_field_plots(model42, physics, coeff, ref, plot_dir, "seed42")
    make_field_plots(model43, physics, coeff, ref, plot_dir, "seed43")

    # ── Step 5: Trajectory plots ───────────────────────────────────────────
    if traj42:
        ep42 = [r["epoch"] for r in traj42]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ep42, [r["t_minus1"] for r in traj42], label="seed 42 |t_{-1}|")
        if traj43:
            ep43 = [r["epoch"] for r in traj43]
            axes[0].plot(ep43, [r["t_minus1"] for r in traj43],
                         ls="--", label="seed 43 |t_{-1}|")
        axes[0].axhline(RCWA_TARGET_T1, color="k", ls=":", lw=1.0, label="RCWA target")
        axes[0].axhline(BASELINE_MODAL_T1, color="r", ls=":", lw=0.8,
                        label="Modal baseline")
        axes[0].set(xlabel="epoch", ylabel=r"$|t_{-1}|$",
                    title="Vanilla PINN: transmitted ±1 vs epoch")
        axes[0].legend(fontsize=8)
        axes[1].semilogy(ep42, [r["L_total"] for r in traj42], label="seed 42")
        if traj43:
            axes[1].semilogy(ep43, [r["L_total"] for r in traj43],
                             ls="--", label="seed 43")
        axes[1].set(xlabel="epoch", ylabel="Total loss", title="Training loss")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, out_dir / "training_trajectory.png")

    # ── Step 6: Summary table ──────────────────────────────────────────────
    t1_s42 = best42.get("t1_full", best42.get("t1", 0.0))
    t1_s43 = best43.get("t1_full", best43.get("t1", 0.0))
    rt_s42 = best42.get("RT", 0.0)
    rt_s43 = best43.get("RT", 0.0)

    # Check seed consistency
    seeds_consistent = abs(t1_s42 - t1_s43) / max(t1_s42, 1e-9) < 0.20

    summary = {
        "git_commit":          _git_commit(),
        "canonical_sha256":    _sha256(ref_path),
        "canonical_sha_ok":    _sha256(ref_path) == man["canonical_sha256"],
        "architecture":        "VanillaMLP (monolithic, no Fourier basis)",
        "n_params":            model42.n_params(),
        "n_params_modal":      1620,
        "n_layers":            N_LAYERS,
        "n_units":             N_UNITS,
        "activation":          "tanh",
        "epochs":              EPOCHS,
        "lr":                  LR,
        "optimizer":           "Adam + CosineAnnealingLR",
        "loss_type":           "PDE-only (no auxiliary ±1 loss)",
        "seed42_best_t1":      t1_s42,
        "seed42_best_epoch":   best42.get("epoch", 0),
        "seed42_RT":           rt_s42,
        "seed43_best_t1":      t1_s43,
        "seed43_best_epoch":   best43.get("epoch", 0),
        "seed43_RT":           rt_s43,
        "seeds_consistent":    seeds_consistent,
        "rcwa_target_t1":      RCWA_TARGET_T1,
        "improvement_over_modal_baseline":
            t1_s42 / max(BASELINE_MODAL_T1, 1e-10),
        # Obstruction metric
        "frac_pm1_at_init":    obs_result["frac_pm1"],
        "pm1_sigma_max":       obs_result["pm1_sigma_max"],
        "pm1_subspace_rank":   obs_result["pm1_subspace_rank"],
        "g_norm_at_init":      obs_result["g_norm"],
        "obstruction_verdict": obs_result["interpretation"],
        # Table 1 comparison (all rows)
        "table1": {
            "PDE-only (Adam) — modal":     {"t1": BASELINE_MODAL_T1, "RT": BASELINE_MODAL_RT},
            "Preconditioned gradient":      {"t1": PREC_GRAD_T1,      "RT": PREC_GRAD_RT},
            "±1 auxiliary loss — modal":   {"t1": AUX_LOSS_T1,       "RT": AUX_LOSS_RT},
            "PDE-only (Adam) — vanilla":   {"t1": round(t1_s42, 5),  "RT": round(rt_s42, 5)},
        },
    }

    (out_dir / "vanilla_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # CSV version
    table_rows = [
        {"experiment": k, "best_t_minus1": v["t1"], "R_plus_T": v["RT"]}
        for k, v in summary["table1"].items()
    ]
    with (out_dir / "vanilla_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment", "best_t_minus1", "R_plus_T"])
        writer.writeheader(); writer.writerows(table_rows)

    # ── Print Table 1 ──────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("TABLE 1 — Updated comparison", flush=True)
    print("="*60, flush=True)
    print(f"{'Experiment':<40s}  {'|t_-1|':>8s}  {'R+T':>7s}", flush=True)
    print("-"*60, flush=True)
    for k, v in summary["table1"].items():
        print(f"{k:<40s}  {v['t1']:>8.4f}  {v['RT']:>7.4f}", flush=True)
    print("-"*60, flush=True)
    print(f"\nRCWA target:  |t_{{-1}}| = {RCWA_TARGET_T1:.4f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("OBSTRUCTION METRIC SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"  Architecture:  VanillaMLP (monolithic, x+z inputs)", flush=True)
    print(f"  N_params:      {obs_result['n_params']} (modal: 1620)", flush=True)
    print(f"  g_norm:        {obs_result['g_norm']:.4f}", flush=True)
    print(f"  frac_pm1:      {obs_result['frac_pm1']:.4e}", flush=True)
    print(f"  σ_max(±1):     {obs_result['pm1_sigma_max']:.4f}", flush=True)
    print(f"  Verdict:       {obs_result['interpretation']}", flush=True)

    # List outputs
    pngs = list(out_dir.rglob("*.png"))
    jsons = list(out_dir.rglob("*.json"))
    print(f"\nOutputs: {len(pngs)} PNGs, {len(jsons)} JSONs in {out_dir.relative_to(ROOT)}",
          flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
