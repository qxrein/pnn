#!/usr/bin/env python3
"""Test three candidate escapes from the vanilla PINN local-minimum trap.

BACKGROUND
----------
Vanilla PINN (VanillaPINN, 4×32 tanh, joint (x,z) input) plateaus at
|t_{-1}| ≈ 0.001–0.004 regardless of:
  - Training length (600 or 5000 epochs)
  - Auxiliary ±1 loss (1.35× improvement only)
  - Fourier-feature embedding (worse)
  - Forcing ratio (ruled out as cause: ratio ≈1.07 everywhere)

The remaining hypothesis: training from zero-field initialisation converges to
a genuine local minimum / saddle point of the PDE+BC loss landscape near the
m=0-only solution that first-order optimisation cannot escape.

EXPERIMENT A — Non-trivial initialisation
  A1: Small random init of final head (0.01× and 0.1× Xavier scale) — breaks
      exact zero-field but keeps the scattered field tiny and x-dependent.
  A2: Warm-start proxy — pre-train for 50 steps on a synthetic target that has
      nonzero ±1 Fourier content, then switch to the real PDE loss.
      The synthetic target is E_syn(x,z) = ε · sin(G₀·x) · f(z) where
      G₀=2π/Λ (the ±1 Bloch wavenumber) and f(z) is a smooth envelope,
      and ε is chosen so ||E_syn|| ≈ 0.01 × ||E_bg||.

EXPERIMENT B — Reweighted interface / boundary losses
  Sweep w_int ∈ {2, 5, 10} × the current uniform weight on interface and
  vertical-edge continuity conditions.  PDE and DtN weights unchanged.
  Rationale: ±1 modes are generated at the ridge boundary; if the boundary
  condition is under-weighted relative to the smooth bulk PDE, the optimizer
  may satisfy bulk at the expense of ridge-discontinuity-driven structure.

EXPERIMENT C — L-BFGS second-order optimiser
  Apply L-BFGS (strong-Wolfe line search, max_iter=20, history_size=50) to
  the vanilla PINN, starting from the SAME zero-field initialisation used in
  all baselines.  Budget: 600 L-BFGS outer steps (each calls the closure up
  to max_iter=20 times internally), chosen to match wall-clock cost of the
  600-epoch Adam baseline.

BASELINES (for comparison):
  - Vanilla PDE-only (Adam, 600 ep):   |t_{-1}|=0.003, R+T=0.994
  - Vanilla + aux loss (Adam, 600 ep): |t_{-1}|=0.004, R+T=0.994
  - Modal + aux loss (Adam, 600 ep):   |t_{-1}|=0.019, R+T=1.005
  - RCWA target:                       |t_{-1}|=0.049

OUTPUT
------
outputs/vanilla_local_min/
    results_table.json / .csv
    exp_A1_randinit/     trajectory_{scale}_{seed}.csv
    exp_A2_warmstart/    trajectory_seed{42,43}.csv
    exp_B_weights/       trajectory_w{2,5,10}_seed{42,43}.csv
    exp_C_lbfgs/         trajectory_seed{42,43}.csv
    trajectories.png     — all meaningful trajectories on one axes
    conclusion.json
"""
from __future__ import annotations

import csv
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
    compute_background_coefficients,
    lbg_bottom_bc,
    lbg_top_bc,
    lbg_vertical_interface_loss,
    maxwell_2d_lbg_pde_residual,
)
from src.utils import set_seed
from src.vanilla_pinn import VanillaMLP, VanillaPINN, N_LAYERS, N_UNITS
from scripts.train_lbg import make_lambda_0p8
from scripts.run_frozen_head_least_squares import N_DTN_ORDERS, SEED
from scripts.run_vanilla_baseline import (
    sample_points,
    _modal_report,
    _energy_balance,
    _spatial_fourier_t,
    EPOCHS, LR, CONFIRM_SEED, LOG_EVERY,
    MODAL_ORDER_MAX, Z_BOT_FRAC,
    BASELINE_MODAL_T1, RCWA_TARGET_T1, AUX_LOSS_T1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Baselines (for reporting)
# ─────────────────────────────────────────────────────────────────────────────

VANILLA_PDE_T1    = 0.003
VANILLA_PDE_RT    = 0.994
VANILLA_AUX_T1    = 0.004
VANILLA_AUX_RT    = 0.994
MODAL_AUX_T1      = AUX_LOSS_T1    # 0.019
MODAL_AUX_RT      = 1.005
RT_MAX            = 1.05

# Meaningful-improvement threshold (must beat by this factor to count)
MEANINGFUL = 1.5 * VANILLA_PDE_T1  # 0.0045


# ─────────────────────────────────────────────────────────────────────────────
# Shared loss (all weights 1.0 — same as vanilla_loss in run_vanilla_baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(
    model: VanillaPINN,
    pts: dict,
    physics: PhysicsConfig,
    coeff: dict,
    w_int: float = 1.0,   # multiplier on interface + vertical-edge terms
) -> dict:
    """PDE + interface + DtN loss with configurable interface weight.

    w_int=1.0 is the unmodified vanilla baseline.
    w_int>1 amplifies horizontal interface (E/H continuity) and vertical
    ridge-edge continuity losses relative to the bulk PDE.
    DtN top/bottom weights are always 1.0 (radiation BCs unchanged).
    """
    zero = torch.zeros(1, dtype=torch.float64)
    p    = physics

    def _pde(key, net, eps_val):
        xk = pts.get(f"x_{key}"); zk = pts.get(f"z_{key}")
        if xk is None or not len(xk):
            return zero
        if key == "grat":
            eps_val = epsilon_r_fn(xk, zk, physics)
        res = maxwell_2d_lbg_pde_residual(net, xk, zk, physics, eps_val, coeff)
        return sum(torch.mean(r**2) for r in res) / len(res)

    La  = _pde("air",  model.net_air,  p.n_air**2)
    Lg  = _pde("grat", model.net_grat, p.n_ridge**2)
    Ls  = _pde("sub",  model.net_sub,  p.n_substrate**2)

    LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat,
                                              p.ridge_z_min, pts["x_int1"])
    LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,
                                              p.ridge_z_max, pts["x_int2"])
    LEv_l, LHv_l = lbg_vertical_interface_loss(model.net_grat, p.ridge_x_min,
                                                pts["z_vleft"])
    LEv_r, LHv_r = lbg_vertical_interface_loss(model.net_grat, p.ridge_x_max,
                                                pts["z_vright"])

    Lt = lbg_top_bc(model.net_air, pts["x_top"], physics,
                    use_dtn=True, n_dtn_orders=N_DTN_ORDERS)
    Lb = lbg_bottom_bc(model.net_sub, pts["x_bot"], physics, coeff,
                       use_dtn=True, n_dtn_orders=N_DTN_ORDERS)

    L_pde   = La + Lg + Ls
    L_int   = LE1 + LH1 + LE2 + LH2 + LEv_l + LHv_l + LEv_r + LHv_r
    L_dtn   = Lt + Lb

    total   = L_pde + w_int * L_int + L_dtn
    return {
        "pde_air":    La, "pde_grat": Lg, "pde_sub": Ls,
        "L_pde":      L_pde,
        "E_int":      LE1 + LE2, "H_int": LH1 + LH2,
        "E_vert":     LEv_l + LEv_r, "H_vert": LHv_l + LHv_r,
        "L_int":      L_int,
        "top_DtN":    Lt, "bottom_DtN": Lb,
        "L_dtn":      L_dtn,
        "total":      total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop (Adam, shared by A and B)
# ─────────────────────────────────────────────────────────────────────────────

def train_adam(
    model: VanillaPINN,
    pts: dict,
    physics: PhysicsConfig,
    coeff: dict,
    epochs: int,
    seed: int,
    label: str,
    w_int: float = 1.0,
    log_every: int = LOG_EVERY,
) -> tuple[list[dict], dict]:
    """Train with Adam + CosineAnnealing; return (trajectory, best_valid_dict)."""
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    traj: list[dict] = []
    best  = {"t1": 0.0, "epoch": 0, "state": None, "RT": 0.0}

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses_d = build_loss(model, pts, physics, coeff, w_int=w_int)
        losses_d["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % log_every == 0 or ep == 1:
            model.eval()
            modal_m   = _modal_report(model, physics)
            t1        = modal_m["t_minus1_abs"]
            RT_now, _ = _energy_balance(model, physics, coeff)

            row = {
                "epoch":     ep, "seed": seed, "label": label,
                "pde_grat":  float(losses_d["pde_grat"].detach()),
                "L_int":     float(losses_d["L_int"].detach()),
                "L_total":   float(losses_d["total"].detach()),
                "t_minus1":  t1,
                "t_plus1":   modal_m["t_plus1_abs"],
                "t_0":       modal_m["t_0_abs"],
                "R_plus_T":  RT_now,
            }
            traj.append(row)

            if RT_now < RT_MAX and t1 > best["t1"]:
                best.update({
                    "t1":    t1,
                    "epoch": ep,
                    "RT":    RT_now,
                    "state": {k: v.cpu().clone()
                               for k, v in model.state_dict().items()},
                })

            if ep % (log_every * 5) == 0 or ep == 1:
                print(f"    ep={ep:4d}  pde_g={float(losses_d['pde_grat'].detach()):.3e}"
                      f"  L_int={float(losses_d['L_int'].detach()):.3e}"
                      f"  |t_{{-1}}|={t1:.5f}  R+T={RT_now:.4f}", flush=True)

    print(f"  Best |t_{{-1}}|={best['t1']:.5f}  R+T={best['RT']:.4f}", flush=True)
    return traj, best


def _save_traj(traj: list[dict], path: Path) -> None:
    if not traj:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
        writer.writeheader(); writer.writerows(traj)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT A1 — Small random initialisation of final head
# ─────────────────────────────────────────────────────────────────────────────

def make_rand_init_model(physics: PhysicsConfig, scale: float) -> VanillaPINN:
    """VanillaPINN with final head initialised to scale × Xavier-normal.

    All hidden layers: Xavier-normal (unchanged from default).
    Final head: scale × Xavier-normal weight + zero bias.
      scale=0   → same as baseline (zero head)
      scale=0.01 → small random, E_scat ≈ 0 but x-dependent
      scale=0.1  → moderate random, |E_scat| ~ 10% of |E_bg|
    """
    model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    head  = model.backbone.net[-1]          # nn.Linear(32, 6)
    nn.init.xavier_normal_(head.weight)
    nn.init.zeros_(head.bias)
    head.weight.data.mul_(scale)
    return model


def run_exp_A1(physics: PhysicsConfig, coeff: dict, pts: dict,
               out_dir: Path) -> list[dict]:
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT A1: Small random final-head initialisation", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for scale in [0.01, 0.1]:
        for seed in [SEED, CONFIRM_SEED]:
            label = f"randinit_{scale:.2f}_s{seed}"
            print(f"\n  scale={scale}  seed={seed}", flush=True)
            set_seed(seed)
            model = make_rand_init_model(physics, scale)
            # Verify the output is NOT identically zero (unlike baseline)
            with torch.no_grad():
                x_t = torch.rand(32, dtype=torch.float64) * physics.period
                z_t = torch.rand(32, dtype=torch.float64) * physics.domain_height
                er, ei, *_ = model.backbone.field_components(x_t, z_t)
                max_E0 = float(max(er.abs().max(), ei.abs().max()))
            print(f"    max|E_scat| at init = {max_E0:.3e}  (baseline=0)", flush=True)

            traj, best = train_adam(model, pts, physics, coeff,
                                    epochs=EPOCHS, seed=seed, label=label)
            _save_traj(traj, out_dir / f"trajectory_{label}.csv")
            results.append({
                "experiment": "A1", "variant": f"scale={scale}",
                "seed": seed, "label": label,
                "max_E0_at_init": max_E0,
                "best_t1": best["t1"], "best_RT": best["RT"],
                "best_epoch": best["epoch"],
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT A2 — Warm-start proxy: pre-train on synthetic ±1 target
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_STEPS = 50     # short pre-training steps on synthetic target
WARMUP_LR    = 1e-3   # slightly higher LR for fast initial alignment


def _synthetic_target(
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
    coeff: dict,
    epsilon: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic scattered field with nonzero ±1 Fourier content.

    E_syn(x,z) = ε · sin(G₀·x) · f(z)

    where G₀ = 2π/Λ (the ±1 Bloch wavenumber), f(z) is a smooth z-envelope
    that peaks inside the grating band and decays to zero at the boundaries,
    and ε is scaled so ||E_syn|| ≈ epsilon × ||E_bg||.

    The magnetic field is derived analytically from Maxwell's equations
    in nondimensional coordinates (∂E/∂z̄ = H_x_i, ∂E/∂x̄ = -H_z_i),
    treating H as purely anti-symmetric in x (consistent with the ±1 structure).

    Returns (Er_syn, Ei_syn) — scattered E_y real/imag parts.
    No RCWA is used; this is a purely synthetic drive, not a ground truth.
    """
    p   = physics
    G0  = 2.0 * np.pi / p.period
    z0  = (p.ridge_z_min + p.ridge_z_max) / 2.0   # ridge centre in z
    sig = p.ridge_height                            # width of Gaussian envelope

    # Smooth z-envelope centred on the ridge
    f_z  = torch.exp(-0.5 * ((z - z0) / sig) ** 2)

    # Estimate ||E_bg|| at collocation points for scaling
    from src.maxwell_layered_bg import background_field_torch
    Ebg_r, Ebg_i, _, _ = background_field_torch(z.detach(), coeff, physics)
    E_bg_rms = float(torch.sqrt(torch.mean(Ebg_r**2 + Ebg_i**2)).detach()) + 1e-12

    amp  = epsilon * E_bg_rms

    # ±1 content: sin(G0 x) · f(z)  (purely real scattered field)
    Er_syn = amp * torch.sin(G0 * x) * f_z
    Ei_syn = torch.zeros_like(Er_syn)    # zero imaginary part for simplicity

    return Er_syn, Ei_syn


def run_exp_A2(physics: PhysicsConfig, coeff: dict, pts: dict,
               out_dir: Path) -> list[dict]:
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT A2: Warm-start on synthetic ±1 target", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in [SEED, CONFIRM_SEED]:
        label = f"warmstart_s{seed}"
        print(f"\n  seed={seed}", flush=True)
        set_seed(seed)
        model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)

        # ── Phase 1: warm-start on synthetic target (50 steps) ────────────
        print(f"  [warm-start] {WARMUP_STEPS} steps fitting E_syn = ε·sin(G₀x)·f(z)",
              flush=True)
        warmup_opt = torch.optim.Adam(model.parameters(), lr=WARMUP_LR)

        # Use all interior collocation points for warm-up
        x_all = torch.cat([pts["x_air"], pts["x_grat"], pts["x_sub"]])
        z_all = torch.cat([pts["z_air"], pts["z_grat"], pts["z_sub"]])

        for ws in range(WARMUP_STEPS):
            model.train()
            warmup_opt.zero_grad(set_to_none=True)
            Er_s, Ei_s, *_ = model.backbone.field_components(
                x_all.requires_grad_(True), z_all.requires_grad_(True))
            Er_syn, Ei_syn = _synthetic_target(x_all, z_all, physics, coeff)
            loss_warm = torch.mean((Er_s - Er_syn)**2 + (Ei_s - Ei_syn)**2)
            loss_warm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            warmup_opt.step()

        with torch.no_grad():
            Er_ws, Ei_ws, *_ = model.backbone.field_components(
                x_all[:32], z_all[:32])
            max_E_warmup = float(max(Er_ws.abs().max(), Ei_ws.abs().max()))
        print(f"  After warm-start: max|E_scat|={max_E_warmup:.3e}"
              f"  loss_warm={float(loss_warm.detach()):.3e}", flush=True)

        # ── Phase 2: full PDE training (600 epochs) ──────────────────────
        print(f"  [PDE training] {EPOCHS} epochs with Adam LR={LR}", flush=True)
        traj, best = train_adam(model, pts, physics, coeff,
                                epochs=EPOCHS, seed=seed, label=label)
        _save_traj(traj, out_dir / f"trajectory_{label}.csv")
        results.append({
            "experiment": "A2", "variant": "warm_start",
            "seed": seed, "label": label,
            "warmup_steps": WARMUP_STEPS,
            "max_E_after_warmup": max_E_warmup,
            "best_t1": best["t1"], "best_RT": best["RT"],
            "best_epoch": best["epoch"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT B — Reweighted interface losses
# ─────────────────────────────────────────────────────────────────────────────

W_INT_SWEEP = [2.0, 5.0, 10.0]


def run_exp_B(physics: PhysicsConfig, coeff: dict, pts: dict,
              out_dir: Path) -> list[dict]:
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT B: Reweighted interface / ridge-edge loss", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for w_int in W_INT_SWEEP:
        for seed in [SEED, CONFIRM_SEED]:
            label = f"wint{w_int:.0f}_s{seed}"
            print(f"\n  w_int={w_int:.0f}  seed={seed}", flush=True)
            set_seed(seed)
            model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
            traj, best = train_adam(model, pts, physics, coeff,
                                    epochs=EPOCHS, seed=seed, label=label,
                                    w_int=w_int)
            _save_traj(traj, out_dir / f"trajectory_{label}.csv")
            results.append({
                "experiment": "B", "variant": f"w_int={w_int:.0f}",
                "w_int": w_int, "seed": seed, "label": label,
                "best_t1": best["t1"], "best_RT": best["RT"],
                "best_epoch": best["epoch"],
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT C — L-BFGS optimiser
# ─────────────────────────────────────────────────────────────────────────────

LBFGS_OUTER   = 300    # outer steps (each allows up to max_iter=20 CG iters)
LBFGS_MAXITER = 20     # max inner line-search iterations per outer step
LBFGS_HISTORY = 50     # L-BFGS memory (pairs)
LBFGS_LR      = 0.1    # initial step size (same as src/train.py)
LBFGS_LOG     = 20     # log every N outer steps


def run_exp_C(physics: PhysicsConfig, coeff: dict, pts: dict,
              out_dir: Path) -> list[dict]:
    """L-BFGS optimiser from zero-field initialisation.

    300 outer steps, each with up to 20 strong-Wolfe line-search evaluations.
    Total function evaluations ≤ 300 × 20 = 6000, comparable to 600 Adam
    epochs (each of which costs one forward+backward pass).

    L-BFGS uses a closure (re-evaluates loss on each inner call) with the
    same fixed collocation points as Adam — same batch throughout.
    """
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT C: L-BFGS optimiser (300 outer steps)", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in [SEED, CONFIRM_SEED]:
        label = f"lbfgs_s{seed}"
        print(f"\n  seed={seed}  outer_steps={LBFGS_OUTER}", flush=True)
        set_seed(seed)
        model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)

        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=LBFGS_LR,
            max_iter=LBFGS_MAXITER,
            history_size=LBFGS_HISTORY,
            line_search_fn="strong_wolfe",
        )

        traj: list[dict] = []
        best  = {"t1": 0.0, "epoch": 0, "state": None, "RT": 0.0}
        step_count = [0]    # mutable counter for closure

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            losses_d = build_loss(model, pts, physics, coeff, w_int=1.0)
            losses_d["total"].backward()
            step_count[0] += 1
            return losses_d["total"]

        for outer in range(1, LBFGS_OUTER + 1):
            model.train()
            opt.step(closure)

            if outer % LBFGS_LOG == 0 or outer == 1:
                model.eval()
                losses_eval = build_loss(model, pts, physics, coeff, w_int=1.0)
                modal_m   = _modal_report(model, physics)
                t1        = modal_m["t_minus1_abs"]
                RT_now, _ = _energy_balance(model, physics, coeff)

                row = {
                    "epoch":     outer, "seed": seed, "label": label,
                    "pde_grat":  float(losses_eval["pde_grat"].detach()),
                    "L_int":     float(losses_eval["L_int"].detach()),
                    "L_total":   float(losses_eval["total"].detach()),
                    "t_minus1":  t1,
                    "t_plus1":   modal_m["t_plus1_abs"],
                    "t_0":       modal_m["t_0_abs"],
                    "R_plus_T":  RT_now,
                    "n_closures": step_count[0],
                }
                traj.append(row)

                if RT_now < RT_MAX and t1 > best["t1"]:
                    best.update({"t1": t1, "epoch": outer, "RT": RT_now,
                                 "state": {k: v.cpu().clone()
                                            for k, v in model.state_dict().items()}})

                print(f"    outer={outer:3d}  closures={step_count[0]:5d}"
                      f"  L={float(losses_eval['total'].detach()):.3e}"
                      f"  |t_{{-1}}|={t1:.5f}  R+T={RT_now:.4f}", flush=True)

        _save_traj(traj, out_dir / f"trajectory_{label}.csv")
        print(f"  Best |t_{{-1}}|={best['t1']:.5f}  R+T={best['RT']:.4f}"
              f"  total closures={step_count[0]}", flush=True)
        results.append({
            "experiment": "C", "variant": "L-BFGS",
            "seed": seed, "label": label,
            "lbfgs_outer_steps": LBFGS_OUTER,
            "lbfgs_total_closures": step_count[0],
            "best_t1": best["t1"], "best_RT": best["RT"],
            "best_epoch": best["epoch"],
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results table + trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "baseline":    ("#666666", ":"),
    "A1_0.01":     ("#4dac26", "-"),
    "A1_0.10":     ("#1a7837", "--"),
    "A2":          ("#d6604d", "-"),
    "B_w2":        ("#f4a582", "-"),
    "B_w5":        ("#d6604d", "--"),
    "B_w10":       ("#8e0152", "-."),
    "C_lbfgs":     ("#2166ac", "-"),
    "modal_aux":   ("#000000", ":"),
}


def make_trajectory_plot(all_trajs: dict[str, list[dict]], out_path: Path) -> None:
    """All |t_{-1}| trajectories on one set of axes for direct comparison."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Baselines (horizontal lines)
    ax.axhline(RCWA_TARGET_T1, color="black", lw=1.2, ls=":",
               label=f"RCWA target {RCWA_TARGET_T1:.3f}")
    ax.axhline(MODAL_AUX_T1, color="#000000", lw=0.9, ls="--",
               label=f"Modal+aux best {MODAL_AUX_T1:.3f}")
    ax.axhline(VANILLA_PDE_T1, color="#888888", lw=0.9, ls=":",
               label=f"Vanilla PDE-only {VANILLA_PDE_T1:.3f}")

    # Trajectories
    color_cycle = [
        "#4dac26", "#1a7837",   # A1 scales
        "#d6604d",               # A2
        "#f4a582", "#d6604d", "#8e0152",   # B weights
        "#2166ac",               # C
    ]
    lscycle = ["-", "--", "-.", ":", "-", "--", "-."]
    for i, (key, traj) in enumerate(all_trajs.items()):
        if not traj:
            continue
        col = color_cycle[i % len(color_cycle)]
        ls  = lscycle[i % len(lscycle)]
        ep  = [r["epoch"] for r in traj]
        t1  = [r["t_minus1"] for r in traj]
        ax.plot(ep, t1, color=col, ls=ls, lw=1.6, label=key)

    ax.set(xlabel="Epoch / L-BFGS outer step",
           ylabel=r"$|t_{-1}|$",
           title="Vanilla PINN local-minimum escape attempts\n"
                 "All experiments vs RCWA target and baselines")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, out_path)


def print_results_table(rows: list[dict]) -> None:
    print(f"\n{'='*70}", flush=True)
    print("RESULTS TABLE", flush=True)
    print(f"{'='*70}", flush=True)
    header = f"{'Experiment':<32} {'seed':>4} {'best |t_-1|':>11} {'R+T':>7} {'escapes?':>9}"
    print(header, flush=True)
    print("-"*70, flush=True)

    # Baselines first
    for name, t1, rt in [
        ("Vanilla PDE-only (Adam 600 ep)", VANILLA_PDE_T1, VANILLA_PDE_RT),
        ("Vanilla + aux (Adam 600 ep)",    VANILLA_AUX_T1, VANILLA_AUX_RT),
        ("Modal + aux (Adam 600 ep)",      MODAL_AUX_T1,   MODAL_AUX_RT),
    ]:
        esc = "✓" if t1 >= MEANINGFUL else "—"
        print(f"  {name:<30} {'—':>4} {t1:>11.5f} {rt:>7.4f} {esc:>9}", flush=True)
    print("-"*70, flush=True)

    for r in rows:
        label   = f"{r['experiment']} {r.get('variant', '')} s{r['seed']}"
        t1      = r["best_t1"]
        rt      = r.get("best_RT", 0.0)
        escapes = "YES" if t1 >= MEANINGFUL else "no"
        print(f"  {label:<32} {r['seed']:>4} {t1:>11.5f} {rt:>7.4f}"
              f" {escapes:>9}", flush=True)
    print("-"*70, flush=True)
    print(f"  RCWA target:                              0.04900", flush=True)
    print(f"  Meaningful threshold (1.5× baseline):     {MEANINGFUL:.5f}", flush=True)
    print(f"{'='*70}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import shutil
    out_dir = ROOT / "outputs/vanilla_local_min"
    if out_dir.exists():
        print(f"Removing existing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cfg     = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    coeff   = compute_background_coefficients(physics)
    pts     = sample_points(physics, seed=SEED)

    print(f"Physics: Λ={physics.period:.3f}λ  dc={physics.ridge_width/physics.period:.2f}"
          f"  n_ridge={physics.n_ridge}  k0={physics.k0:.4f}", flush=True)
    print(f"Baselines: vanilla_pde={VANILLA_PDE_T1}  vanilla_aux={VANILLA_AUX_T1}"
          f"  modal_aux={MODAL_AUX_T1}  RCWA={RCWA_TARGET_T1}", flush=True)
    print(f"Meaningful threshold: |t_{{-1}}| ≥ {MEANINGFUL:.4f} "
          f"(1.5× vanilla PDE-only)", flush=True)

    all_results: list[dict] = []
    all_trajs:   dict[str, list[dict]] = {}

    # ── Exp A1 ───────────────────────────────────────────────────────────────
    res_A1 = run_exp_A1(physics, coeff, pts, out_dir / "exp_A1_randinit")
    all_results.extend(res_A1)
    for r in res_A1:
        key = f"A1 {r['variant']} s{r['seed']}"
        csv_p = out_dir / "exp_A1_randinit" / f"trajectory_{r['label']}.csv"
        if csv_p.exists():
            rows_t = list(csv.DictReader(csv_p.open()))
            for row in rows_t:
                for k in row: 
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): pass
            all_trajs[key] = rows_t

    # ── Exp A2 ───────────────────────────────────────────────────────────────
    res_A2 = run_exp_A2(physics, coeff, pts, out_dir / "exp_A2_warmstart")
    all_results.extend(res_A2)
    for r in res_A2:
        key   = f"A2 warm-start s{r['seed']}"
        csv_p = out_dir / "exp_A2_warmstart" / f"trajectory_{r['label']}.csv"
        if csv_p.exists():
            rows_t = list(csv.DictReader(csv_p.open()))
            for row in rows_t:
                for k in row:
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): pass
            all_trajs[key] = rows_t

    # ── Exp B ────────────────────────────────────────────────────────────────
    res_B = run_exp_B(physics, coeff, pts, out_dir / "exp_B_weights")
    all_results.extend(res_B)
    for r in res_B:
        key   = f"B {r['variant']} s{r['seed']}"
        csv_p = out_dir / "exp_B_weights" / f"trajectory_{r['label']}.csv"
        if csv_p.exists():
            rows_t = list(csv.DictReader(csv_p.open()))
            for row in rows_t:
                for k in row:
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): pass
            all_trajs[key] = rows_t

    # ── Exp C ────────────────────────────────────────────────────────────────
    res_C = run_exp_C(physics, coeff, pts, out_dir / "exp_C_lbfgs")
    all_results.extend(res_C)
    for r in res_C:
        key   = f"C L-BFGS s{r['seed']}"
        csv_p = out_dir / "exp_C_lbfgs" / f"trajectory_{r['label']}.csv"
        if csv_p.exists():
            rows_t = list(csv.DictReader(csv_p.open()))
            for row in rows_t:
                for k in row:
                    try: row[k] = float(row[k])
                    except (ValueError, TypeError): pass
            all_trajs[key] = rows_t

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(all_results)

    # ── Trajectory plot ───────────────────────────────────────────────────────
    # Only include seed-42 trajectories to avoid clutter
    trajs_42 = {k: v for k, v in all_trajs.items() if "s42" in k or "42" in k}
    make_trajectory_plot(trajs_42, out_dir / "trajectories_seed42.png")
    # All seeds
    make_trajectory_plot(all_trajs, out_dir / "trajectories_all.png")

    # ── Determine best overall ───────────────────────────────────────────────
    best_row = max(all_results, key=lambda r: r["best_t1"])
    any_escape = any(r["best_t1"] >= MEANINGFUL for r in all_results)
    escaping   = [r for r in all_results if r["best_t1"] >= MEANINGFUL]

    # ── Save JSON / CSV ───────────────────────────────────────────────────────
    (out_dir / "results_table.json").write_text(
        json.dumps(all_results, indent=2, default=float) + "\n")
    csv_keys = ["experiment", "variant", "seed", "label",
                "best_t1", "best_RT", "best_epoch"]
    with (out_dir / "results_table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_results)

    # ── Conclusion ────────────────────────────────────────────────────────────
    # Auto-generate verdict
    if not any_escape:
        verdict = (
            "NONE of the three approaches escape the plateau. "
            f"Best |t_{{-1}}|={best_row['best_t1']:.4f} "
            f"(exp {best_row['experiment']}, {best_row.get('variant','')}, "
            f"seed {best_row['seed']}), still below the meaningful threshold "
            f"({MEANINGFUL:.4f}). The vanilla PINN local-minimum remains "
            "unresolved and should be reported as an open problem."
        )
    else:
        top = sorted(escaping, key=lambda r: -r["best_t1"])[0]
        verdict = (
            f"Experiment {top['experiment']} ({top.get('variant','')}) "
            f"escapes the plateau: best |t_{{-1}}|={top['best_t1']:.4f} "
            f"(seed {top['seed']}), R+T={top.get('best_RT',0):.4f}. "
            f"This supports '{top['experiment']}' as a working fix. "
            f"Best overall: {top['label']}."
        )

    conclusion = {
        "git_commit":          _git_commit(),
        "any_escape":          any_escape,
        "meaningful_threshold": MEANINGFUL,
        "best_label":          best_row["label"],
        "best_t1":             best_row["best_t1"],
        "best_RT":             best_row.get("best_RT", 0.0),
        "escaping_experiments": escaping,
        "verdict":             verdict,
        "baselines": {
            "vanilla_pde_only_600ep": VANILLA_PDE_T1,
            "vanilla_aux_loss_600ep": VANILLA_AUX_T1,
            "modal_aux_loss_600ep":   MODAL_AUX_T1,
            "rcwa_target":            RCWA_TARGET_T1,
        },
        "all_results": all_results,
    }
    (out_dir / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, default=float) + "\n")

    print(f"\nVERDICT: {verdict}", flush=True)
    pngs = list(out_dir.rglob("*.png"))
    jsons = list(out_dir.rglob("*.json"))
    print(f"\nOutputs: {len(pngs)} PNGs, {len(jsons)} JSONs → {out_dir.relative_to(ROOT)}",
          flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
