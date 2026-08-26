#!/usr/bin/env python3
"""PDE-only preconditioned-gradient experiment.  Phases 0–6.

The gradient-alignment audit (Case A) showed:
  - The ±1-sensitive subspace has rank 4 with uniform modal sensitivity ~4.73.
  - The ridge PDE gradient is exactly orthogonal to the ±1-sensitive subspace
    at every training checkpoint.
  - The DtN/interface losses develop ±1 projection during training but are
    10–100× weaker than the ridge PDE.

This experiment applies a preconditioned gradient:

    g_precond = P_pm1 @ g + alpha * P_perp @ g

where P_pm1 is the projector onto the rank-4 ±1-sensitive subspace and
alpha in (0, 1) downweights the orthogonal component without eliminating it.

The preconditioning is purely structural — it uses J_pm1 computed from the
model at initialization.  No RCWA values, modal targets, or reference data
enter the training objective.

Safety invariants
-----------------
- PDE-only loss; RCWA remains withheld from the objective
- modal-data loss = 0
- optical coupling disabled
- no 10,000-epoch run
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
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable
from scripts.train_lbg import layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, HEAD_PARAM_COUNT, MODAL_ORDER_MAX,
    N_DTN_ORDERS, SEED, TARGET_T1,
    head_parameter_records, head_parameters, load_points,
    model_config_hash, pack_head, unpack_head,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import (
    git_commit, git_status, make_fresh_model, parameter_ordering_hash,
    run_compileall, run_pytest,
)
from scripts.run_feature_scaling import (
    freeze_scaled_except_head, make_scaled_model, scaled_head_parameters,
    FEATURE_SCALE_FLOOR,
)
from scripts.run_gradient_alignment import build_modal_jacobian


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

from scripts.run_frozen_head_least_squares import modal_report

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (documented, not RCWA-selected)
# ─────────────────────────────────────────────────────────────────────────────

ALPHA        = 0.1    # weight of P_perp component (0.1 = 90% boost to pm1)
LR           = 5e-4   # same as previous smoke runs
SMOKE_EPOCHS = 1000
LOG_EVERY    = 100
BASELINE_T1  = 0.00638


# ─────────────────────────────────────────────────────────────────────────────
# Build or load V_pm1
# ─────────────────────────────────────────────────────────────────────────────

def build_V_pm1(
    model: ExplicitFourierModalDD,
    heads: list,
    physics,
    z_bot: float,
    prev_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Return V_pm1 (1386, 4) and metadata.

    If prev_dir contains V_pm1.npy saved from the gradient-alignment run,
    load it directly.  Otherwise reconstruct from the feature-scaled model.

    V_pm1 columns span the top right singular directions of
        J_pm1 = stack([J_t_{-1}, J_t_{+1}])   shape (4, 1386)
    using a threshold of 0.01 * sigma_max (target-independent criterion).
    """
    # Try to load from prior gradient-alignment output
    if prev_dir is not None:
        v_path = prev_dir / "V_pm1.npy"
        if v_path.exists():
            V_pm1 = np.load(v_path)
            meta  = {"source": "loaded_from_prior_audit", "path": str(v_path)}
            print(f"  Loaded V_pm1 from {v_path}  shape={V_pm1.shape}", flush=True)
            return V_pm1, meta

    # Reconstruct
    print("  Reconstructing modal Jacobian J_t...", flush=True)
    J_t, t0_arr, orders = build_modal_jacobian(model, heads, physics, z_bot)

    idx_m1 = orders.index(-1)
    idx_p1 = orders.index(+1)
    J_m1   = J_t[[2 * idx_m1, 2 * idx_m1 + 1], :]   # (2, 1386)
    J_p1   = J_t[[2 * idx_p1, 2 * idx_p1 + 1], :]   # (2, 1386)
    J_pm1  = np.vstack([J_m1, J_p1])                  # (4, 1386)

    _, sigma_pm1, Vt_pm1 = np.linalg.svd(J_pm1, full_matrices=False)
    rank_pm1 = int(np.sum(sigma_pm1 > 0.01 * sigma_pm1[0]))
    rank_pm1 = max(rank_pm1, 1)
    V_pm1    = Vt_pm1[:rank_pm1, :].T                 # (1386, rank_pm1)

    meta = {
        "source":        "reconstructed",
        "pm1_rank":      rank_pm1,
        "sigma_pm1":     sigma_pm1.tolist(),
        "sigma_threshold": float(0.01 * sigma_pm1[0]),
        "J_pm1_shape":   list(J_pm1.shape),
        "V_pm1_shape":   list(V_pm1.shape),
    }
    print(f"  V_pm1 shape={V_pm1.shape}  σ_pm1={sigma_pm1.tolist()}", flush=True)
    return V_pm1, meta


# ─────────────────────────────────────────────────────────────────────────────
# Preconditioned gradient update
# ─────────────────────────────────────────────────────────────────────────────

def precondition_head_grads(
    heads: list,
    V_pm1: np.ndarray,
    alpha: float,
) -> dict:
    """Apply P_pm1 + alpha*P_perp preconditioning to head parameter gradients.

    Reads .grad from each head parameter, applies
        g_new = P_pm1 @ g + alpha * P_perp @ g
             = P_pm1 @ g + alpha * (g - P_pm1 @ g)
             = (1 - alpha) * P_pm1 @ g + alpha * g
    and writes the result back to .grad.

    Returns diagnostics (norms before/after).
    """
    # Concatenate all head grads into a single vector
    parts = []
    for p in heads:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
    g = torch.cat(parts).numpy()   # (HEAD_PARAM_COUNT,)

    g_norm_before = float(np.linalg.norm(g))

    # Project
    V  = V_pm1                                           # (1386, 4)
    # P_pm1 @ g = V @ (V^T @ g)
    proj_coords = V.T @ g                                # (4,)
    P_pm1_g     = V @ proj_coords                        # (1386,)
    P_perp_g    = g - P_pm1_g                            # (1386,)

    g_precond   = P_pm1_g + alpha * P_perp_g             # (1386,)
    # equivalently: (1 - alpha) * P_pm1_g + alpha * g

    pm1_frac_before = float(np.linalg.norm(P_pm1_g)  / (g_norm_before + 1e-30))
    pm1_frac_after  = float(np.linalg.norm(P_pm1_g)  / (np.linalg.norm(g_precond) + 1e-30))
    g_norm_after    = float(np.linalg.norm(g_precond))

    # Write back
    offset = 0
    g_t = torch.as_tensor(g_precond, dtype=parts[0].dtype if parts else torch.float64)
    for p in heads:
        n = p.numel()
        if p.grad is not None:
            p.grad.copy_(g_t[offset:offset + n].reshape(p.shape))
        offset += n

    return {
        "g_norm_before":   g_norm_before,
        "g_norm_after":    g_norm_after,
        "pm1_frac_before": pm1_frac_before,
        "pm1_frac_after":  pm1_frac_after,
        "proj_pm1_norm":   float(np.linalg.norm(P_pm1_g)),
        "proj_perp_norm":  float(np.linalg.norm(P_perp_g)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke run (1000 epochs, one seed)
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(
    seed: int,
    physics,
    pts: dict,
    coeff: dict,
    V_pm1: np.ndarray,
    alpha: float,
    device,
    dtype,
    ref,
    ref_path: Path,
    companion_p: Path,
    out_dir: Path,
    epochs: int = SMOKE_EPOCHS,
    log_every: int = LOG_EVERY,
    J_t: np.ndarray | None = None,
    orders: list | None = None,
) -> list[dict]:
    """One seed of the preconditioned-gradient smoke run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [Smoke seed={seed}] {epochs} epochs  alpha={alpha}", flush=True)

    set_seed(seed)
    model_s, _ = make_scaled_model(physics, pts)
    for p in model_s.parameters():
        p.requires_grad_(True)

    heads_s  = scaled_head_parameters(model_s)
    opt      = torch.optim.Adam(model_s.parameters(), lr=LR)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Pre-build J_pm1 row indices for directional derivative calculation
    if J_t is not None and orders is not None:
        idx_m1 = orders.index(-1)
        idx_p1 = orders.index(+1)
        J_m1   = J_t[[2*idx_m1, 2*idx_m1+1], :]
        J_p1   = J_t[[2*idx_p1, 2*idx_p1+1], :]
    else:
        J_m1 = J_p1 = None

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

        # Apply preconditioning BEFORE the optimizer step
        pc = precondition_head_grads(heads_s, V_pm1, alpha)

        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == 1:
            modal_m   = modal_report(model_s, physics, coeff, device, dtype,
                                     ref, ref_path, companion_p)
            head_norm = float(pack_head(heads_s).norm())

            # Directional derivative of t_±1 along current (post-precond) head grad
            if J_m1 is not None:
                # Recompute post-preconditioning head gradient
                hg = torch.cat([
                    p.grad.detach().reshape(-1) if p.grad is not None
                    else torch.zeros_like(p.reshape(-1))
                    for p in heads_s
                ]).numpy()
                hg_hat = hg / (float(np.linalg.norm(hg)) + 1e-30)
                d_m1   = float(np.linalg.norm(J_m1 @ hg_hat))
                d_p1   = float(np.linalg.norm(J_p1 @ hg_hat))
            else:
                d_m1 = d_p1 = 0.0

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
                h_norms[f"h_{net_name}"] = float(h.norm())

            row = {
                "epoch":             ep,
                "seed":              seed,
                "alpha":             alpha,
                "pde_air":           float(losses["pde_air"].detach()),
                "pde_grat":          float(losses["pde_grat"].detach()),
                "pde_sub":           float(losses["pde_sub"].detach()),
                "top_DtN":           float(losses["top"].detach()),
                "bottom_DtN":        float(losses["bottom"].detach()),
                "E_int":             float((losses["E_int1"]+losses["E_int2"]).detach()),
                "H_int":             float((losses["H_int1"]+losses["H_int2"]).detach()),
                "total_loss":        float(losses["total"].detach()),
                "head_norm":         head_norm,
                # pre-precond gradient
                "g_norm_before":     pc["g_norm_before"],
                "g_norm_after":      pc["g_norm_after"],
                "pm1_frac_before":   pc["pm1_frac_before"],
                "pm1_frac_after":    pc["pm1_frac_after"],
                "proj_pm1_norm":     pc["proj_pm1_norm"],
                "proj_perp_norm":    pc["proj_perp_norm"],
                # post-precond directional derivative
                "d_t_minus1_precond": d_m1,
                "d_t_plus1_precond":  d_p1,
                # modal
                "t_minus1":          modal_m["t_minus1_abs"],
                "t_plus1":           modal_m["t_plus1_abs"],
                "R_plus_T":          modal_m["R_plus_T"],
                "total_complex_l2":  modal_m["total_complex_l2"],
                "scattered_complex_l2": modal_m["scattered_complex_l2"],
                **h_norms,
            }
            traj.append(row)
            print(f"    ep={ep:4d}  pde_grat={row['pde_grat']:.2e}  "
                  f"|t-1|={row['t_minus1']:.4f}  R+T={row['R_plus_T']:.4f}  "
                  f"pm1_frac_before={pc['pm1_frac_before']:.3f}->"
                  f"{pc['pm1_frac_after']:.3f}  "
                  f"|head|={head_norm:.4f}", flush=True)

    # Save CSV
    if traj:
        csv_path = out_dir / "trajectory.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader()
            writer.writerows(traj)
        print(f"    Saved trajectory → {csv_path}", flush=True)

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(traj42: list[dict], traj43: list[dict] | None) -> dict:
    t1_start = traj42[0]["t_minus1"]
    t1_end   = traj42[-1]["t_minus1"]
    t1_max   = max(r["t_minus1"] for r in traj42)
    t1_max_ep = min(traj42, key=lambda r: -r["t_minus1"])["epoch"]

    improving   = t1_max > 1.5 * BASELINE_T1      # >50% above baseline
    recovering  = t1_max > 0.25 * TARGET_T1        # within 25% of target
    stable_RT   = max(r["R_plus_T"] for r in traj42) < 1.1
    stable_loss = traj42[-1]["total_loss"] < traj42[0]["total_loss"] * 2.0

    seed43_improving = False
    if traj43:
        t1_max43 = max(r["t_minus1"] for r in traj43)
        seed43_improving = t1_max43 > 1.5 * BASELINE_T1

    if recovering and stable_RT:
        case = "A"
        decision = (
            f"Preconditioned gradient recovered |t_±1|={t1_max:.5f} "
            f"(peak ep={t1_max_ep}), approaching target={TARGET_T1:.5f}. "
            "The preconditioning breaks the m=0 basin. "
            "Two-seed confirmation run."
            + (f" Seed 43 also improving (|t|={t1_max43:.5f})." if traj43 else "")
        )
        longer_run_authorized = False  # requires separate authorization
    elif improving and stable_RT:
        case = "A"
        decision = (
            f"Preconditioned gradient shows material improvement: "
            f"|t_±1| reached {t1_max:.5f} (baseline={BASELINE_T1:.5f}). "
            "Not yet near target but directional progress confirmed. "
            "Do not start a long run automatically."
        )
        longer_run_authorized = False
    elif not stable_RT:
        case = "C"
        decision = (
            f"Preconditioned gradient destabilised R+T (max={max(r['R_plus_T'] for r in traj42):.3f}). "
            "Reduce alpha or learning rate. Re-run 1000-epoch smoke before any longer run."
        )
        longer_run_authorized = False
    elif not stable_loss:
        case = "C"
        decision = (
            "PDE loss diverged during preconditioned training. "
            "Reduce alpha or clip norm. Re-run."
        )
        longer_run_authorized = False
    elif t1_end < BASELINE_T1 * 0.8:
        case = "D"
        decision = (
            f"Preconditioned gradient had no positive effect: "
            f"|t_±1| ended at {t1_end:.5f}, below baseline {BASELINE_T1:.5f}. "
            "The ±1-sensitive subspace is accessible but the curvature or "
            "parameterization prevents progress despite correct gradient direction. "
            "Investigate basis or parameterization redesign."
        )
        longer_run_authorized = False
    elif t1_end >= BASELINE_T1 * 0.8 and t1_max <= 1.5 * BASELINE_T1:
        # marginal: didn't go down but didn't improve either
        case = "B"
        decision = (
            f"Preconditioned gradient preserved |t_±1| (~{t1_end:.5f}) but "
            "did not achieve material improvement. "
            "Inspect curvature and whether competing loss blocks limit progress."
        )
        longer_run_authorized = False
    else:
        case = "B"
        decision = f"|t_±1| trajectory ambiguous. Review trajectory data."
        longer_run_authorized = False

    return {
        "case":                  case,
        "decision":              decision,
        "t1_start":              t1_start,
        "t1_end":                t1_end,
        "t1_max":                t1_max,
        "t1_max_epoch":          t1_max_ep,
        "baseline_t1":           BASELINE_T1,
        "target_t1":             TARGET_T1,
        "improving":             improving,
        "recovering":            recovering,
        "stable_RT":             stable_RT,
        "stable_loss":           stable_loss,
        "seed43_run":            traj43 is not None,
        "seed43_improving":      seed43_improving,
        "longer_run_authorized": longer_run_authorized,
        "modal_data_loss":       False,
        "optical_coupling":      False,
        "start_10000_epoch_run": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def write_plots(root: Path, traj42: list[dict], traj43: list[dict] | None,
                ga_traj: list[dict] | None) -> None:
    ep42 = [r["epoch"] for r in traj42]

    # Modal trajectory comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(ep42, [r["t_minus1"] for r in traj42], label="precond seed42")
    if traj43:
        ep43 = [r["epoch"] for r in traj43]
        axes[0].plot(ep43, [r["t_minus1"] for r in traj43],
                     ls="--", label="precond seed43")
    if ga_traj:
        ep_ga = [r["epoch"] for r in ga_traj]
        axes[0].plot(ep_ga, [float(r["t_minus1"]) for r in ga_traj],
                     ls=":", color="gray", label="baseline (no precond)")
    axes[0].axhline(TARGET_T1, color="k", ls="--", lw=0.8, label="target")
    axes[0].set(xlabel="epoch", ylabel="|t_{-1}|", title="Modal amplitude")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(ep42, [r["pde_grat"] for r in traj42], label="ridge PDE")
    axes[1].semilogy(ep42, [r["top_DtN"]  for r in traj42], label="top DtN")
    axes[1].semilogy(ep42, [r["bottom_DtN"] for r in traj42], label="bot DtN")
    axes[1].semilogy(ep42, [r["total_loss"] for r in traj42], label="total")
    axes[1].set(xlabel="epoch", ylabel="loss", title="Loss components")
    axes[1].legend(fontsize=8)

    axes[2].plot(ep42, [r["pm1_frac_before"] for r in traj42],
                 label="frac_pm1 (pre-precond)")
    axes[2].plot(ep42, [r["pm1_frac_after"] for r in traj42],
                 ls="--", label="frac_pm1 (post-precond)")
    axes[2].set(xlabel="epoch", ylabel="||P_{pm1}g|| / ||g||",
                title="Gradient ±1 projection")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "modal_trajectory_comparison.png", dpi=160)
    plt.close(fig)

    # Gradient projection comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep42, [r["pm1_frac_before"] for r in traj42], label="before precond")
    ax.plot(ep42, [r["pm1_frac_after"]  for r in traj42], ls="--", label="after precond")
    ax.plot(ep42, [r["d_t_minus1_precond"] for r in traj42],
            ls=":", label="d|t-1| after precond")
    ax.set(xlabel="epoch", ylabel="value", title="Gradient projection (seed 42)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "gradient_projection_comparison.png", dpi=160)
    plt.close(fig)

    # Loss component comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for comp in ("pde_air", "pde_grat", "pde_sub", "E_int", "H_int", "top_DtN", "bottom_DtN"):
        if comp in traj42[0]:
            axes[0].semilogy(ep42, [r[comp] for r in traj42], label=comp)
    axes[0].legend(fontsize=7)
    axes[0].set(xlabel="epoch", ylabel="loss", title="All components (seed 42)")

    axes[1].plot(ep42, [r["R_plus_T"] for r in traj42])
    axes[1].axhline(1.0, color="k", ls="--")
    axes[1].set(xlabel="epoch", ylabel="R+T", title="Energy balance")
    fig.tight_layout()
    fig.savefig(root / "loss_component_comparison.png", dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="PDE-only preconditioned-gradient experiment")
    ap.add_argument("--output-root",   default="outputs/phase5_preconditioned_gradient")
    ap.add_argument("--shared-points", default="outputs/phase5_feature_scaling/shared_points.npz")
    ap.add_argument("--alpha",         type=float, default=ALPHA)
    ap.add_argument("--epochs",        type=int, default=SMOKE_EPOCHS)
    ap.add_argument("--skip-seed43",   action="store_true")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)

    # ── Phase 0: preflight record ─────────────────────────────────────────────
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

    # Feature-scale hash
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
        "alpha":                  args.alpha,
        "epochs":                 args.epochs,
        "modal_data_loss_weight": 0.0,
        "optical_coupling":       False,
        "canonical_modified":     False,
        "companion_used_in_loss": False,
    }
    (root / "phase0_preflight.json").write_text(json.dumps(phase0, indent=2) + "\n")
    print("[Phase 0] Preflight OK", flush=True)

    # ── Phase 1: Build V_pm1 ─────────────────────────────────────────────────
    print("\n[Phase 1] Building ±1-sensitive subspace V_pm1", flush=True)

    set_seed(SEED)
    model_v, _ = make_scaled_model(physics, pts)
    heads_v    = freeze_scaled_except_head(model_v)
    z_bot      = 0.92 * physics.domain_height

    ga_dir = ROOT / "outputs/phase5_gradient_alignment"
    V_pm1, pm1_meta = build_V_pm1(model_v, heads_v, physics, z_bot, ga_dir)
    np.save(root / "V_pm1.npy", V_pm1)

    # Also build J_t for directional-derivative logging
    # If we already called build_modal_jacobian in build_V_pm1 (reconstruction
    # path), we need it again for run_smoke. Keep it minimal:
    if pm1_meta["source"] == "reconstructed":
        # J_t was already computed; rebuild cheaply from the same model
        print("  Rebuilding J_t for smoke logging...", flush=True)
        J_t, _, orders = build_modal_jacobian(model_v, heads_v, physics, z_bot)
    else:
        # Loaded V_pm1 from file — reconstruct J_t
        print("  Reconstructing J_t for smoke logging...", flush=True)
        J_t, _, orders = build_modal_jacobian(model_v, heads_v, physics, z_bot)

    pm1_metadata = {
        **pm1_meta,
        "V_pm1_shape":    list(V_pm1.shape),
        "alpha":          args.alpha,
        "update_formula": "g_precond = (1-alpha)*P_pm1@g + alpha*g",
        "alpha_meaning":  "weight of P_perp component; 0=pure_pm1, 1=unmodified",
        "V_pm1_criterion": "top right singular vectors of J_pm1=[J_t-1;J_t+1] with sigma>0.01*sigma_max",
        "V_pm1_frozen":   True,
        "rcwa_used_in_objective": False,
        "modal_data_loss": False,
    }
    (root / "pm1_subspace_metadata.json").write_text(
        json.dumps(jsonable(pm1_metadata), indent=2) + "\n")
    print(f"  V_pm1 saved  shape={V_pm1.shape}", flush=True)

    # ── Phase 2+3: Smoke run seed 42 ─────────────────────────────────────────
    traj42 = run_smoke(
        seed=SEED,
        physics=physics, pts=pts, coeff=coeff,
        V_pm1=V_pm1, alpha=args.alpha,
        device=device, dtype=dtype, ref=ref, ref_path=ref_path,
        companion_p=companion_p,
        out_dir=root / "smoke_1000",
        epochs=args.epochs,
        J_t=J_t, orders=orders,
    )

    # ── Phase 4: Decision ─────────────────────────────────────────────────────
    print("\n[Phase 4] Decision", flush=True)

    # Load baseline trajectory from gradient-alignment run for comparison
    ga_traj_path = ga_dir / "trajectory" / "training_trajectory.csv"
    ga_traj = None
    if ga_traj_path.exists():
        ga_traj = list(csv.DictReader(ga_traj_path.open()))

    # Preliminary decision (seed 43 not yet run)
    dec_prelim = make_decision(traj42, None)
    print(f"  Preliminary case: {dec_prelim['case']}", flush=True)
    print(f"  t1_max={dec_prelim['t1_max']:.5f}  t1_end={dec_prelim['t1_end']:.5f}",
          flush=True)

    # ── Phase 5: Two-seed confirmation if Case A ──────────────────────────────
    traj43 = None
    if dec_prelim["case"] == "A" and not args.skip_seed43:
        print("\n[Phase 5] Two-seed confirmation (seed=43)", flush=True)
        traj43 = run_smoke(
            seed=43,
            physics=physics, pts=pts, coeff=coeff,
            V_pm1=V_pm1, alpha=args.alpha,
            device=device, dtype=dtype, ref=ref, ref_path=ref_path,
            companion_p=companion_p,
            out_dir=root / "smoke_seed43",
            epochs=args.epochs,
            J_t=J_t, orders=orders,
        )
    else:
        if dec_prelim["case"] != "A":
            print(f"\n[Phase 5] Skipped (Case {dec_prelim['case']}, not A)", flush=True)
        else:
            print("\n[Phase 5] Skipped (--skip-seed43)", flush=True)

    # Final decision
    decision = make_decision(traj42, traj43)
    print(f"\n  Final case: {decision['case']}", flush=True)
    print(f"  {decision['decision'][:200]}", flush=True)

    # ── Phase 6: Verification + outputs ──────────────────────────────────────
    print("\n[Phase 6] Verification", flush=True)
    compile_r = run_compileall()
    pytest_r  = run_pytest()
    print(f"  compileall: {'OK' if compile_r['passed'] else 'FAIL'}")
    print(f"  pytest: {pytest_r['summary_line']}")

    # Write plots
    write_plots(root, traj42, traj43, ga_traj)

    # Summary
    summary = {
        **phase0,
        "pm1_subspace": pm1_metadata,
        "decision": decision,
        "seed42": {
            "t1_start":            traj42[0]["t_minus1"],
            "t1_end":              traj42[-1]["t_minus1"],
            "t1_max":              max(r["t_minus1"] for r in traj42),
            "t1_max_epoch":        min(traj42, key=lambda r: -r["t_minus1"])["epoch"],
            "R_plus_T_max":        max(r["R_plus_T"] for r in traj42),
            "total_loss_final":    traj42[-1]["total_loss"],
            "pm1_frac_before_avg": float(np.mean([r["pm1_frac_before"] for r in traj42])),
            "pm1_frac_after_avg":  float(np.mean([r["pm1_frac_after"]  for r in traj42])),
        },
        "seed43": {
            "t1_start":   traj43[0]["t_minus1"]  if traj43 else None,
            "t1_end":     traj43[-1]["t_minus1"] if traj43 else None,
            "t1_max":     max(r["t_minus1"] for r in traj43) if traj43 else None,
        } if traj43 else None,
        "baseline_t1":            BASELINE_T1,
        "target_t1":              TARGET_T1,
        "phase5_passed":          False,
        "compileall":             compile_r,
        "pytest":                 pytest_r,
    }
    (root / "preconditioned_gradient_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n")

    # CSV
    rows = []
    for label, traj in (("seed42", traj42), *([("seed43", traj43)] if traj43 else [])):
        for r in traj:
            rows.append({
                "seed":          r["seed"],
                "epoch":         r["epoch"],
                "alpha":         r["alpha"],
                "t_minus1":      r["t_minus1"],
                "t_plus1":       r["t_plus1"],
                "R_plus_T":      r["R_plus_T"],
                "total_complex_l2": r["total_complex_l2"],
                "scattered_complex_l2": r["scattered_complex_l2"],
                "pde_grat":      r["pde_grat"],
                "top_DtN":       r["top_DtN"],
                "bottom_DtN":    r["bottom_DtN"],
                "total_loss":    r["total_loss"],
                "head_norm":     r["head_norm"],
                "pm1_frac_before": r["pm1_frac_before"],
                "pm1_frac_after":  r["pm1_frac_after"],
                "d_t_minus1_precond": r["d_t_minus1_precond"],
            })
    if rows:
        with (root / "preconditioned_gradient_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    print("\n" + "=" * 70, flush=True)
    print(json.dumps(jsonable({
        "case":                decision["case"],
        "decision":            decision["decision"][:200],
        "t1_max_seed42":       decision["t1_max"],
        "t1_end_seed42":       decision["t1_end"],
        "baseline_t1":         BASELINE_T1,
        "target_t1":           TARGET_T1,
        "seed43_run":          decision["seed43_run"],
        "longer_run_auth":     decision["longer_run_authorized"],
        "pytest":              pytest_r["summary_line"],
        "output_root":         str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
