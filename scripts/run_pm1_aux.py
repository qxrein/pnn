#!/usr/bin/env python3
"""Final bounded experiment: PDE + ±1-encouraging auxiliary loss.

Approach
--------
Add a small auxiliary term to the PDE loss that encourages the model to
build transmitted ±1 amplitude:

    L_total = L_PDE + w_pm1 * L_pm1

where

    L_pm1 = -(|t_{-1}|^2 + |t_{+1}|^2)

and t_m are computed via differentiable spatial Fourier quadrature at z_bot.
This is reference-free — it does not use RCWA amplitudes, only the model's
own predicted field at the boundary.

The auxiliary term is a soft encouragement, not a hard modal target.
w_pm1 is chosen so that L_pm1 is 1–5% of L_PDE at initialization.

Constraints
-----------
- PDE base loss unchanged
- No RCWA targets in objective
- Modal-data loss disabled (no c_refl / c_trans supervision)
- Optical coupling disabled
- ≤ 2000 total epochs

Protocol
--------
Phase 1: w_pm1 sweep (250 epochs each, 4 values) to find working range.
Phase 2: Best w_pm1 → 2000-epoch main run, seed 42.
Phase 3: If |t_±1| > 0.02 at any epoch, run seed 43 for 1000 epochs.

Decision: if |t_±1| > 0.02, proceed to field visualisation and paper
preparation as partial success.  Otherwise, record as negative result and
proceed to paper.
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
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable, metrics
from scripts.train_lbg import evaluate, layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, MODAL_ORDER_MAX, N_DTN_ORDERS, SEED, TARGET_T1,
    model_config_hash, sha256_file,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import git_commit, git_status, run_compileall, run_pytest
from scripts.run_feature_scaling import make_scaled_model, FEATURE_SCALE_FLOOR


# ── Patch spatial_fourier_t ──────────────────────────────────────────────────
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_T1 = 0.00638
SUCCESS_T1  = 0.020       # threshold for "modest growth"
SWEEP_W     = [0.25, 0.5, 1.0, 2.0]
SWEEP_EPOCHS = 250
MAIN_EPOCHS  = 2000
CONFIRM_EPOCHS = 1000
LR           = 5e-4
LOG_EVERY    = 100
Z_BOT_FRAC   = 0.92       # monitor plane fraction for L_pm1


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable L_pm1
# ─────────────────────────────────────────────────────────────────────────────

def l_pm1_loss(
    model: ExplicitFourierModalDD,
    physics,
    n_quad: int = 128,
) -> torch.Tensor:
    """Differentiable -( |t_{-1}|^2 + |t_{+1}|^2 ) auxiliary loss.

    Uses spatial Fourier quadrature over a uniform grid at z_bot.
    No RCWA values are used — only the model's predicted scattered field.
    The loss is NEGATIVE so minimizing it increases |t_±1|.

    L_pm1 = -(|t_{-1}|^2 + |t_{+1}|^2)

    where t_m = (1/N) sum_j E_scat(x_j, z_bot) * exp(-i m G0 x_j).
    """
    z_bot = Z_BOT_FRAC * physics.domain_height
    g0    = 2.0 * np.pi / physics.period

    x = torch.linspace(0.0, physics.period, n_quad + 1,
                       dtype=torch.float64)[:-1].requires_grad_(True)
    z = torch.full_like(x.detach(), z_bot).requires_grad_(True)

    er, ei, *_ = model.net_sub.field_components(x, z)  # (n_quad,) each

    pm1_sum = torch.zeros(1, dtype=torch.float64)
    for m in (-1, +1):
        gm        = float(m) * g0
        x_np      = x.detach().numpy()
        cos_m     = torch.as_tensor(np.cos(gm * x_np), dtype=torch.float64)
        sin_m     = torch.as_tensor(np.sin(gm * x_np), dtype=torch.float64)
        # t_m = (1/N) * sum [Er * cos(gm*x) + Ei * sin(gm*x)
        #                   + i*(Ei*cos(gm*x) - Er*sin(gm*x))]
        tm_r = torch.mean(er * cos_m + ei * sin_m)
        tm_i = torch.mean(ei * cos_m - er * sin_m)
        pm1_sum = pm1_sum + tm_r**2 + tm_i**2

    return -pm1_sum   # negative: minimizing this increases |t_±1|^2


def read_l_pm1_magnitude(model: ExplicitFourierModalDD, physics) -> float:
    """Current |t_{-1}|^2 + |t_{+1}|^2 from the model (uses enable_grad internally)."""
    with torch.enable_grad():
        return float(-l_pm1_loss(model, physics).detach())


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def train_run(
    seed: int,
    physics,
    pts: dict,
    coeff: dict,
    w_pm1: float,
    epochs: int,
    device,
    dtype,
    ref,
    ref_path: Path,
    companion_p: Path,
    out_dir: Path,
    label: str = "",
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{label} seed={seed}]  w_pm1={w_pm1:.3f}  epochs={epochs}",
          flush=True)

    set_seed(seed)
    model_s, _ = make_scaled_model(physics, pts)
    for p in model_s.parameters():
        p.requires_grad_(True)

    opt   = torch.optim.Adam(model_s.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Measure initial w_pm1 balance (use enable_grad since field_components needs it)
    with torch.enable_grad():
        l0_losses = layered_bg_loss(
            model_s, pts, physics, coeff,
            w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
            use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
            rcwa_amps=None, w_modal=0.0,
        )
        l0_pde = float(l0_losses["pde_grat"].detach())
    l0_pm1  = read_l_pm1_magnitude(model_s, physics)
    frac0   = w_pm1 * l0_pm1 / max(l0_pde, 1e-30)
    print(f"    init: pde_grat={l0_pde:.3e}  |t±1|^2={l0_pm1:.3e}  "
          f"pm1_frac={frac0:.3f}", flush=True)

    traj  = []
    best        = {"t1": 0.0, "epoch": 0, "state": None}      # raw best
    best_valid  = {"t1": 0.0, "epoch": 0, "state": None}      # best with R+T < 1.05
    RT_MAX_VALID = 1.05
    diverged     = False

    for ep in range(1, epochs + 1):
        model_s.train()
        opt.zero_grad(set_to_none=True)

        losses_d = layered_bg_loss(
            model_s, pts, physics, coeff,
            w_pde=1.0, w_E=1.0, w_H=1.0, w_top=1.0, w_bot=1.0,
            use_dtn=True, n_dtn_orders=N_DTN_ORDERS,
            rcwa_amps=None, w_modal=0.0,
        )
        L_pde = losses_d["total"]

        # Auxiliary ±1 term (differentiable, no RCWA)
        L_pm1_val = l_pm1_loss(model_s, physics)
        L_total   = L_pde + w_pm1 * L_pm1_val

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        opt.step()
        sched.step()

        # Track best (use enable_grad since field_components needs it)
        with torch.enable_grad():
            t1_now = read_l_pm1_magnitude(model_s, physics) ** 0.5
        if t1_now > best["t1"]:
            best["t1"]    = t1_now
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model_s.state_dict().items()}

        if ep % LOG_EVERY == 0 or ep == 1:
            modal_m = modal_report(model_s, physics, coeff, device, dtype,
                                   ref, ref_path, companion_p)
            row = {
                "epoch":       ep,
                "seed":        seed,
                "w_pm1":       w_pm1,
                "label":       label,
                "pde_air":     float(losses_d["pde_air"].detach()),
                "pde_grat":    float(losses_d["pde_grat"].detach()),
                "pde_sub":     float(losses_d["pde_sub"].detach()),
                "top_DtN":     float(losses_d["top"].detach()),
                "bottom_DtN":  float(losses_d["bottom"].detach()),
                "L_pde":       float(L_pde.detach()),
                "L_pm1":       float(-L_pm1_val.detach()),
                "L_total":     float(L_total.detach()),
                "pm1_frac":    float(-L_pm1_val.detach() * w_pm1 / max(float(L_pde.detach()), 1e-30)),
                "t_minus1":    modal_m["t_minus1_abs"],
                "t_plus1":     modal_m["t_plus1_abs"],
                "R_plus_T":    modal_m["R_plus_T"],
                "total_complex_l2": modal_m["total_complex_l2"],
                "scattered_complex_l2": modal_m["scattered_complex_l2"],
            }
            traj.append(row)
            # Track best physically-valid checkpoint (R+T < 1.05)
            if row["R_plus_T"] < RT_MAX_VALID and row["t_minus1"] > best_valid["t1"]:
                best_valid["t1"]    = row["t_minus1"]
                best_valid["epoch"] = ep
                best_valid["state"] = {k: v.detach().cpu().clone()
                                       for k, v in model_s.state_dict().items()}
            # Early stop if completely diverged
            if row["R_plus_T"] > 50.0 and ep > 100:
                print(f"    DIVERGED: R+T={row['R_plus_T']:.1f} at ep={ep}. Stopping.",
                      flush=True)
                diverged = True
                break
            print(f"    ep={ep:5d}  pde_g={row['pde_grat']:.2e}  "
                  f"L_pm1={row['L_pm1']:.2e}  frac={row['pm1_frac']:.3f}  "
                  f"|t-1|={row['t_minus1']:.4f}  R+T={row['R_plus_T']:.4f}",
                  flush=True)

    # Save best checkpoint (raw)
    ckpt_path = out_dir / "best_checkpoint.pt"
    if best["state"] is not None:
        torch.save({
            "state_dict": best["state"],
            "epoch": best["epoch"],
            "t1_abs": best["t1"],
            "w_pm1": w_pm1, "seed": seed,
        }, ckpt_path)
        print(f"    Best raw:   epoch={best['epoch']}  |t-1|={best['t1']:.5f}",
              flush=True)

    # Save best physically-valid checkpoint (R+T < 1.05)
    ckpt_valid_path = out_dir / "best_valid_checkpoint.pt"
    if best_valid["state"] is not None:
        torch.save({
            "state_dict": best_valid["state"],
            "epoch": best_valid["epoch"],
            "t1_abs": best_valid["t1"],
            "w_pm1": w_pm1, "seed": seed,
        }, ckpt_valid_path)
        print(f"    Best valid: epoch={best_valid['epoch']}  "
              f"|t-1|={best_valid['t1']:.5f}  (R+T < {RT_MAX_VALID})",
              flush=True)

    # Save CSV
    if traj:
        with (out_dir / "trajectory.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader(); writer.writerows(traj)

    return traj, best, best_valid


# ─────────────────────────────────────────────────────────────────────────────
# Field visualizations from best checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def make_field_plots(
    state_dict: dict,
    physics,
    coeff: dict,
    ref,
    device,
    dtype,
    out_dir: Path,
    label: str,
    ref_path: Path,
    companion_p: Path,
    pts: dict,            # collocation points needed for make_scaled_model
) -> None:
    """Generate |E_y|, phase(E_y), Poynting-flux, and modal comparison plots."""
    set_seed(SEED)
    model_p, _ = make_scaled_model(physics, pts)
    model_p.load_state_dict(state_dict)
    model_p.eval()

    # Evaluate on visualization grid
    fields = evaluate(model_p, physics, device, dtype, "layered_bg", coeff)
    x1d    = fields["x"][0]
    z1d    = fields["z"][:, 0]

    # Reconstruct total field
    from src.maxwell_layered_bg import background_field_np
    Ebg_r, Ebg_i, _, _ = background_field_np(z1d, coeff)
    E_total_r = fields["E_scat_real"] + Ebg_r[:, np.newaxis]
    E_total_i = fields["E_scat_imag"] + Ebg_i[:, np.newaxis]
    E_total   = E_total_r + 1j * E_total_i
    X, Z      = np.meshgrid(x1d, z1d)

    # RCWA reference total field
    rr, ri, _ = interpolate_reference_to_grid(ref, x1d, z1d)
    E_rcwa    = rr + 1j * ri

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 0: PINN
    im0 = axes[0, 0].pcolormesh(X, Z, np.abs(E_total), cmap="viridis", shading="auto")
    axes[0, 0].set(title=f"PINN |E_y| ({label})", xlabel="x / λ", ylabel="z / λ")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(X, Z, np.angle(E_total) * 180 / np.pi,
                                  cmap="hsv", shading="auto")
    axes[0, 1].set(title="PINN phase(E_y) [°]", xlabel="x / λ")
    plt.colorbar(im1, ax=axes[0, 1])

    # Poynting flux Sy = Re(E_y * conj(H_x)) / 2
    # H_x from model
    with torch.no_grad():
        xf = torch.as_tensor(X.ravel(), dtype=torch.float64)
        zf = torch.as_tensor(Z.ravel(), dtype=torch.float64)
        p  = physics
        Hx_scat = torch.zeros_like(xf)
        for mask, net in [
            (zf <= p.ridge_z_min, model_p.net_air),
            ((zf > p.ridge_z_min) & (zf <= p.ridge_z_max), model_p.net_grat),
            (zf > p.ridge_z_max, model_p.net_sub),
        ]:
            if mask.any():
                xm = xf[mask].requires_grad_(True)
                zm = zf[mask].requires_grad_(True)
                out = net.forward(xm, zm)
                Hx_scat[mask] = out[:, 2].detach()

    Hx_s  = Hx_scat.numpy().reshape(X.shape)
    Hbg_r, Hbg_i, *_ = background_field_np(z1d, coeff)
    Hx_total = Hx_s + Hbg_r[:, np.newaxis]
    Sy = 0.5 * np.real(E_total * np.conj(Hx_total + 1j * Hx_total))
    im2 = axes[0, 2].pcolormesh(X, Z, np.real(E_total * np.conj(Hx_total)),
                                  cmap="RdBu_r", shading="auto")
    axes[0, 2].set(title="PINN Poynting Sy", xlabel="x / λ")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 1: RCWA reference
    im3 = axes[1, 0].pcolormesh(X, Z, np.abs(E_rcwa), cmap="viridis", shading="auto")
    axes[1, 0].set(title="RCWA |E_y|", xlabel="x / λ", ylabel="z / λ")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(X, Z, np.angle(E_rcwa) * 180 / np.pi,
                                  cmap="hsv", shading="auto")
    axes[1, 1].set(title="RCWA phase(E_y) [°]", xlabel="x / λ")
    plt.colorbar(im4, ax=axes[1, 1])

    # Modal comparison bar chart
    modal_pinn = extract_modal_amplitudes(
        E_total, x1d, z1d, physics, n_orders=3,
        formulation="layered_bg", field_representation="total")
    companion  = np.load(companion_p, allow_pickle=False)
    n_harm     = int((len(companion["c_trans"]) - 1) // 2)
    z_bot_mon  = float(modal_pinn["z_bot_monitor"])
    orders     = [-3, -2, -1, 0, 1, 2, 3]
    t_pinn_abs = [abs(modal_pinn["t_m_complex"][3 + m]) for m in orders]
    t_rcwa_abs = []
    for m in orders:
        idx   = n_harm + m
        kz_m  = complex(companion["kz_substrate"][idx])
        t_raw = complex(companion["c_trans"][idx])
        t_at  = t_raw * np.exp(-1j * kz_m * (z_bot_mon - physics.ridge_z_max))
        t_rcwa_abs.append(abs(t_at))
    x_pos = np.arange(len(orders))
    axes[1, 2].bar(x_pos - 0.2, t_rcwa_abs, 0.4, label="RCWA")
    axes[1, 2].bar(x_pos + 0.2, t_pinn_abs, 0.4, label="PINN")
    axes[1, 2].set_xticks(x_pos, [str(m) for m in orders])
    axes[1, 2].set(xlabel="m", ylabel="|t_m|", title="Modal amplitudes")
    axes[1, 2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "field_visualization.png", dpi=160)
    plt.close(fig)
    print(f"    Saved field visualisation → {out_dir / 'field_visualization.png'}",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_plots(root: Path, all_trajs: list[tuple[str, list[dict]]],
                        baseline_path: Path | None) -> None:
    # Load baseline
    baseline = []
    if baseline_path and baseline_path.exists():
        baseline = list(csv.DictReader(baseline_path.open()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for label, traj in all_trajs:
        ep  = [r["epoch"] for r in traj]
        t1  = [r["t_minus1"] for r in traj]
        axes[0].plot(ep, t1, label=label)
    if baseline:
        ep_b = [int(float(r["epoch"])) for r in baseline]
        t_b  = [float(r["t_minus1"]) for r in baseline]
        axes[0].plot(ep_b, t_b, "k:", lw=0.8, label="baseline")
    axes[0].axhline(TARGET_T1, color="k", ls="--", lw=0.8, label="target")
    axes[0].axhline(SUCCESS_T1, color="orange", ls=":", lw=0.8, label="success")
    axes[0].axhline(BASELINE_T1, color="r", ls=":", lw=0.8, label="init")
    axes[0].set(xlabel="epoch", ylabel="|t_{-1}|", title="|t_-1| trajectory")
    axes[0].legend(fontsize=7)

    for label, traj in all_trajs:
        ep = [r["epoch"] for r in traj]
        axes[1].semilogy(ep, [r["L_pde"] for r in traj], label=label)
    axes[1].set(xlabel="epoch", ylabel="L_PDE", title="PDE loss")
    axes[1].legend(fontsize=7)

    for label, traj in all_trajs:
        ep  = [r["epoch"] for r in traj]
        frc = [r["pm1_frac"] for r in traj]
        axes[2].plot(ep, frc, label=label)
    axes[2].set(xlabel="epoch", ylabel="pm1_frac", title="L_pm1 / L_PDE fraction")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(root / "pm1_trajectory_comparison.png", dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Final bounded ±1-auxiliary experiment")
    ap.add_argument("--output-root",   default="outputs/phase5_pm1_aux")
    ap.add_argument("--shared-points", default="outputs/phase5_feature_scaling/shared_points.npz")
    ap.add_argument("--sweep-epochs",  type=int,   default=SWEEP_EPOCHS)
    ap.add_argument("--main-epochs",   type=int,   default=MAIN_EPOCHS)
    ap.add_argument("--w-pm1",         type=float, default=None,
                    help="Override sweep: use this fixed w_pm1")
    ap.add_argument("--skip-seed43",   action="store_true")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)

    # ── Load config ───────────────────────────────────────────────────────────
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

    pts_raw = np.load(ROOT / args.shared_points)
    pts     = {k: torch.as_tensor(v, dtype=dtype) for k, v in pts_raw.items()}

    # Phase 0 record
    phase0 = {
        "git_commit":             git_commit(),
        "working_tree_status":    git_status(),
        "canonical_sha256":       can_sha,
        "canonical_sha_ok":       True,
        "companion_sha256":       com_sha,
        "companion_sha_ok":       True,
        "model_config_hash":      model_config_hash(physics),
        "seed":                   SEED,
        "baseline_t1":            BASELINE_T1,
        "success_threshold":      SUCCESS_T1,
        "target_t1":              TARGET_T1,
        "loss_formula":           "L_total = L_PDE + w_pm1 * L_pm1",
        "L_pm1_formula":          "-(|t_{-1}|^2 + |t_{+1}|^2)  [no RCWA]",
        "modal_data_loss_weight": 0.0,
        "optical_coupling":       False,
        "rcwa_in_objective":      False,
        "max_total_epochs":       args.main_epochs,
    }
    (root / "phase0_preflight.json").write_text(json.dumps(phase0, indent=2) + "\n")
    print("[Phase 0] Preflight OK", flush=True)
    print(f"  target |t_±1| > {SUCCESS_T1}  (RCWA reference = 0.04941)", flush=True)

    # ── Phase 1: w_pm1 sweep ──────────────────────────────────────────────────
    sweep_w = [args.w_pm1] if args.w_pm1 is not None else SWEEP_W
    print(f"\n[Phase 1] Sweep w_pm1 in {sweep_w}  ({args.sweep_epochs} epochs each)",
          flush=True)

    sweep_results = {}
    sweep_trajs   = []
    for w in sweep_w:
        traj, best, best_v = train_run(
            seed=SEED, physics=physics, pts=pts, coeff=coeff,
            w_pm1=w, epochs=args.sweep_epochs,
            device=device, dtype=dtype, ref=ref,
            ref_path=ref_path, companion_p=companion_p,
            out_dir=root / f"sweep_w{w:.2f}",
            label=f"sweep_w{w:.2f}",
        )
        t1_max = max(r["t_minus1"] for r in traj)
        t1_valid_max = best_v["t1"] if best_v["state"] is not None else t1_max
        sweep_results[w] = {
            "t1_max": t1_max, "t1_final": traj[-1]["t_minus1"],
            "t1_valid_max": t1_valid_max,
            "best_epoch": best_v["epoch"] if best_v["state"] else best["epoch"],
            "best_valid_t1": best_v["t1"],
            "RT_max": max(r["R_plus_T"] for r in traj),
        }
        sweep_trajs.append((f"sweep_w{w:.2f}", traj))
        print(f"  w={w:.2f}:  t1_max={t1_max:.5f}  t1_final={traj[-1]['t_minus1']:.5f}  "
              f"RT_max={sweep_results[w]['RT_max']:.4f}", flush=True)

    # Choose best w: highest t1_valid_max while RT_max < 1.1
    stable = {w: v for w, v in sweep_results.items() if v["RT_max"] < 1.1}
    if stable:
        best_w = max(stable, key=lambda w: stable[w]["t1_valid_max"])
    else:
        best_w = max(sweep_results, key=lambda w: sweep_results[w]["t1_valid_max"])
    print(f"\n  Best w_pm1 = {best_w}  (t1_max={sweep_results[best_w]['t1_max']:.5f})",
          flush=True)

    (root / "sweep_summary.json").write_text(
        json.dumps(jsonable({"sweep_results": sweep_results,
                              "best_w": best_w}), indent=2) + "\n")

    # ── Phase 2: Main run seed 42 ─────────────────────────────────────────────
    print(f"\n[Phase 2] Main run  w_pm1={best_w}  seed=42  epochs={args.main_epochs}",
          flush=True)

    traj42, best42, best42_valid = train_run(
        seed=SEED, physics=physics, pts=pts, coeff=coeff,
        w_pm1=best_w, epochs=args.main_epochs,
        device=device, dtype=dtype, ref=ref,
        ref_path=ref_path, companion_p=companion_p,
        out_dir=root / "main_seed42",
        label=f"main_seed42_w{best_w:.2f}",
    )

    # Use the physically valid best for t1_max comparison and field plots
    t1_max42  = best42_valid["t1"] if best42_valid["state"] else max(r["t_minus1"] for r in traj42)
    t1_end42  = traj42[-1]["t_minus1"]
    success42 = t1_max42 > SUCCESS_T1
    print(f"\n  Seed 42 result: t1_valid_max={t1_max42:.5f}  success={success42}", flush=True)

    # Field plots from best VALID checkpoint
    plot_state = best42_valid["state"] if best42_valid["state"] else best42["state"]
    if plot_state is not None:
        make_field_plots(
            plot_state, physics, coeff, ref, device, dtype,
            root / "main_seed42",
            label=f"seed42_ep{best42_valid.get('epoch', best42['epoch'])}",
            ref_path=ref_path, companion_p=companion_p, pts=pts,
        )

    # ── Phase 3: Seed 43 confirmation ────────────────────────────────────────
    traj43 = None
    best43 = {}
    if success42 and not args.skip_seed43:
        print(f"\n[Phase 3] Seed 43 confirmation  w_pm1={best_w}  "
              f"epochs={CONFIRM_EPOCHS}", flush=True)
        traj43, best43, best43_valid = train_run(
            seed=43, physics=physics, pts=pts, coeff=coeff,
            w_pm1=best_w, epochs=CONFIRM_EPOCHS,
            device=device, dtype=dtype, ref=ref,
            ref_path=ref_path, companion_p=companion_p,
            out_dir=root / "confirm_seed43",
            label=f"seed43_w{best_w:.2f}",
        )
        plot_state43 = best43_valid["state"] if best43_valid.get("state") else best43.get("state")
        if plot_state43 is not None:
            make_field_plots(
                plot_state43, physics, coeff, ref, device, dtype,
                root / "confirm_seed43",
                label=f"seed43_ep{best43_valid.get('epoch', best43.get('epoch'))}",
                ref_path=ref_path, companion_p=companion_p, pts=pts,
            )
    else:
        reason = "success threshold not met" if not success42 else "--skip-seed43"
        print(f"\n[Phase 3] Skipped ({reason})", flush=True)

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n[Verification]", flush=True)
    compile_r = run_compileall()
    pytest_r  = run_pytest()
    print(f"  compileall: {'OK' if compile_r['passed'] else 'FAIL'}")
    print(f"  pytest: {pytest_r['summary_line']}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    all_trajs = sweep_trajs + [(f"main_seed42_w{best_w:.2f}", traj42)]
    if traj43:
        all_trajs.append((f"seed43_w{best_w:.2f}", traj43))
    baseline_path = (ROOT / "outputs/phase5_gradient_alignment"
                     / "trajectory/training_trajectory.csv")
    write_summary_plots(root, all_trajs, baseline_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    seed43_result = None
    if traj43:
        t43_valid = best43_valid["t1"] if best43_valid.get("state") else max(r["t_minus1"] for r in traj43)
        seed43_result = {
            "t1_valid_max": t43_valid,
            "t1_end": traj43[-1]["t_minus1"],
        }

    if success42:
        if seed43_result and seed43_result.get("t1_valid_max", 0) > SUCCESS_T1:
            outcome = "SUCCESS_CONFIRMED"
            outcome_detail = (
                f"Both seeds exceeded |t_±1| > {SUCCESS_T1:.3f}. "
                f"Seed 42 peak: {t1_max42:.4f}. "
                f"Seed 43 peak: {seed43_result['t1_max']:.4f}. "
                "Modest growth confirmed. Proceed to optical coupling analysis."
            )
        else:
            outcome = "SUCCESS_UNCONFIRMED"
            outcome_detail = (
                f"Seed 42 exceeded {SUCCESS_T1:.3f} (peak {t1_max42:.4f}). "
                "Seed 43 did not confirm. Single-seed result only."
            )
    else:
        outcome = "NEGATIVE"
        outcome_detail = (
            f"|t_±1| did not exceed {SUCCESS_T1:.3f}. "
            f"Seed 42 peak: {t1_max42:.5f} (baseline={BASELINE_T1:.5f}). "
            "Proceed to AIP paper as diagnostic/methods contribution."
        )

    summary = {
        **phase0,
        "sweep_best_w": best_w,
        "sweep_results": sweep_results,
        "seed42": {
            "t1_max":        t1_max42,
            "t1_end":        t1_end42,
            "best_epoch":    best42.get("epoch"),
            "success":       success42,
            "RT_max":        max(r["R_plus_T"] for r in traj42),
        },
        "seed43": seed43_result,
        "outcome":        outcome,
        "outcome_detail": outcome_detail,
        "phase5_passed":  False,
        "compileall":     compile_r,
        "pytest":         pytest_r,
    }
    (root / "pm1_aux_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n")

    # CSV
    all_rows = []
    for _, traj in all_trajs:
        for r in traj:
            all_rows.append({k: v for k, v in r.items()
                              if not isinstance(v, (dict, list))})
    if all_rows:
        with (root / "pm1_aux_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)

    print("\n" + "=" * 70, flush=True)
    print(json.dumps(jsonable({
        "outcome":           outcome,
        "outcome_detail":    outcome_detail[:200],
        "best_w":            best_w,
        "seed42_t1_max":     t1_max42,
        "seed42_t1_end":     t1_end42,
        "success_threshold": SUCCESS_T1,
        "target_t1":         TARGET_T1,
        "seed43_run":        traj43 is not None,
        "seed43_t1_max":     seed43_result["t1_max"] if seed43_result else None,
        "pytest":            pytest_r["summary_line"],
        "output_root":       str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
