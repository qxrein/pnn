#!/usr/bin/env python3
"""Generate all paper-ready assets.  Phases 0–8.

Phases
------
0  Safe preflight
1  Seed-43 confirmation run (w_pm1=1.0, 600 epochs)
2  Field plots from best valid checkpoint (seed 42)
3  Poynting-flux diagrams
4  Modal comparison plots + trajectories
5  Loss component breakdown
6  Gradient-alignment visualisations
7  Summary JSON/CSV
8  Verification (compileall + pytest)
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
from matplotlib.colors import CenteredNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.field_comparison import compare_modal_with_rcwa, extract_modal_amplitudes
from src.maxwell_layered_bg import (
    background_field_np,
    compute_background_coefficients,
)
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable
from scripts.train_lbg import evaluate, layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, MODAL_ORDER_MAX, N_DTN_ORDERS,
    SEED, TARGET_T1, model_config_hash, head_parameter_records,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import (
    git_commit, git_status, make_fresh_model, parameter_ordering_hash,
    run_compileall, run_pytest,
)
from scripts.run_feature_scaling import make_scaled_model
from scripts.run_pm1_aux import (
    BASELINE_T1, LR, LOG_EVERY, SUCCESS_T1, Z_BOT_FRAC,
    l_pm1_loss, read_l_pm1_magnitude, train_run,
)


# ── patch spatial_fourier_t ──────────────────────────────────────────────────
def _sfq(model, physics, z_bot, n_quad=512):
    x = torch.linspace(0, physics.period, n_quad+1, dtype=torch.float64)[:-1].requires_grad_(True)
    z = torch.empty_like(x.detach()).fill_(z_bot).requires_grad_(True)
    er, ei, *_ = model.net_sub.field_components(x, z)
    e = (er + 1j*ei).detach().numpy()
    xn = x.detach().numpy()
    g0 = 2*np.pi/physics.period
    return {m: complex(np.mean(e*np.exp(-1j*m*g0*xn)))
            for m in range(-MODAL_ORDER_MAX, MODAL_ORDER_MAX+1)}
_fhls.spatial_fourier_t = _sfq
from scripts.run_frozen_head_least_squares import modal_report

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

W_PM1 = 1.0
CONFIRM_EPOCHS = 600
CONFIRM_SEED   = 43

def _savefig(fig, path: Path, dpi=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _load_companion(companion_p: Path):
    return np.load(companion_p, allow_pickle=False)

def _modal_from_companion(companion, physics, z_bot):
    """t_m from companion at z_bot, de-embedded."""
    n = int((len(companion["c_trans"])-1)//2)
    out = {}
    for m in range(-3, 4):
        idx = n+m
        t   = complex(companion["c_trans"][idx])
        kz  = complex(companion["kz_substrate"][idx])
        t  *= np.exp(-1j*kz*(z_bot - physics.ridge_z_max))
        out[m] = t
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — seed-43 confirmation
# ─────────────────────────────────────────────────────────────────────────────

def phase1_seed43(physics, pts, coeff, device, dtype, ref, ref_path,
                  companion_p, root: Path) -> tuple[list[dict], dict]:
    print("\n[Phase 1] Seed-43 confirmation run", flush=True)
    out_dir = root / "seed43_run"
    traj43, best43, best43_valid = train_run(
        seed=CONFIRM_SEED,
        physics=physics, pts=pts, coeff=coeff,
        w_pm1=W_PM1, epochs=CONFIRM_EPOCHS,
        device=device, dtype=dtype, ref=ref,
        ref_path=ref_path, companion_p=companion_p,
        out_dir=out_dir,
        label=f"seed43_w{W_PM1:.1f}",
    )
    t1_valid = best43_valid["t1"] if best43_valid.get("state") else 0.0
    print(f"  Seed 43: best valid |t-1|={t1_valid:.5f}  "
          f"epoch={best43_valid.get('epoch',0)}", flush=True)

    # Save CSV + JSON
    csv_path = root / "seed43_trajectory.csv"
    if traj43:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj43[0]))
            writer.writeheader(); writer.writerows(traj43)

    summ43 = {
        "seed": CONFIRM_SEED, "w_pm1": W_PM1, "epochs": CONFIRM_EPOCHS,
        "t1_valid_max": t1_valid,
        "t1_valid_epoch": best43_valid.get("epoch", 0),
        "t1_final": traj43[-1]["t_minus1"] if traj43 else 0.0,
        "RT_max": max(r["R_plus_T"] for r in traj43) if traj43 else 0.0,
        "trajectory": [{k: v for k, v in r.items()
                        if not isinstance(v, (dict, list))} for r in traj43],
    }
    (root / "seed43_summary.json").write_text(json.dumps(jsonable(summ43), indent=2)+"\n")
    return traj43, best43_valid


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — field plots
# ─────────────────────────────────────────────────────────────────────────────

def phase2_field_plots(state_dict, physics, coeff, ref, ref_path,
                       companion_p, pts, device, dtype, root: Path):
    print("\n[Phase 2] Field plots", flush=True)
    out_dir = root / "field_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    model_p, _ = make_scaled_model(physics, pts)
    model_p.load_state_dict(state_dict)
    model_p.eval()

    # Evaluate on visualization grid
    with torch.enable_grad():
        fields = evaluate(model_p, physics, device, dtype, "layered_bg", coeff)
    x1d = fields["x"][0];  z1d = fields["z"][:, 0]
    X, Z = np.meshgrid(x1d, z1d)

    # PINN total field
    Ebg_r, Ebg_i, Hbg_x_r, Hbg_x_i = background_field_np(z1d, coeff)
    E_total_r = fields["E_scat_real"] + Ebg_r[:, None]
    E_total_i = fields["E_scat_imag"] + Ebg_i[:, None]
    E_pinn    = E_total_r + 1j * E_total_i

    # RCWA reference
    rr, ri, _ = interpolate_reference_to_grid(ref, x1d, z1d)
    E_rcwa    = rr + 1j * ri

    # ── plot helpers ──────────────────────────────────────────────────────────
    cmap_amp   = "viridis"
    cmap_phase = "hsv"
    extent     = [x1d[0], x1d[-1], z1d[-1], z1d[0]]  # z from top

    def _pcolour(ax, data, cmap, title, vmin=None, vmax=None):
        im = ax.pcolormesh(X, Z, data, cmap=cmap,
                           shading="auto", vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="x / λ", ylabel="z / λ")
        ax.invert_yaxis()
        return im

    # 1 |E_y| PINN
    fig, ax = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, np.abs(E_pinn), cmap_amp, r"PINN $|E_y|$")
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "Ey_PINN.png")

    # 2 |E_y| RCWA
    fig, ax = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, np.abs(E_rcwa), cmap_amp, r"RCWA $|E_y|$")
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "Ey_RCWA.png")

    # 3 phase PINN
    fig, ax = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, np.angle(E_pinn)*180/np.pi, cmap_phase,
                  r"PINN $\angle E_y$ [°]", vmin=-180, vmax=180)
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "phase_PINN.png")

    # 4 phase RCWA
    fig, ax = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, np.angle(E_rcwa)*180/np.pi, cmap_phase,
                  r"RCWA $\angle E_y$ [°]", vmin=-180, vmax=180)
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "phase_RCWA.png")

    # 5 |E_y| difference
    diff_abs = np.abs(E_pinn - E_rcwa)
    fig, ax  = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, diff_abs, "inferno", r"$|E_{\rm PINN} - E_{\rm RCWA}|$")
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "Ey_diff.png")

    # 6 phase difference  (wrapped ±180°)
    phase_diff = np.angle(E_pinn * np.conj(E_rcwa)) * 180/np.pi
    fig, ax    = plt.subplots(figsize=(5, 4))
    im = _pcolour(ax, phase_diff, "RdBu_r",
                  r"Phase diff $\angle(E_{\rm PINN}/E_{\rm RCWA})$ [°]",
                  vmin=-180, vmax=180)
    plt.colorbar(im, ax=ax)
    _savefig(fig, out_dir / "phase_diff.png")

    # Save NPZ
    np.savez(out_dir / "field_comparison.npz",
             x=x1d, z=z1d,
             E_pinn_r=E_total_r, E_pinn_i=E_total_i,
             E_rcwa_r=rr, E_rcwa_i=ri,
             Ey_diff=diff_abs, phase_diff=phase_diff)
    print(f"  Saved 6 field plots + NPZ to {out_dir}", flush=True)

    # Return for downstream use
    return E_pinn, E_rcwa, x1d, z1d, model_p, fields


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Poynting-flux diagrams
# ─────────────────────────────────────────────────────────────────────────────

def phase3_poynting(model_p, E_pinn, E_rcwa, x1d, z1d,
                    physics, coeff, companion_p, root: Path):
    print("\n[Phase 3] Poynting-flux diagrams", flush=True)
    out_dir = root / "poynting_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Z = np.meshgrid(x1d, z1d)

    # ── PINN magnetic fields ─────────────────────────────────────────────────
    xf = torch.as_tensor(X.ravel(), dtype=torch.float64)
    zf = torch.as_tensor(Z.ravel(), dtype=torch.float64)
    p  = physics
    Hx_scat = torch.zeros_like(xf)
    Hz_scat = torch.zeros_like(xf)
    with torch.enable_grad():
        for mask, net in [
            (zf <= p.ridge_z_min, model_p.net_air),
            ((zf > p.ridge_z_min) & (zf <= p.ridge_z_max), model_p.net_grat),
            (zf > p.ridge_z_max,  model_p.net_sub),
        ]:
            if mask.any():
                xm = xf[mask].requires_grad_(True)
                zm = zf[mask].requires_grad_(True)
                out = net.forward(xm, zm)
                Hx_scat[mask] = out[:, 2].detach()
                Hz_scat[mask] = out[:, 4].detach()

    Hx_s = Hx_scat.numpy().reshape(X.shape)
    Hz_s = Hz_scat.numpy().reshape(X.shape)

    # Add background magnetic field
    Ebg_r, Ebg_i, Hbg_r, Hbg_i = background_field_np(z1d, coeff)
    Hx_total_r = Hx_s + Hbg_r[:, None]
    Hz_total   = Hz_s  # Hz_bg = 0 for 1D background

    # TE Poynting components (real-valued, time-averaged)
    # S_x = 0.5 Re(E_y * conj(H_z)) — note TE: E_y, H_x, H_z
    # S_z = 0.5 Re(-E_y * conj(H_x))
    # Using complex E_pinn and Hx_total = Hx_r (real part for now)
    # Full complex H_x requires imaginary part; use stored fields
    # Approximate with real parts only (H_x_i from horizontal derivative of E)
    Sz_pinn = 0.5 * np.real(-E_pinn * np.conj(Hx_total_r + 1j * 0))
    Sx_pinn = 0.5 * np.real(E_pinn  * np.conj(Hz_total   + 1j * 0))

    # ── RCWA magnetic fields from companion ───────────────────────────────────
    companion = _load_companion(companion_p)
    Hx_rcwa = companion["Hx_total"].reshape(len(z1d), len(x1d)) if "Hx_total" in companion.files else None
    Hz_rcwa = companion["Hz_total"].reshape(len(z1d), len(x1d)) if "Hz_total" in companion.files else None
    if Hx_rcwa is None:
        # Interpolate from companion grid
        from src.reference_data import interpolate_reference_to_grid
        # Companion stores Ey on its own grid
        Sz_rcwa = 0.5 * np.real(-E_rcwa * np.conj(np.zeros_like(E_rcwa)))
        Sx_rcwa = Sz_rcwa.copy()
        print("  RCWA H fields not available in companion; using zero H reference",
              flush=True)
    else:
        Sz_rcwa = 0.5 * np.real(-E_rcwa * np.conj(Hx_rcwa))
        Sx_rcwa = 0.5 * np.real(E_rcwa  * np.conj(Hz_rcwa))

    # Output plane (z_bot)
    iz_bot = int(np.argmin(np.abs(z1d - Z_BOT_FRAC * physics.domain_height)))
    z_bot_val = z1d[iz_bot]

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x1d, Sz_pinn[iz_bot, :], label=f"PINN z={z_bot_val:.2f}λ")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="x / λ", ylabel=r"$S_z$ [arb.]",
           title=r"Poynting $S_z$ at output plane (PINN)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "Sz_output_PINN.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x1d, Sz_rcwa[iz_bot, :], label=f"RCWA z={z_bot_val:.2f}λ")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="x / λ", ylabel=r"$S_z$ [arb.]",
           title=r"Poynting $S_z$ at output plane (RCWA)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "Sz_output_RCWA.png")

    # Vertical cut at ridge centre
    ix_ridge = int(np.argmin(np.abs(x1d - physics.period/2)))
    fig, ax  = plt.subplots(figsize=(4, 5))
    ax.plot(Sx_pinn[:, ix_ridge], z1d, label="PINN")
    ax.axhline(physics.ridge_z_min, color="gray", ls="--", lw=0.7, label="ridge")
    ax.axhline(physics.ridge_z_max, color="gray", ls="--", lw=0.7)
    ax.invert_yaxis()
    ax.set(xlabel=r"$S_x$ [arb.]", ylabel="z / λ",
           title=r"Poynting $S_x$ vertical cut (PINN)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "Sx_cut_PINN.png")

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(Sx_rcwa[:, ix_ridge], z1d, label="RCWA")
    ax.axhline(physics.ridge_z_min, color="gray", ls="--", lw=0.7, label="ridge")
    ax.axhline(physics.ridge_z_max, color="gray", ls="--", lw=0.7)
    ax.invert_yaxis()
    ax.set(xlabel=r"$S_x$ [arb.]", ylabel="z / λ",
           title=r"Poynting $S_x$ vertical cut (RCWA)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "Sx_cut_RCWA.png")

    np.savez(out_dir / "poynting_data.npz",
             x=x1d, z=z1d,
             Sz_pinn=Sz_pinn, Sx_pinn=Sx_pinn,
             Sz_rcwa=Sz_rcwa, Sx_rcwa=Sx_rcwa,
             iz_bot=iz_bot, ix_ridge=ix_ridge)
    print(f"  Saved 4 Poynting plots + NPZ to {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — modal comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def phase4_modal_plots(model_p, E_pinn, x1d, z1d, physics, companion_p,
                       ref_path, traj42, traj43, root: Path):
    print("\n[Phase 4] Modal comparison plots", flush=True)
    out_dir = root / "modal_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modal amplitudes from PINN (production path)
    modal_pinn = extract_modal_amplitudes(
        E_pinn, x1d, z1d, physics, n_orders=3,
        formulation="layered_bg", field_representation="total")
    z_bot_mon = float(modal_pinn["z_bot_monitor"])
    orders_all = list(range(-3, 4))
    idx_offset = 3   # n_orders=3 → m=-3..+3, index 0..6

    companion = _load_companion(companion_p)
    t_rcwa    = _modal_from_companion(companion, physics, z_bot_mon)

    t_pinn_abs = [abs(modal_pinn["t_m_complex"][idx_offset + m]) for m in orders_all]
    t_rcwa_abs = [abs(t_rcwa[m]) for m in orders_all]
    t_pinn_c   = [complex(modal_pinn["t_m_complex"][idx_offset + m]) for m in orders_all]
    t_rcwa_c   = [t_rcwa[m] for m in orders_all]

    # 1 Bar chart of |t_m|
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos   = np.arange(len(orders_all))
    ax.bar(x_pos - 0.2, t_rcwa_abs, 0.4, label="RCWA", color="steelblue")
    ax.bar(x_pos + 0.2, t_pinn_abs, 0.4, label="PINN (ep 600)", color="coral")
    ax.set_xticks(x_pos, [str(m) for m in orders_all])
    ax.set(xlabel="Diffraction order m", ylabel=r"$|t_m|$",
           title="Transmitted modal amplitudes")
    ax.legend()
    _savefig(fig, out_dir / "modal_amplitudes_comparison.png")

    # 2 Complex plane
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, m in enumerate(orders_all):
        ax.scatter(t_rcwa_c[i].real, t_rcwa_c[i].imag, marker="+",
                   s=120, color="steelblue", zorder=3)
        ax.scatter(t_pinn_c[i].real, t_pinn_c[i].imag, marker="o",
                   s=60, color="coral", zorder=3)
        ax.annotate(f"m={m}", (t_rcwa_c[i].real, t_rcwa_c[i].imag),
                    fontsize=8, color="steelblue",
                    xytext=(4, 4), textcoords="offset points")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="+", color="steelblue", ms=10, ls="", label="RCWA"),
        Line2D([0],[0], marker="o", color="coral",    ms=8,  ls="", label="PINN"),
    ])
    ax.set(xlabel=r"Re$(t_m)$", ylabel=r"Im$(t_m)$",
           title="Complex modal amplitudes")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    _savefig(fig, out_dir / "complex_modal_plane.png")

    # 3 |t_±1| trajectory seed 42 + seed 43
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if traj42:
        ep42 = [r["epoch"] for r in traj42]
        axes[0].plot(ep42, [r["t_minus1"] for r in traj42], label="seed 42 |t_{-1}|")
        axes[0].plot(ep42, [r["t_plus1"]  for r in traj42], ls="--", label="seed 42 |t_{+1}|")
    if traj43:
        ep43 = [r["epoch"] for r in traj43]
        axes[0].plot(ep43, [r["t_minus1"] for r in traj43], label="seed 43 |t_{-1}|", alpha=0.8)
        axes[0].plot(ep43, [r["t_plus1"]  for r in traj43], ls="--", label="seed 43 |t_{+1}|", alpha=0.8)
    axes[0].axhline(TARGET_T1,  color="k",      ls="--", lw=0.8, label=f"target {TARGET_T1:.4f}")
    axes[0].axhline(BASELINE_T1, color="r",     ls=":",  lw=0.8, label=f"baseline {BASELINE_T1:.4f}")
    axes[0].axhline(0.019,       color="green",  ls=":",  lw=0.8, label="best valid 0.019")
    axes[0].set(xlabel="epoch", ylabel=r"$|t_{\pm1}|$",
                title=r"Transmitted $\pm1$ amplitudes vs epoch")
    axes[0].legend(fontsize=7)

    # 4 R+T trajectory
    if traj42:
        axes[1].plot(ep42, [r["R_plus_T"] for r in traj42], label="seed 42")
    if traj43:
        axes[1].plot(ep43, [r["R_plus_T"] for r in traj43], label="seed 43", alpha=0.8)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].axhline(1.05, color="r", ls=":", lw=0.8, label="R+T=1.05 gate")
    axes[1].set(xlabel="epoch", ylabel="R+T", title="Energy balance R+T vs epoch")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_dir / "t_pm1_trajectory.png")

    # Also save standalone R+T plot
    fig, ax = plt.subplots(figsize=(7, 4))
    if traj42:
        ax.plot(ep42, [r["R_plus_T"] for r in traj42], label="seed 42")
    if traj43:
        ax.plot([r["epoch"] for r in traj43], [r["R_plus_T"] for r in traj43],
                label="seed 43", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(1.05, color="r", ls=":", lw=0.8)
    ax.set(xlabel="epoch", ylabel="R+T", title="Energy balance R+T")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "RT_trajectory.png")

    # Save modal comparison CSV
    rows = []
    for i, m in enumerate(orders_all):
        rows.append({
            "m": m,
            "t_pinn_abs":      t_pinn_abs[i],
            "t_rcwa_abs":      t_rcwa_abs[i],
            "t_pinn_re":       t_pinn_c[i].real,
            "t_pinn_im":       t_pinn_c[i].imag,
            "t_rcwa_re":       t_rcwa_c[i].real,
            "t_rcwa_im":       t_rcwa_c[i].imag,
            "amp_error":       abs(t_pinn_abs[i] - t_rcwa_abs[i]),
            "complex_error":   abs(t_pinn_c[i] - t_rcwa_c[i]),
            "z_bot_monitor":   z_bot_mon,
        })
    with (out_dir / "modal_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Trajectory CSVs
    if traj42:
        with (out_dir / "trajectory_seed42.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj42[0]))
            writer.writeheader(); writer.writerows(traj42)
    if traj43:
        with (out_dir / "trajectory_seed43.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj43[0]))
            writer.writeheader(); writer.writerows(traj43)

    print(f"  Saved modal plots + CSVs to {out_dir}", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — loss component breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _get(row, *keys):
    """Get first matching key from row, default 0."""
    for k in keys:
        if k in row:
            v = row[k]
            return float(v) if not isinstance(v, float) else v
    return 0.0


def phase5_loss_plots(traj42, traj43, root: Path):
    print("\n[Phase 5] Loss component plots", flush=True)
    out_dir = root / "loss_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not traj42:
        print("  No trajectory data", flush=True)
        return

    ep = [int(float(r["epoch"])) for r in traj42]

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [("pde_air","air"),("pde_grat","ridge"),("pde_sub","substrate")]:
        vals = [_get(r, key) for r in traj42]
        if any(v > 0 for v in vals):
            ax.semilogy(ep, vals, label=label)
    ax.set(xlabel="epoch", ylabel="PDE MSE loss", title="PDE loss by region (seed 42)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "pde_loss_by_region.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [("E_int","E continuity"),("H_int","H continuity")]:
        vals = [_get(r, key) for r in traj42]
        if any(v > 0 for v in vals):
            ax.semilogy(ep, vals, label=label)
    ax.set(xlabel="epoch", ylabel="interface loss", title="Interface loss (seed 42)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "interface_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [("top_DtN","top DtN"),("bottom_DtN","bottom DtN")]:
        vals = [_get(r, key) for r in traj42]
        if any(v > 0 for v in vals):
            ax.semilogy(ep, vals, label=label)
    ax.set(xlabel="epoch", ylabel="DtN loss", title="DtN boundary loss (seed 42)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "dtn_loss.png")

    l_pm1_vals = [_get(r, "L_pm1") for r in traj42]
    fig, ax = plt.subplots(figsize=(8, 4))
    if any(v > 0 for v in l_pm1_vals):
        ax.semilogy(ep, [max(v, 1e-20) for v in l_pm1_vals],
                    label=r"$|t_{-1}|^2+|t_{+1}|^2$")
    ax.set(xlabel="epoch", ylabel=r"$L_{\pm1}$",
           title=r"Auxiliary $\pm1$ loss (seed 42)")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "aux_loss.png")

    frac_vals = [_get(r, "pm1_frac") for r in traj42]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, frac_vals)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(0.05, color="r", ls=":", lw=0.8)
    ax.set(xlabel="epoch", ylabel=r"$w_{\pm1}L_{\pm1}/L_{\rm PDE}$",
           title=r"$\pm1$ loss fraction (seed 42)")
    _savefig(fig, out_dir / "loss_fractions.png")

    with (out_dir / "loss_components.csv").open("w", newline="") as fh:
        if traj42:
            writer = csv.DictWriter(fh, fieldnames=list(traj42[0]))
            writer.writeheader(); writer.writerows(traj42)
    print(f"  Saved 5 loss plots + CSV to {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — gradient-alignment visualisations
# ─────────────────────────────────────────────────────────────────────────────

def phase6_gradient_plots(root: Path):
    print("\n[Phase 6] Gradient-alignment visualisations", flush=True)
    out_dir = root / "gradient_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ga_dir = ROOT / "outputs/phase5_gradient_alignment"
    if not ga_dir.exists():
        print("  Gradient-alignment directory not found — skipping", flush=True)
        return

    # Load data
    sigma = np.load(ga_dir / "singular_values.npy")   # (1386,)

    import csv as _csv
    ms_rows = list(_csv.DictReader((ga_dir / "modal_singular_direction_sensitivity.csv").open()))
    ga_traj = list(_csv.DictReader((ga_dir / "trajectory/training_trajectory.csv").open()))
    ga_summary = json.loads((ga_dir / "gradient_alignment_summary.json").read_text())

    # 1 Singular-value spectrum
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(sigma)+1),
                np.maximum(sigma, np.finfo(float).tiny), lw=0.8, color="steelblue")
    rank = ga_summary.get("svd", {}).get("rank", 216)
    ax.axvline(rank, color="r", ls="--", lw=0.8, label=f"rank={rank}")
    ax.set(xlabel="Singular value index", ylabel=r"$\sigma_i$",
           title="SVD spectrum of residual Jacobian")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "singular_values.png")

    # 2 ±1 sensitivity by singular direction
    pm1_vals  = [float(r["combined_pm1_sensitivity"]) for r in ms_rows]
    active    = [r["active"] == "True" for r in ms_rows]
    pm1_active = [v if a else np.nan for v, a in zip(pm1_vals, active)]
    pm1_null   = [v if not a else np.nan for v, a in zip(pm1_vals, active)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x_idx   = np.arange(len(pm1_vals))
    ax.semilogy(x_idx, np.where(np.isfinite(pm1_active), pm1_active, np.nan),
                "b.", ms=2, label="active")
    ax.semilogy(x_idx, np.where(np.isfinite(pm1_null), pm1_null, np.nan),
                "r.", ms=1, alpha=0.5, label="null")
    ax.axvline(rank, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="Singular direction index", ylabel=r"$s_{\pm1}$",
           title=r"$\pm1$ sensitivity of each singular direction")
    ax.legend(fontsize=9)
    _savefig(fig, out_dir / "pm1_sensitivity.png")

    # 3 Gradient projection over epochs (from GA trajectory)
    ep_ga    = [int(float(r["epoch"])) for r in ga_traj]
    frac_pm1 = [float(r["hgrad_frac_pm1"]) for r in ga_traj]
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(ep_ga, frac_pm1, lw=1.5)
    ax.set(xlabel="epoch", ylabel=r"$\|P_{\pm1}\,g\| / \|g\|$",
           title=r"Head gradient fraction in $\pm1$-sensitive subspace")
    _savefig(fig, out_dir / "gradient_projection.png")

    # 4 Directional derivative of t_±1
    d_t1 = [float(r["d_t_minus1_along_grad"]) for r in ga_traj]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep_ga, d_t1, lw=1.5)
    ax.set(xlabel="epoch", ylabel=r"$\|J_{t_{-1}}\,\hat{g}\|$",
           title=r"Directional derivative of $|t_{-1}|$ along gradient")
    _savefig(fig, out_dir / "directional_derivative.png")

    # Save CSV
    rows = []
    for r in ga_traj:
        rows.append({
            "epoch": r["epoch"],
            "hgrad_frac_pm1": r["hgrad_frac_pm1"],
            "d_t_minus1_along_grad": r["d_t_minus1_along_grad"],
            "t_minus1": r["t_minus1"],
            "R_plus_T": r["R_plus_T"],
        })
    with (out_dir / "gradient_alignment.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"  Saved 4 gradient plots + CSV to {out_dir}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — summary JSON / CSV
# ─────────────────────────────────────────────────────────────────────────────

def phase7_summary(
    traj42, best42_valid,
    traj43, best43_valid,
    modal_rows,
    physics, companion_p,
    root: Path,
) -> dict:
    print("\n[Phase 7] Summary table", flush=True)

    companion = _load_companion(companion_p)
    n_harm    = int((len(companion["c_trans"])-1)//2)

    # RCWA companion values (at bulk z_bot=0)
    t_rcwa_m1 = abs(companion["c_trans"][n_harm-1])
    t_rcwa_p1 = abs(companion["c_trans"][n_harm+1])
    t_rcwa_0  = abs(companion["c_trans"][n_harm])

    # Best valid checkpoints
    t1_s42  = best42_valid.get("t1", 0.0)
    ep_s42  = best42_valid.get("epoch", 0)
    t1_s43  = best43_valid.get("t1", 0.0)
    ep_s43  = best43_valid.get("epoch", 0)

    # modal comparison
    m_row = {r["m"]: r for r in modal_rows} if modal_rows else {}

    def _rt_at_best(traj, best_ep):
        for r in traj:
            if int(float(r.get("epoch", 0))) == best_ep:
                return float(r.get("R_plus_T", 0))
        return 0.0

    rt_s42 = _rt_at_best(traj42, ep_s42)
    rt_s43 = _rt_at_best(traj43, ep_s43) if traj43 else 0.0

    # hashes
    can_path = ROOT / 'outputs/reference_lambda_0p8_geometry_consistent_20260824.npz'
    man      = json.loads((ROOT / 'outputs/reference_companion/manifest.json').read_text())
    can_sha  = hashlib.sha256(can_path.read_bytes()).hexdigest()
    com_sha  = hashlib.sha256(companion_p.read_bytes()).hexdigest()

    from src.config import load_config
    from scripts.run_frozen_head_least_squares import model_config_hash, head_parameter_records
    from scripts.run_frozen_head_integrity import parameter_ordering_hash
    from scripts.run_feature_scaling import FEATURE_SCALE_FLOOR
    cfg_hash = model_config_hash(physics)
    records  = head_parameter_records(ExplicitFourierModalDD(physics, 3, True).double())
    ord_hash = parameter_ordering_hash(records)
    fs_rep   = json.loads((ROOT / 'outputs/phase5_feature_scaling/feature_scale_report.json').read_text())
    fs_hash  = hashlib.sha256(
        json.dumps({k: v['scales'] for k, v in fs_rep['phase1_feature_scales'].items()},
                   sort_keys=True).encode()).hexdigest()

    summary = {
        "experiment": "pm1_auxiliary_loss_w1.0",
        "git_commit":  git_commit(),
        "canonical_sha256":  can_sha,
        "canonical_sha_ok":  can_sha == man["canonical_sha256"],
        "companion_sha256":  com_sha,
        "companion_sha_ok":  com_sha == man["companion_sha256"],
        "model_config_hash": cfg_hash,
        "param_order_hash":  ord_hash,
        "feature_scale_hash": fs_hash,
        "seed":   SEED, "confirm_seed": CONFIRM_SEED,
        "w_pm1":  W_PM1, "epochs": CONFIRM_EPOCHS,
        "modal_data_loss_weight": 0.0,
        "optical_coupling": False,
        "phase5_passed": False,
        # seed 42
        "seed42_best_t_minus1": t1_s42,
        "seed42_best_t_plus1":  float(m_row.get("1", {}).get("t_pinn_abs", 0)),
        "seed42_best_epoch":    ep_s42,
        "seed42_RT_at_best":    rt_s42,
        # seed 43
        "seed43_best_t_minus1": t1_s43,
        "seed43_best_epoch":    ep_s43,
        "seed43_RT_at_best":    rt_s43,
        "seed43_reproduced":    abs(t1_s43 - t1_s42) / max(t1_s42, 1e-8) < 0.20,
        # RCWA reference
        "rcwa_t_minus1":   t_rcwa_m1,
        "rcwa_t_plus1":    t_rcwa_p1,
        "rcwa_t0":         t_rcwa_0,
        "target_t_pm1":    TARGET_T1,
        "baseline_t_pm1":  BASELINE_T1,
        "improvement_factor": t1_s42 / BASELINE_T1,
        "fraction_of_target": t1_s42 / TARGET_T1,
        # modal errors (at seed42 best valid)
        "t_minus1_amp_error": abs(t1_s42 - t_rcwa_m1),
        "t_minus1_frac_of_rcwa": t1_s42 / max(t_rcwa_m1, 1e-10),
        # modal rows from field comparison
        "modal_comparison": modal_rows,
    }

    (root / "paper_summary.json").write_text(json.dumps(jsonable(summary), indent=2)+"\n")

    # CSV
    rows = [{
        "metric": k, "value": v
        } for k, v in summary.items()
        if isinstance(v, (int, float, str, bool)) and k != "modal_comparison"]
    with (root / "paper_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric","value"])
        writer.writeheader(); writer.writerows(rows)

    print(f"  Saved paper_summary.json + .csv to {root}", flush=True)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    root = ROOT / "outputs/paper_assets"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)

    # Load config + reference
    cfg         = load_config(ROOT / "configs/default.yaml")
    physics     = make_lambda_0p8(cfg.physics)
    ref_path    = ROOT / CANONICAL_REF
    companion_p = ROOT / COMPANION_PATH
    validate_reference(ref_path, physics)
    ref         = normalize_reference_orientation(load_reference_npz(ref_path))
    device      = torch.device("cpu")
    dtype       = torch.float64
    coeff       = compute_background_coefficients(physics)

    pts_raw = np.load(ROOT / "outputs/phase5_feature_scaling/shared_points.npz")
    pts     = {k: torch.as_tensor(v, dtype=dtype) for k, v in pts_raw.items()}

    # Phase 0 — record
    can_sha = sha256_file(ref_path)
    com_sha = sha256_file(companion_p)
    man     = json.loads((ROOT / "outputs/reference_companion/manifest.json").read_text())
    assert can_sha == man["canonical_sha256"], "canonical SHA mismatch"
    assert com_sha == man["companion_sha256"], "companion SHA mismatch"
    print("[Phase 0] Preflight OK", flush=True)

    # Load seed-42 trajectory and best valid checkpoint
    pm1_dir    = ROOT / "outputs/phase5_pm1_aux/main_seed42"
    traj_path  = pm1_dir / "trajectory.csv"
    ckpt_path  = pm1_dir / "best_valid_checkpoint.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Seed-42 best valid checkpoint not found: {ckpt_path}")

    traj42 = list(csv.DictReader(traj_path.open())) if traj_path.exists() else []
    # Convert string values to float where appropriate
    for r in traj42:
        for k in list(r):
            try: r[k] = float(r[k])
            except ValueError: pass

    ckpt42     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best42_valid = {
        "t1":    ckpt42.get("t1_abs", 0.0),
        "epoch": ckpt42.get("epoch", 0),
        "state": ckpt42.get("state_dict"),
    }
    print(f"  Seed 42 best valid: epoch={best42_valid['epoch']}  "
          f"|t-1|={best42_valid['t1']:.5f}", flush=True)

    # Phase 1 — seed-43 confirmation
    traj43, best43_valid = phase1_seed43(
        physics, pts, coeff, device, dtype, ref, ref_path, companion_p, root)

    # Phase 2 — field plots
    E_pinn, E_rcwa, x1d, z1d, model_p, fields = phase2_field_plots(
        best42_valid["state"], physics, coeff, ref, ref_path,
        companion_p, pts, device, dtype, root)

    # Phase 3 — Poynting
    phase3_poynting(model_p, E_pinn, E_rcwa, x1d, z1d,
                    physics, coeff, companion_p, root)

    # Phase 4 — modal plots
    modal_rows = phase4_modal_plots(
        model_p, E_pinn, x1d, z1d, physics, companion_p,
        ref_path, traj42, traj43, root)

    # Phase 5 — loss plots
    phase5_loss_plots(traj42, traj43, root)

    # Phase 6 — gradient alignment
    phase6_gradient_plots(root)

    # Phase 7 — summary
    summary = phase7_summary(
        traj42, best42_valid,
        traj43, best43_valid,
        modal_rows,
        physics, companion_p, root)

    # Phase 8 — verification
    print("\n[Phase 8] Verification", flush=True)
    cr = run_compileall()
    pr = run_pytest()
    print(f"  compileall: {'OK' if cr['passed'] else 'FAIL'}")
    print(f"  pytest: {pr['summary_line']}")

    # List all output files
    all_files = sorted(root.rglob("*"))
    file_list = [str(p.relative_to(ROOT)) for p in all_files if p.is_file()]

    final = {
        **{k: v for k, v in summary.items()
           if isinstance(v, (int, float, str, bool))},
        "compileall": cr,
        "pytest": pr,
        "output_files": file_list,
    }
    (root / "generation_report.json").write_text(
        json.dumps(jsonable(final), indent=2)+"\n")

    print("\n" + "="*70, flush=True)
    print(json.dumps(jsonable({
        "seed42_t_minus1_best_valid": best42_valid["t1"],
        "seed42_best_epoch":         best42_valid["epoch"],
        "seed43_t_minus1_best_valid": best43_valid.get("t1", 0.0),
        "seed43_best_epoch":          best43_valid.get("epoch", 0),
        "seed43_reproduced":          summary.get("seed43_reproduced"),
        "improvement_vs_baseline":    f"{summary['improvement_factor']:.1f}×",
        "fraction_of_rcwa_target":    f"{summary['fraction_of_target']*100:.1f}%",
        "rcwa_t_pm1":                 TARGET_T1,
        "phase5_passed":              False,
        "pytest":                     pr["summary_line"],
        "n_output_files":             len(file_list),
        "output_root":                str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
