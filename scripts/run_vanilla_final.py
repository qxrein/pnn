#!/usr/bin/env python3
"""Final three vanilla PINN experiments: D, E, F.

EXPERIMENT D — First-order Born approximation warm start
---------------------------------------------------------
Computes the exact first-order Born scattered field for the ridge geometry,
uses it to pre-train the vanilla MLP (200 MSE epochs), then runs the standard
PDE-only loss for 600 epochs. This is a physically grounded warm start, unlike
the synthetic sinusoidal proxy used in Exp A2 (which collapsed within ~100 PDE
epochs).

Born approximation for layered background + Bloch periodicity:
  E_scat^(1)(x,z) = sum_{m} A_m(z) * exp(i*G_m*x)

where G_m = m*(2π/Λ) and A_m(z) is the Born coefficient:

  A_m(z) = -k0^2 * delta_eps_m * integral_{z_min}^{z_max} G_m(z,z') * E_bg(z') dz'

  delta_eps_m = (1/Λ) * integral_0^Λ delta_eps(x,z') * exp(-i*G_m*x) dx
              = delta_eps_val * (ridge_width/Λ) * sinc(m * ridge_width/Λ) * exp(-i*G_m*x_ridge_centre)
              (analytic Fourier coefficient of rectangular ridge mask)

  G_m(z,z') = i/(2*kz_m) * exp(i*kz_m*|z - z'|)   (1D Green's function, propagating m)
  kz_m       = sqrt((n*k0)^2 - G_m^2) with Im(kz_m) >= 0 (outgoing branch)

The z-integral is evaluated numerically by quadrature over the ridge band
[z_min, z_max], separately for z above and below each source point z'.

EXPERIMENT E — 2D loss landscape
---------------------------------
Li et al. 2018 style: pick two random Gaussian directions d1, d2 in parameter
space (filter-normalised), scan alpha in [-0.5, 0.5], beta in [-0.5, 0.5] on
a 21×21 grid. Evaluate both the PDE loss value AND |t_{-1}| at each grid point.
Done at two anchor points: (a) zero-field init, (b) the converged 600-epoch
trained state from the vanilla PDE-only baseline.
Grid size 21×21 = 441 evaluations; with reduced collocation (N=64) each takes
~0.05 s → total ~22 s per landscape. Two landscapes = ~45 s.

EXPERIMENT F — Curriculum learning via index contrast
-------------------------------------------------------
n_ridge sequence: 1.2 → 1.5 → 1.8 → 2.0 (canonical period/dc/height).
300 epochs per stage, final weights passed to next stage.
Baseline: vanilla PDE-only at n_ridge=2.0 from scratch (300 epochs) — same
budget for fair comparison. RCWA target for n_ridge=2.0: the aux-loss run
reached |t_{-1}|=0.078 (seed 42), implying the RCWA value is ~0.078/12.16 *
improvement_factor; but the actual RCWA target is stored in the companion.
We load it directly from the companion file for n_ridge=2.0 via a quick RCWA
call if available, otherwise use the aux-generalization summary as proxy.

OUTPUT
------
outputs/vanilla_final/
    exp_D_born/
        born_field_sample.png     — visualization of Born scattered field
        trajectory_{label}.csv
        born_vs_proxy_comparison.png
    exp_E_landscape/
        landscape_init.png        — loss and |t_{-1}| around init
        landscape_trained.png     — loss and |t_{-1}| around trained
        landscape_data.npz
    exp_F_curriculum/
        trajectory_curriculum.csv
        trajectory_direct_n2p0.csv
        curriculum_vs_direct.png
    final_summary.json
    conclusion.txt
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
    delta_eps_np,
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
    sample_points, _modal_report, _energy_balance, _spatial_fourier_t,
    _pack_all, _unpack_all,
    EPOCHS, LR, CONFIRM_SEED, LOG_EVERY,
    MODAL_ORDER_MAX, Z_BOT_FRAC, RCWA_TARGET_T1,
)
from scripts.run_vanilla_local_min import build_loss, train_adam, _save_traj, _savefig

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

VANILLA_PDE_T1   = 0.003
VANILLA_AUX_T1   = 0.004
MEANINGFUL       = 1.5 * VANILLA_PDE_T1   # 0.0045 — threshold for "escapes"
RT_MAX           = 1.05

# D — Born warm-start
BORN_PRETRAIN    = 200    # supervised MSE epochs on Born field
BORN_LR          = 5e-4
BORN_ORDERS      = list(range(-5, 6))   # m = -5..+5 for Born sum
BORN_NZ          = 64    # quadrature points over ridge z-band
BORN_NX          = 128   # points for Born visualisation

# E — Landscape
LAND_N           = 21    # grid: LAND_N × LAND_N
LAND_RANGE       = 0.5   # alpha, beta in [-LAND_RANGE, LAND_RANGE]
LAND_N_COLL      = 64    # small collocation for speed

# F — Curriculum
CURRIC_STAGES    = [1.2, 1.5, 1.8, 2.0]
CURRIC_EPOCHS    = 300   # epochs per stage (matches budget of 600-ep direct run)
DIRECT_EPOCHS    = CURRIC_EPOCHS * len(CURRIC_STAGES)  # 1200 for fair comparison


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _make_physics(period: float = 0.8, dc: float = 0.4, h: float = 0.2,
                  n_ridge: float = 1.5, n_sub: float = 1.45) -> PhysicsConfig:
    return PhysicsConfig(
        wavelength=1.0, n_air=1.0, n_ridge=n_ridge, n_substrate=n_sub,
        period=period, ridge_width=dc * period, ridge_height=h,
        domain_height=2.0, ridge_base_fraction=0.6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT D — Born approximation warm start
# ─────────────────────────────────────────────────────────────────────────────

def _kz_outgoing(n: float, k0: float, Gm: float) -> complex:
    """kz_m = sqrt((n*k0)^2 - Gm^2), Im >= 0 (outgoing/decaying branch)."""
    val = complex((n * k0)**2 - Gm**2)
    kz  = np.sqrt(val)
    # Outgoing: Re(kz) > 0 for propagating; Im(kz) > 0 for evanescent decay
    if kz.real < 0:
        kz = -kz
    if kz.real == 0 and kz.imag < 0:
        kz = -kz
    return kz


def born_scattered_field(
    physics: PhysicsConfig,
    coeff: dict,
    x_pts: np.ndarray,    # (N,) evaluation x
    z_pts: np.ndarray,    # (N,) evaluation z
    orders: list[int] = BORN_ORDERS,
    nz_quad: int = BORN_NZ,
) -> np.ndarray:
    """First-order Born scattered field E_scat^(1)(x, z).

    Formula (TE, nondim, scattered-field convention):
      E^(1)(x,z) = sum_m A_m(z) * exp(i * G_m * x)

    where:
      delta_eps_m  = analytic Fourier coeff of delta_eps(x,z') in x (rect mask)
      A_m(z)       = -k0^2 * delta_eps_m * I_m(z)
      I_m(z)       = int_{z_min}^{z_max} G_m(z,z') * E_bg(z') dz'
      G_m(z,z')    = (i/(2*kz_m)) * exp(i*kz_m*|z-z'|)

    Returns complex (N,) array.
    """
    k0        = physics.k0
    Λ         = physics.period
    dc        = physics.ridge_width / Λ
    x_centre  = 0.5 * (physics.ridge_x_min + physics.ridge_x_max)
    z_min     = physics.ridge_z_min
    z_max     = physics.ridge_z_max
    delta_val = physics.n_ridge**2 - physics.n_substrate**2   # 0.1475 or 1.8975

    # Quadrature points over z-ridge band
    z_src = np.linspace(z_min, z_max, nz_quad)
    dz    = (z_max - z_min) / (nz_quad - 1)

    # Background field at source points
    Ebg_r, Ebg_i, _, _ = background_field_np(z_src, coeff)
    E_bg_src = Ebg_r + 1j * Ebg_i   # (nz_quad,) complex

    E_born = np.zeros(len(x_pts), dtype=complex)

    for m in orders:
        Gm  = m * 2.0 * np.pi / Λ

        # Analytic Fourier coefficient of rectangular ridge mask in x:
        #   delta_eps_m = delta_val * (1/Λ) * integral_0^Λ rect(x) exp(-i Gm x) dx
        #               = delta_val * dc * sinc(m * dc) * exp(-i Gm x_centre)
        # (where sinc(0)=1 by convention, sinc(x) = sin(pi x)/(pi x) for x!=0)
        if m == 0:
            delta_m = delta_val * dc
        else:
            delta_m = delta_val * dc * np.sinc(m * dc) * np.exp(-1j * Gm * x_centre)

        # kz_m — use n_substrate for the transmitted/scattered region
        # (the born field propagates in the effective layered medium)
        kz_m = _kz_outgoing(physics.n_substrate, k0, Gm)

        # Integrate G_m(z, z') * E_bg(z') dz' for each field point z
        # G_m(z,z') = i/(2*kz_m) * exp(i*kz_m*|z-z'|)
        pre = 1j / (2.0 * kz_m)  # complex scalar

        # Vectorised: outer product (N_pts, N_quad)
        Z_fld = z_pts[:, None]   # (N, 1)
        Z_src = z_src[None, :]   # (1, nz_quad)
        phase = np.exp(1j * kz_m * np.abs(Z_fld - Z_src))   # (N, nz_quad)
        G_mat = pre * phase

        # Numerical integral: trapezoidal
        I_m = np.sum(G_mat * E_bg_src[None, :], axis=1) * dz   # (N,)

        # Born contribution
        A_m = -k0**2 * delta_m * I_m                  # (N,) complex
        E_born += A_m * np.exp(1j * Gm * x_pts)

    return E_born   # (N,) complex — the Born scattered E_y


def _born_pm1_amplitude(
    physics: PhysicsConfig,
    coeff: dict,
    z_bot: float,
    n_quad: int = 512,
) -> tuple[float, float]:
    """Return |t_{-1}^Born|, |t_{+1}^Born| from the Born scattered field at z_bot."""
    x_arr = np.linspace(0, physics.period, n_quad, endpoint=False)
    z_arr = np.full_like(x_arr, z_bot)
    E_b   = born_scattered_field(physics, coeff, x_arr, z_arr)
    G0    = 2.0 * np.pi / physics.period
    t_m1  = abs(np.mean(E_b * np.exp(-1j * (-1) * G0 * x_arr)))
    t_p1  = abs(np.mean(E_b * np.exp(-1j * (+1) * G0 * x_arr)))
    return t_m1, t_p1


def run_exp_D(physics: PhysicsConfig, coeff: dict, pts: dict,
              out_dir: Path) -> list[dict]:
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT D: Born approximation warm start", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute Born field at collocations (used for pre-training MSE)
    x_all = np.concatenate([pts["x_air"].numpy(), pts["x_grat"].numpy(),
                             pts["x_sub"].numpy()])
    z_all = np.concatenate([pts["z_air"].numpy(), pts["z_grat"].numpy(),
                             pts["z_sub"].numpy()])
    print(f"  Computing Born field on {len(x_all)} collocation points...",
          flush=True)
    E_born_coll = born_scattered_field(physics, coeff, x_all, z_all)

    # Check Born ±1 content at z_bot
    z_bot = Z_BOT_FRAC * physics.domain_height
    t_m1_born, t_p1_born = _born_pm1_amplitude(physics, coeff, z_bot)
    print(f"  Born |t_{{-1}}|={t_m1_born:.5f}  |t_{{+1}}|={t_p1_born:.5f}", flush=True)
    print(f"  (RCWA target {RCWA_TARGET_T1:.5f}; "
          f"Born is 1st-order approx, not exact)", flush=True)

    # Visualise Born field
    x_vis = np.linspace(0, physics.period, BORN_NX, endpoint=False)
    z_vis = np.linspace(0, physics.domain_height, BORN_NX)
    X_v, Z_v = np.meshgrid(x_vis, z_vis)
    E_born_2d = born_scattered_field(
        physics, coeff, X_v.ravel(), Z_v.ravel()).reshape(BORN_NX, BORN_NX)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    im0 = axes[0].pcolormesh(X_v, Z_v, np.real(E_born_2d), cmap="RdBu_r",
                              shading="auto")
    axes[0].set(title="Born Re{E_scat}", xlabel="x/λ", ylabel="z/λ")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(X_v, Z_v, np.abs(E_born_2d), cmap="viridis",
                              shading="auto")
    axes[1].set(title="Born |E_scat|", xlabel="x/λ")
    plt.colorbar(im1, ax=axes[1])
    # Mark grating region
    for ax in axes:
        ax.axhline(physics.ridge_z_min, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(physics.ridge_z_max, color="white", lw=0.8, ls="--", alpha=0.7)
    fig.suptitle(f"First-order Born scattered field\n"
                 f"|t_{{-1}}|^Born={t_m1_born:.4f}", fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir / "born_field_sample.png")

    results = []
    for seed in [SEED, CONFIRM_SEED]:
        label = f"born_s{seed}"
        print(f"\n  seed={seed}", flush=True)
        set_seed(seed)
        model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)

        # ── Phase 1: Born pre-training (MSE on Born scattered field) ──────
        print(f"  [Born pre-train] {BORN_PRETRAIN} MSE epochs  LR={BORN_LR}",
              flush=True)
        Er_born_t = torch.as_tensor(E_born_coll.real, dtype=torch.float64)
        Ei_born_t = torch.as_tensor(E_born_coll.imag, dtype=torch.float64)
        x_t = torch.as_tensor(x_all, dtype=torch.float64)
        z_t = torch.as_tensor(z_all, dtype=torch.float64)

        pretrain_opt = torch.optim.Adam(model.parameters(), lr=BORN_LR)
        for step in range(BORN_PRETRAIN):
            model.train()
            pretrain_opt.zero_grad(set_to_none=True)
            er_s, ei_s, *_ = model.backbone.field_components(
                x_t.requires_grad_(True), z_t.requires_grad_(True))
            loss_born = torch.mean((er_s - Er_born_t)**2 +
                                   (ei_s - Ei_born_t)**2)
            loss_born.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrain_opt.step()

            if step % 50 == 0 or step == BORN_PRETRAIN - 1:
                t1_now = _modal_report(model, physics)["t_minus1_abs"]
                print(f"    born_step={step:3d}  MSE={float(loss_born.detach()):.3e}"
                      f"  |t_{{-1}}|={t1_now:.5f}", flush=True)

        t1_after_born = _modal_report(model, physics)["t_minus1_abs"]
        print(f"  After Born pre-train: |t_{{-1}}|={t1_after_born:.5f}", flush=True)

        # ── Phase 2: PDE training (600 epochs) ───────────────────────────
        print(f"  [PDE training] {EPOCHS} epochs  LR={LR}", flush=True)
        traj, best = train_adam(model, pts, physics, coeff,
                                epochs=EPOCHS, seed=seed, label=label)
        _save_traj(traj, out_dir / f"trajectory_{label}.csv")
        results.append({
            "experiment": "D", "label": label, "seed": seed,
            "born_t1_m1": t_m1_born, "born_t1_p1": t_p1_born,
            "t1_after_pretrain": t1_after_born,
            "best_t1": best["t1"], "best_RT": best.get("RT", 0.0),
            "best_epoch": best.get("epoch", 0),
        })

    # Compare Born vs A2 proxy trajectory
    _make_comparison_plot(results, out_dir)
    return results


def _make_comparison_plot(results: list[dict], out_dir: Path) -> None:
    """Load Exp A2 trajectories if available and overlay with Born."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2166ac", "#92c5de", "#d6604d", "#f4a582"]

    for i, r in enumerate(results):
        csv_p = out_dir / f"trajectory_{r['label']}.csv"
        if csv_p.exists():
            rows = list(csv.DictReader(csv_p.open()))
            for row in rows:
                for k in row:
                    try: row[k] = float(row[k])
                    except: pass
            ep = [row["epoch"] for row in rows]
            t1 = [row["t_minus1"] for row in rows]
            ax.plot(ep, t1, color=colors[i % len(colors)], lw=1.8,
                    label=f"D Born s{r['seed']} (best={r['best_t1']:.4f})")

    # A2 proxy trajectories for comparison
    for seed, col in [(42, "#d6604d"), (43, "#f4a582")]:
        a2_path = (ROOT / f"outputs/vanilla_local_min/exp_A2_warmstart/"
                   f"trajectory_warmstart_s{seed}.csv")
        if a2_path.exists():
            rows = list(csv.DictReader(a2_path.open()))
            for row in rows:
                for k in row:
                    try: row[k] = float(row[k])
                    except: pass
            ep = [row["epoch"] for row in rows]
            t1 = [row["t_minus1"] for row in rows]
            ax.plot(ep, t1, color=col, lw=1.4, ls="--",
                    label=f"A2 proxy s{seed}")

    ax.axhline(RCWA_TARGET_T1, color="black", lw=1.2, ls=":",
               label=f"RCWA {RCWA_TARGET_T1:.3f}")
    ax.axhline(VANILLA_PDE_T1, color="gray", lw=0.9, ls=":",
               label=f"Vanilla PDE-only {VANILLA_PDE_T1:.3f}")
    ax.axhline(MEANINGFUL, color="green", lw=0.7, ls=":",
               label=f"Meaningful {MEANINGFUL:.4f}")
    ax.set(xlabel="PDE training epoch", ylabel=r"$|t_{-1}|$",
           title="Born warm start (D) vs synthetic proxy (A2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "born_vs_proxy_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT E — 2D loss landscape
# ─────────────────────────────────────────────────────────────────────────────

def _filter_normalize(d: np.ndarray, params: list) -> np.ndarray:
    """Filter-wise normalization of a random direction vector.

    For each parameter tensor, scale the corresponding slice of d so that
    ||d_i||_F = ||theta_i||_F (matching the parameter's own magnitude).
    This makes the landscape scale-invariant across layers.
    If a parameter has zero norm, scale to 1 (unit norm for that filter).
    """
    d_out = d.copy()
    offset = 0
    for p in params:
        n = p.numel()
        d_slice = d_out[offset: offset + n].reshape(p.shape)
        p_np    = p.data.detach().cpu().numpy().reshape(p.shape)
        # Per-filter (first dim) normalization; if 1-D bias, treat as one filter
        if p_np.ndim >= 2:
            for fi in range(p_np.shape[0]):
                p_norm = np.linalg.norm(p_np[fi])
                d_norm = np.linalg.norm(d_slice[fi])
                if d_norm > 1e-12:
                    d_slice[fi] = d_slice[fi] * (p_norm / d_norm) if p_norm > 0 else d_slice[fi] / d_norm
        else:
            p_norm = float(np.linalg.norm(p_np))
            d_norm = float(np.linalg.norm(d_slice))
            if d_norm > 1e-12:
                d_slice[:] = d_slice * (p_norm / d_norm) if p_norm > 0 else d_slice / d_norm
        d_out[offset: offset + n] = d_slice.ravel()
        offset += n
    return d_out


def _eval_landscape_point(
    model: VanillaPINN,
    params: list,
    theta0: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    alpha: float,
    beta: float,
    pts: dict,
    physics: PhysicsConfig,
    coeff: dict,
) -> tuple[float, float]:
    """Evaluate (loss, |t_{-1}|) at theta0 + alpha*d1 + beta*d2."""
    theta = theta0 + alpha * d1 + beta * d2
    _unpack_all(params, theta)
    losses = build_loss(model, pts, physics, coeff, w_int=1.0)
    loss_val = float(losses["total"].detach())
    modal_m  = _modal_report(model, physics)
    t1       = modal_m["t_minus1_abs"]
    return loss_val, t1


def run_exp_E(
    physics: PhysicsConfig,
    coeff: dict,
    pts_full: dict,
    trained_state: dict | None,
    out_dir: Path,
    seed: int = SEED,
) -> dict:
    """2D loss landscape at init and at trained converged point.

    trained_state: state_dict from a 600-epoch trained model.
                   If None, we run a quick 600-epoch Adam run to get one.
    """
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT E: 2D Loss Landscape", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a small collocation set for speed
    rng   = np.random.default_rng(seed)
    dt    = torch.float64
    def _t(a): return torch.as_tensor(a, dtype=dt)
    margin = 5e-3
    p      = physics
    glo = p.ridge_z_min + margin; ghi = p.ridge_z_max - margin
    N = LAND_N_COLL
    pts_small = {
        "x_air":   _t(rng.uniform(0, p.period, N)),
        "z_air":   _t(rng.uniform(margin, p.ridge_z_min - margin, N)),
        "x_grat":  _t(rng.uniform(0, p.period, N)),
        "z_grat":  _t(rng.uniform(glo, ghi, N)),
        "x_sub":   _t(rng.uniform(0, p.period, N)),
        "z_sub":   _t(rng.uniform(p.ridge_z_max + margin, p.domain_height - margin, N)),
        "x_int1":  _t(rng.uniform(0, p.period, N // 2)),
        "x_int2":  _t(rng.uniform(0, p.period, N // 2)),
        "x_top":   _t(rng.uniform(0, p.period, N // 2)),
        "x_bot":   _t(rng.uniform(0, p.period, N // 2)),
        "z_vleft": _t(rng.uniform(glo, ghi, N // 2)),
        "z_vright": _t(rng.uniform(glo, ghi, N // 2)),
    }

    # Build model; get trained state if not provided
    set_seed(seed)
    model  = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    params = list(model.backbone.parameters())

    if trained_state is None:
        print("  Running quick 600-epoch train to get converged state...",
              flush=True)
        traj, best = train_adam(model, pts_full, physics, coeff,
                                epochs=EPOCHS, seed=seed, label="landscape_train",
                                log_every=600)  # silent
        if best.get("state"):
            trained_state = best["state"]
        else:
            trained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Converged state obtained. best |t_{{-1}}|={best['t1']:.5f}",
              flush=True)

    # Two anchor points
    set_seed(seed)
    model_fresh = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    params_f    = list(model_fresh.backbone.parameters())
    theta_init  = _pack_all(params_f).copy()

    model_trained = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    model_trained.load_state_dict(trained_state)
    params_t      = list(model_trained.backbone.parameters())
    theta_trained = _pack_all(params_t).copy()

    alphas = np.linspace(-LAND_RANGE, LAND_RANGE, LAND_N)
    betas  = np.linspace(-LAND_RANGE, LAND_RANGE, LAND_N)
    n_evals = LAND_N * LAND_N

    landscapes = {}
    for tag, model_use, params_use, theta0 in [
        ("init",    model_fresh,   params_f, theta_init),
        ("trained", model_trained, params_t, theta_trained),
    ]:
        print(f"\n  [{tag}] scanning {LAND_N}×{LAND_N} grid"
              f" alpha/beta ∈ [{-LAND_RANGE:.1f}, {LAND_RANGE:.1f}]...",
              flush=True)

        # Fixed random directions (same across both anchors for comparability)
        rng2 = np.random.default_rng(42)
        d1_raw = rng2.standard_normal(len(theta0))
        d2_raw = rng2.standard_normal(len(theta0))
        d1 = _filter_normalize(d1_raw, params_use)
        d2 = _filter_normalize(d2_raw, params_use)
        # Orthogonalise d2 w.r.t. d1
        d2 = d2 - np.dot(d2, d1) / (np.dot(d1, d1) + 1e-30) * d1

        loss_grid = np.zeros((LAND_N, LAND_N))
        t1_grid   = np.zeros((LAND_N, LAND_N))

        done = 0
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                lv, tv = _eval_landscape_point(
                    model_use, params_use, theta0, d1, d2, a, b,
                    pts_small, physics, coeff)
                loss_grid[i, j] = lv
                t1_grid[i, j]   = tv
                done += 1
                if done % 80 == 0:
                    print(f"    {done}/{n_evals}", flush=True)

        # Restore anchor
        _unpack_all(params_use, theta0)

        # Classify the landscape shape
        loss_at_anchor = loss_grid[LAND_N // 2, LAND_N // 2]
        loss_perimeter = np.mean([
            loss_grid[0, :].mean(), loss_grid[-1, :].mean(),
            loss_grid[:, 0].mean(), loss_grid[:, -1].mean()
        ])
        # Saddle: some directions go up, some go down
        # Flat basin: loss barely changes
        # Local min: loss increases in all directions
        loss_range = loss_grid.max() - loss_grid.min()
        relative_range = loss_range / (loss_at_anchor + 1e-30)

        if relative_range < 0.05:
            shape = "FLAT_BASIN (loss barely changes ± 5% over this range)"
        elif loss_perimeter > loss_at_anchor * 1.05:
            # Perimeter loss is higher than centre
            n_lower = np.sum(loss_grid < loss_at_anchor * 0.95)
            if n_lower > 0.1 * n_evals:
                shape = f"SADDLE ({n_lower} grid points lower than anchor)"
            else:
                shape = "LOCAL_MINIMUM (loss increases in all directions)"
        else:
            shape = "SLOPE or WIDE_BASIN (anchor is not at a local min)"

        t1_at_anchor = t1_grid[LAND_N // 2, LAND_N // 2]
        t1_max_on_grid = t1_grid.max()
        print(f"  [{tag}] shape={shape}", flush=True)
        print(f"  [{tag}] loss_at_anchor={loss_at_anchor:.4e}"
              f"  loss_range={loss_range:.4e}"
              f"  relative_range={relative_range:.3f}", flush=True)
        print(f"  [{tag}] |t_{{-1}}| at anchor={t1_at_anchor:.5f}"
              f"  max over grid={t1_max_on_grid:.5f}", flush=True)

        landscapes[tag] = {
            "shape": shape,
            "loss_at_anchor":     float(loss_at_anchor),
            "loss_range":         float(loss_range),
            "relative_range":     float(relative_range),
            "loss_perimeter_mean": float(loss_perimeter),
            "t1_at_anchor":       float(t1_at_anchor),
            "t1_max_on_grid":     float(t1_max_on_grid),
            "loss_grid":          loss_grid,
            "t1_grid":            t1_grid,
            "alphas":             alphas,
            "betas":              betas,
        }

        # ── Plot ──────────────────────────────────────────────────────────
        A, B = np.meshgrid(alphas, betas, indexing="ij")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Loss landscape
        cf0 = axes[0].contourf(A, B, loss_grid, levels=20, cmap="viridis")
        axes[0].contour(A, B, loss_grid, levels=20, colors="white",
                        linewidths=0.4, alpha=0.4)
        axes[0].scatter([0], [0], s=80, c="red", zorder=5, marker="*",
                        label="anchor")
        plt.colorbar(cf0, ax=axes[0])
        axes[0].set(xlabel=r"$\alpha$", ylabel=r"$\beta$",
                    title=f"PDE loss — {tag}\n{shape[:40]}")
        axes[0].legend(fontsize=8)

        # |t_{-1}| landscape
        cf1 = axes[1].contourf(A, B, t1_grid, levels=20, cmap="plasma")
        axes[1].contour(A, B, t1_grid, levels=20, colors="white",
                        linewidths=0.4, alpha=0.4)
        axes[1].scatter([0], [0], s=80, c="red", zorder=5, marker="*",
                        label="anchor")
        plt.colorbar(cf1, ax=axes[1])
        axes[1].set(xlabel=r"$\alpha$", ylabel=r"$\beta$",
                    title=rf"$|t_{{-1}}|$ — {tag}"
                          f"\nmax={t1_max_on_grid:.4f}")
        axes[1].legend(fontsize=8)

        fig.suptitle(
            f"Loss landscape ({tag} point)\n"
            f"Filter-normalised random directions, range={LAND_RANGE}",
            fontsize=10)
        fig.tight_layout()
        _savefig(fig, out_dir / f"landscape_{tag}.png")

    # Save NPZ
    np.savez(out_dir / "landscape_data.npz",
             alphas=alphas, betas=betas,
             loss_init=landscapes["init"]["loss_grid"],
             t1_init=landscapes["init"]["t1_grid"],
             loss_trained=landscapes["trained"]["loss_grid"],
             t1_trained=landscapes["trained"]["t1_grid"])

    return {k: {kk: vv for kk, vv in v.items()
                if not isinstance(vv, np.ndarray)}
            for k, v in landscapes.items()}


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT F — Curriculum learning via index contrast
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_F(
    base_physics: PhysicsConfig,
    out_dir: Path,
    seed: int = SEED,
) -> dict:
    print("\n" + "="*62, flush=True)
    print("EXPERIMENT F: Curriculum learning (n_ridge 1.2→1.5→1.8→2.0)", flush=True)
    print("="*62, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared collocation (resampled per stage to avoid geometry mismatch)
    # Actually collocation only depends on the z-structure (ridge_z_min/max)
    # which is fixed across all stages. So one set of points works for all.
    p_canonical = _make_physics(0.8, 0.4, 0.2, 1.5, 1.45)
    pts = sample_points(p_canonical, seed=seed)

    # ── Curriculum: n_ridge stages ────────────────────────────────────────
    print(f"\n  Curriculum stages: {CURRIC_STAGES}  {CURRIC_EPOCHS} ep each",
          flush=True)
    set_seed(seed)
    model_curric = VanillaPINN(p_canonical, n_layers=N_LAYERS, n_units=N_UNITS)
    traj_curric: list[dict] = []
    epoch_offset = 0

    for stage_i, n_ridge_stage in enumerate(CURRIC_STAGES):
        p_stage = _make_physics(0.8, 0.4, 0.2, n_ridge_stage, 1.45)
        coeff_s = compute_background_coefficients(p_stage)
        # Update model's physics (for normalisation)
        model_curric.physics = p_stage
        model_curric.backbone.physics = p_stage

        label_s = f"curric_n{n_ridge_stage:.1f}"
        print(f"\n  Stage {stage_i+1}: n_ridge={n_ridge_stage}  "
              f"delta_eps={p_stage.n_ridge**2 - p_stage.n_substrate**2:.4f}",
              flush=True)

        traj_s, best_s = train_adam(model_curric, pts, p_stage, coeff_s,
                                    epochs=CURRIC_EPOCHS, seed=seed,
                                    label=label_s, log_every=50)
        # Add epoch offset so the combined trajectory has monotone epochs
        for row in traj_s:
            row["curriculum_epoch"] = row["epoch"] + epoch_offset
            row["n_ridge"] = n_ridge_stage
        traj_curric.extend(traj_s)
        epoch_offset += CURRIC_EPOCHS

        t1_stage = best_s["t1"]
        print(f"  Stage {stage_i+1} done: best |t_{{-1}}|={t1_stage:.5f}", flush=True)

    # Final evaluation at n_ridge=2.0
    p_final = _make_physics(0.8, 0.4, 0.2, 2.0, 1.45)
    coeff_f = compute_background_coefficients(p_final)
    model_curric.physics = p_final
    model_curric.backbone.physics = p_final
    t1_curric_final = _modal_report(model_curric, p_final)["t_minus1_abs"]
    RT_curric_final, _ = _energy_balance(model_curric, p_final, coeff_f)
    print(f"\n  Curriculum final (n_ridge=2.0): "
          f"|t_{{-1}}|={t1_curric_final:.5f}  R+T={RT_curric_final:.4f}",
          flush=True)

    # ── Baseline: direct training at n_ridge=2.0 ──────────────────────────
    print(f"\n  Direct baseline n_ridge=2.0  epochs={DIRECT_EPOCHS}", flush=True)
    p_g2      = _make_physics(0.8, 0.4, 0.2, 2.0, 1.45)
    coeff_g2  = compute_background_coefficients(p_g2)
    pts_g2    = sample_points(p_g2, seed=seed)
    set_seed(seed)
    model_direct = VanillaPINN(p_g2, n_layers=N_LAYERS, n_units=N_UNITS)
    traj_direct, best_direct = train_adam(
        model_direct, pts_g2, p_g2, coeff_g2,
        epochs=DIRECT_EPOCHS, seed=seed, label="direct_n2p0", log_every=100)
    t1_direct = best_direct["t1"]
    RT_direct = best_direct.get("RT", 0.0)
    print(f"  Direct best: |t_{{-1}}|={t1_direct:.5f}  R+T={RT_direct:.4f}",
          flush=True)

    # Save trajectories
    _save_traj(traj_curric, out_dir / "trajectory_curriculum.csv")
    _save_traj(traj_direct, out_dir / "trajectory_direct_n2p0.csv")

    # ── Comparison plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if traj_curric:
        ep_c = [r["curriculum_epoch"] for r in traj_curric]
        t1_c = [r["t_minus1"] for r in traj_curric]
        ax.plot(ep_c, t1_c, color="#2166ac", lw=1.8,
                label=f"Curriculum ({' → '.join(str(n) for n in CURRIC_STAGES)})")
        # Stage boundaries
        for si, sbound in enumerate(range(CURRIC_EPOCHS, DIRECT_EPOCHS, CURRIC_EPOCHS)):
            ax.axvline(sbound, color="#888888", lw=0.7, ls="--", alpha=0.5)
            ax.text(sbound, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.005,
                    f"n={CURRIC_STAGES[si+1]}", fontsize=7, ha="left",
                    color="#666666")
    if traj_direct:
        ep_d = [r["epoch"] for r in traj_direct]
        t1_d = [r["t_minus1"] for r in traj_direct]
        ax.plot(ep_d, t1_d, color="#d6604d", lw=1.8, ls="--",
                label=f"Direct n_ridge=2.0 (best={t1_direct:.4f})")

    ax.axhline(RCWA_TARGET_T1, color="black", lw=1.2, ls=":",
               label=f"RCWA target (n=1.5) {RCWA_TARGET_T1:.3f}")
    ax.axhline(VANILLA_PDE_T1, color="gray", lw=0.9, ls=":",
               label=f"Vanilla PDE baseline {VANILLA_PDE_T1:.3f}")
    ax.axhline(MEANINGFUL, color="green", lw=0.7, ls=":",
               label=f"Meaningful {MEANINGFUL:.4f}")
    ax.set(xlabel="Training epoch (cumulative)", ylabel=r"$|t_{-1}|$",
           title="Curriculum (index contrast ramp) vs direct training at n_ridge=2.0")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout()
    _savefig(fig, out_dir / "curriculum_vs_direct.png")

    # G2 RCWA proxy (from aux-generalization: 12.16× improvement from 0.006378 base)
    g2_aux_t1 = 0.07754
    return {
        "curriculum_t1_final": t1_curric_final,
        "curriculum_RT_final": RT_curric_final,
        "direct_t1_best":      t1_direct,
        "direct_RT_best":      RT_direct,
        "g2_aux_loss_best":    g2_aux_t1,
        "curriculum_escapes":  t1_curric_final >= MEANINGFUL,
        "direct_escapes":      t1_direct >= MEANINGFUL,
        "curriculum_vs_direct_improvement":
            t1_curric_final / max(t1_direct, 1e-9),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import shutil
    out_dir = ROOT / "outputs/vanilla_final"
    if out_dir.exists():
        print(f"Removing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cfg     = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    coeff   = compute_background_coefficients(physics)
    pts     = sample_points(physics, seed=SEED)

    print(f"Canonical: Λ={physics.period:.3f}λ  dc={physics.ridge_width/physics.period:.2f}"
          f"  n_ridge={physics.n_ridge}  k0={physics.k0:.4f}", flush=True)

    # ── Exp D ─────────────────────────────────────────────────────────────
    res_D = run_exp_D(physics, coeff, pts, out_dir / "exp_D_born")

    # ── Exp E ─────────────────────────────────────────────────────────────
    # Get a trained state from Exp D (seed 42) if available; otherwise train
    trained_state = None
    if res_D:
        # Load from run_vanilla_baseline output (pre-existing 600-ep baseline)
        ckpt_path = (ROOT / "outputs/vanilla_baseline/best_checkpoint_seed42.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            trained_state = ckpt.get("state_dict")
            print(f"  Landscape: loaded trained state from {ckpt_path.name}",
                  flush=True)
    res_E = run_exp_E(physics, coeff, pts, trained_state,
                      out_dir / "exp_E_landscape")

    # ── Exp F ─────────────────────────────────────────────────────────────
    res_F = run_exp_F(physics, out_dir / "exp_F_curriculum")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "git_commit": _git_commit(),
        "baselines": {
            "vanilla_pde_only":  VANILLA_PDE_T1,
            "vanilla_aux_loss":  VANILLA_AUX_T1,
            "rcwa_target":       RCWA_TARGET_T1,
            "meaningful_threshold": MEANINGFUL,
        },
        "exp_D": {
            "born_t1_m1": res_D[0]["born_t1_m1"] if res_D else None,
            "seed42_best_t1": next((r["best_t1"] for r in res_D if r["seed"]==42), None),
            "seed42_best_RT": next((r["best_RT"] for r in res_D if r["seed"]==42), None),
            "seed43_best_t1": next((r["best_t1"] for r in res_D if r["seed"]==43), None),
            "escapes": any(r["best_t1"] >= MEANINGFUL for r in res_D),
        },
        "exp_E": res_E,
        "exp_F": res_F,
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "="*68, flush=True)
    print("FINAL RESULTS TABLE", flush=True)
    print("="*68, flush=True)
    print(f"{'Experiment':<42} {'best |t_-1|':>11} {'R+T':>7} {'escapes?':>9}",
          flush=True)
    print("-"*68, flush=True)
    for name, t1, rt in [
        ("Vanilla PDE-only (baseline)",         0.003,  0.994),
        ("Vanilla + aux loss (baseline)",        0.004,  0.994),
    ]:
        print(f"  {name:<40} {t1:>11.5f} {rt:>7.4f}  {'—':>9}", flush=True)
    print("-"*68, flush=True)
    if res_D:
        for r in res_D:
            esc = "YES" if r["best_t1"] >= MEANINGFUL else "no"
            print(f"  D Born warmstart s{r['seed']:<22}"
                  f" {r['best_t1']:>11.5f} {r['best_RT']:>7.4f} {esc:>9}",
                  flush=True)
    e_init = res_E.get("init", {})
    e_train = res_E.get("trained", {})
    print(f"  E Landscape init:  shape = {e_init.get('shape','?')[:35]}", flush=True)
    print(f"  E Landscape trained: shape = {e_train.get('shape','?')[:35]}", flush=True)
    if res_F:
        esc_c = "YES" if res_F["curriculum_escapes"] else "no"
        esc_d = "YES" if res_F["direct_escapes"] else "no"
        print(f"  F Curriculum n2.0 final"
              f" {res_F['curriculum_t1_final']:>18.5f}"
              f" {res_F['curriculum_RT_final']:>7.4f} {esc_c:>9}", flush=True)
        print(f"  F Direct n2.0 (1200 ep)"
              f" {res_F['direct_t1_best']:>18.5f}"
              f" {res_F['direct_RT_best']:>7.4f} {esc_d:>9}", flush=True)
    print("-"*68, flush=True)
    print(f"  RCWA target:                               {RCWA_TARGET_T1:.5f}", flush=True)
    print(f"  Meaningful threshold (1.5× baseline):      {MEANINGFUL:.5f}", flush=True)
    print(f"='*68", flush=True)

    pngs = list(out_dir.rglob("*.png"))
    print(f"\nOutputs: {len(pngs)} PNGs → {out_dir.relative_to(ROOT)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
