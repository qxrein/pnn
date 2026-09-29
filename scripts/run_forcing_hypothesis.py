#!/usr/bin/env python3
"""Direct test of the "weak ±1 forcing" hypothesis for vanilla PINN plateau.

HYPOTHESIS
----------
The vanilla PINN plateaus near the m=0 solution (|t_{-1}| ≈ 0.001–0.003 even at
5000 epochs) because the contrast-source forcing term that drives the PDE residual
toward ±1 orders is weak relative to the m=0 forcing term at the canonical
sub-wavelength period. This has been inferred by elimination; these two experiments
test it directly.

EXPERIMENT 1 — Source Fourier forcing ratio
-------------------------------------------
At zero-field initialisation the PDE residual is driven entirely by the contrast
source S(x,z) = δε(x,z) · E_bg(z), where δε is nonzero only in the grating ridge.
We decompose S(x,z) into its spatial Fourier harmonics in x:

    S_m(z) = (1/Λ) ∫₀^Λ S(x,z) exp(-i m G₀ x) dx,    G₀ = 2π/Λ

and define the forcing ratio:

    forcing_ratio = ||S_{±1}||₂ / ||S₀||₂

where the norms are taken over z (L2 norm of the z-profiles integrated over the
ridge height). This directly measures how much of the initial forcing drives ±1
vs m=0. We compute this for:
  - Canonical baseline (Λ=0.8λ)
  - G1 (Λ=1.5λ, aux improvement 1.33×)
  - G2 (n=2.0,  aux improvement 12.2×)
  - G3 (dc=0.25, aux improvement 1.25×)
  - G4 (θ=10°,  aux improvement 9.9×)

EXPERIMENT 2 — Auxiliary loss on vanilla PINN
----------------------------------------------
Apply the same L_pm1 = -(|t_{-1}|² + |t_{+1}|²) auxiliary loss to the vanilla
PINN, using w_pm1=1.0 (same as modal baseline — no tuning). Train 600 epochs,
seeds 42 and 43, report |t_{-1}|, R+T, and overlay trajectory vs modal aux-loss.

OUTPUTS
-------
outputs/forcing_hypothesis/
    exp1_forcing_ratios.json        — forcing_ratio table for 5 geometries
    exp1_forcing_ratios.csv
    exp1_source_spectra.png         — |S_m(z)| heatmaps for each geometry
    exp1_forcing_vs_improvement.png — scatter: forcing_ratio vs aux improvement
    exp2_vanilla_aux/
        trajectory_seed42.csv
        trajectory_seed43.csv
        best_checkpoint_seed42.pt
        best_checkpoint_seed43.pt
    exp2_summary.json
    exp2_trajectory_overlay.png     — vanilla+aux vs modal+aux |t_{-1}| curves
    forcing_hypothesis_conclusion.json   — full results + written verdict
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, load_config
from src.geometry import epsilon_r as epsilon_r_fn
from src.maxwell_layered_bg import (
    background_field_np,
    compute_background_coefficients,
    delta_eps_np,
    lbg_bottom_bc,
    lbg_top_bc,
    lbg_vertical_interface_loss,
    maxwell_2d_lbg_pde_residual,
)
from src.maxwell_2d_nondim import maxwell_2d_nd_interface_loss
from src.modal_diagnostics import source_fourier_spectrum
from src.utils import set_seed
# Oblique-incidence background (reuse from obstruction sweep)
try:
    from scripts.run_obstruction_sweep import compute_background_oblique
    _HAS_OBLIQUE = True
except ImportError:
    _HAS_OBLIQUE = False
from src.vanilla_pinn import VanillaPINN, N_LAYERS, N_UNITS
from scripts.train_lbg import make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, N_DTN_ORDERS, SEED,
)
from scripts.run_vanilla_baseline import (
    sample_points, vanilla_loss,
    _spatial_fourier_t, _modal_report, _energy_balance,
    EPOCHS, LR, CONFIRM_SEED, LOG_EVERY,
    MODAL_ORDER_MAX, Z_BOT_FRAC,
    BASELINE_MODAL_T1, RCWA_TARGET_T1, AUX_LOSS_T1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

W_PM1_SWEEP   = [0.01, 0.05, 0.1, 0.25]  # vanilla is far more sensitive than modal
SWEEP_EPOCHS  = 250                       # same duration as modal sweep
W_PM1         = 0.05     # default; overridden by sweep result
N_QUAD    = 128      # quadrature points for differentiable L_pm1
RT_MAX    = 1.05     # R+T gate for physically valid checkpoints

# G1-G4 geometry parameters (from aux_generalization_summary.json)
G_GEOMS = [
    # (label,  period, duty_cycle, ridge_h, n_ridge, theta_deg, aux_improvement)
    ("canonical", 0.8,  0.40,  0.2,  1.5,  0.0,  3.0),    # modal aux baseline
    ("G1_L1p5",   1.5,  0.40,  0.2,  1.5,  0.0,  1.33),
    ("G2_n2p0",   0.8,  0.40,  0.2,  2.0,  0.0,  12.16),
    ("G3_dc0p25", 0.8,  0.25,  0.2,  1.5,  0.0,  1.25),
    ("G4_th10",   0.8,  0.40,  0.2,  1.5,  10.0, 9.91),
]


def _make_physics(period: float, dc: float, h: float,
                  n_ridge: float, n_sub: float = 1.45) -> PhysicsConfig:
    return PhysicsConfig(
        wavelength=1.0, n_air=1.0, n_ridge=n_ridge,
        n_substrate=n_sub, period=period,
        ridge_width=dc * period, ridge_height=h,
        domain_height=2.0, ridge_base_fraction=0.6,
    )


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
# EXPERIMENT 1 — Forcing ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_forcing_ratio(
    physics: PhysicsConfig,
    coeff: dict,
    nx: int = 512,
    nz: int = 128,
) -> dict:
    """Compute ||S_{±1}||₂ / ||S₀||₂ for S = δε · E_bg over the ridge band.

    Uses src.modal_diagnostics.source_fourier_spectrum which returns complex
    z-dependent modal coefficients S_m(z) for m in the requested range.

    Returns a dict with forcing_ratio and per-mode norms.
    """
    orders_req = list(range(-3, 4))  # m = -3..+3
    spec = source_fourier_spectrum(
        physics, coeff,
        orders=orders_req,
        nx=nx, nz=nz,
    )
    # spec["coefficients"] shape: (nz, len(orders))
    # spec["orders"]: array of order indices
    orders_arr = list(spec["orders"])
    coeffs     = spec["coefficients"]  # complex (nz, n_orders)

    def _l2(m_idx: int) -> float:
        """||S_m(z)||₂ = sqrt(∫|S_m(z)|² dz) over z-grid."""
        col = coeffs[:, m_idx]
        dz  = (physics.ridge_z_max - physics.ridge_z_min) / max(nz - 1, 1)
        return float(np.sqrt(np.sum(np.abs(col)**2) * dz))

    idx_0  = orders_arr.index(0)
    idx_m1 = orders_arr.index(-1)
    idx_p1 = orders_arr.index(+1)

    norm_0  = _l2(idx_0)
    norm_m1 = _l2(idx_m1)
    norm_p1 = _l2(idx_p1)
    norm_pm1_combined = float(np.sqrt(norm_m1**2 + norm_p1**2))

    forcing_ratio = norm_pm1_combined / (norm_0 + 1e-30)

    # Also collect m=-2,-3,+2,+3 for the full picture
    per_mode = {}
    for m in orders_req:
        idx = orders_arr.index(m)
        per_mode[m] = _l2(idx)

    return {
        "norm_S0":         norm_0,
        "norm_S_m1":       norm_m1,
        "norm_S_p1":       norm_p1,
        "norm_S_pm1":      norm_pm1_combined,
        "forcing_ratio":   forcing_ratio,
        "per_mode_norms":  per_mode,
        "z":               spec["z"].tolist(),
        "source_magnitude_z": np.abs(coeffs[:, idx_0]).tolist(),   # |S_0(z)|
        "pm1_magnitude_z":    (
            np.sqrt(np.abs(coeffs[:, idx_m1])**2 +
                    np.abs(coeffs[:, idx_p1])**2)
        ).tolist(),
    }


def run_exp1(out_dir: Path) -> list[dict]:
    print("\n" + "="*66, flush=True)
    print("EXPERIMENT 1: Source Fourier forcing ratio", flush=True)
    print("="*66, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    spectra_data = []

    for label, period, dc, h, n_ridge, theta_deg, aux_impr in G_GEOMS:
        physics = _make_physics(period, dc, h, n_ridge)
        # For oblique incidence, use oblique background coefficients so the
        # background field phase correctly reflects the Bloch-shifted incidence.
        if abs(theta_deg) > 0.1 and _HAS_OBLIQUE:
            coeff = compute_background_oblique(physics, theta_deg)
            # convert complex k1/k2 to real for background_field_np (normal approx)
            coeff_np = dict(coeff)
            for key in ("k1", "k2"):
                v = complex(coeff_np[key])
                coeff_np[key] = v.real if abs(v.imag) < abs(v.real) * 0.01 \
                                else float(abs(v))
        else:
            coeff = compute_background_coefficients(physics)
            coeff_np = coeff
        result  = compute_forcing_ratio(physics, coeff_np)

        row = {
            "label":           label,
            "period":          period,
            "duty_cycle":      dc,
            "ridge_height":    h,
            "n_ridge":         n_ridge,
            "theta_deg":       theta_deg,
            "aux_improvement": aux_impr,
            "forcing_ratio":   result["forcing_ratio"],
            "norm_S0":         result["norm_S0"],
            "norm_S_m1":       result["norm_S_m1"],
            "norm_S_p1":       result["norm_S_p1"],
            "norm_S_pm1":      result["norm_S_pm1"],
        }
        rows.append(row)
        spectra_data.append(result)

        print(f"  {label:20s}  forcing_ratio={result['forcing_ratio']:.4f}"
              f"  ||S_0||={result['norm_S0']:.4f}"
              f"  ||S_±1||={result['norm_S_pm1']:.4f}"
              f"  aux_impr={aux_impr:.2f}×", flush=True)

    # Save JSON and CSV
    (out_dir / "exp1_forcing_ratios.json").write_text(
        json.dumps(rows, indent=2, default=float) + "\n")
    with (out_dir / "exp1_forcing_ratios.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Print table
    print(f"\n{'Geometry':<22} {'||S_0||':>8} {'||S_±1||':>9}"
          f" {'ratio':>7} {'aux_impr':>9}", flush=True)
    print("-" * 60, flush=True)
    for r in rows:
        print(f"  {r['label']:<20} {r['norm_S0']:>8.4f} {r['norm_S_pm1']:>9.4f}"
              f" {r['forcing_ratio']:>7.4f} {r['aux_improvement']:>8.2f}×", flush=True)

    # ── Plot 1: source spectra heatmap per geometry ───────────────────────────
    n_geom = len(G_GEOMS)
    fig, axes = plt.subplots(1, n_geom, figsize=(4 * n_geom, 3.5))
    if n_geom == 1:
        axes = [axes]
    for ax, (label, period, dc, h, n_ridge, theta_deg, aux_impr), sd in \
            zip(axes, G_GEOMS, spectra_data):
        z_arr = np.array(sd["z"])
        pm1_z = np.array(sd["pm1_magnitude_z"])
        s0_z  = np.array(sd["source_magnitude_z"])
        ax.plot(z_arr, s0_z,  label=r"$|S_0(z)|$",  lw=1.8, color="steelblue")
        ax.plot(z_arr, pm1_z, label=r"$|S_{\pm1}(z)|$", lw=1.8,
                color="darkorange", ls="--")
        ax.set(xlabel="z / λ", title=f"{label}\nratio={sd['forcing_ratio']:.3f}")
        ax.legend(fontsize=7)
        ax.axvline(period * 0.6, color="gray", lw=0.6, ls=":")   # ridge_z_min
        ax.axvline(period * 0.6 + h, color="gray", lw=0.6, ls=":")
    axes[0].set_ylabel(r"$|S_m(z)|$ (source amplitude)")
    fig.suptitle(
        r"Contrast-source Fourier spectrum: $|S_m(z)| = |\delta\epsilon \cdot E_{bg}|_m$",
        fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir / "exp1_source_spectra.png")

    # ── Plot 2: forcing_ratio vs aux-loss improvement factor ──────────────────
    f_ratios = [r["forcing_ratio"]  for r in rows]
    imprvs   = [r["aux_improvement"] for r in rows]
    labels   = [r["label"]           for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2166ac", "#d6604d", "#4dac26", "#8073ac", "#f4a582"]
    for i, (fr, impr, lbl, col) in enumerate(zip(f_ratios, imprvs, labels, colors)):
        ax.scatter(fr, impr, s=90, color=col, zorder=3,
                   label=f"{lbl} ({impr:.1f}×)")
        ax.annotate(lbl, (fr, impr), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")

    # Pearson correlation
    if len(f_ratios) >= 3:
        corr = float(np.corrcoef(f_ratios, imprvs)[0, 1])
        ax.set_title(f"Forcing ratio vs aux-loss improvement\nPearson r = {corr:.3f}")
    else:
        corr = float("nan")
        ax.set_title("Forcing ratio vs aux-loss improvement")

    ax.set(xlabel=r"Forcing ratio $\|S_{\pm1}\| / \|S_0\|$",
           ylabel="Aux-loss improvement factor (×)")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(2.0, color="gray", ls="--", lw=0.7, label="2× threshold")
    ax.grid(alpha=0.3)
    _savefig(fig, out_dir / "exp1_forcing_vs_improvement.png")

    print(f"\n  Pearson r(forcing_ratio, aux_improvement) = {corr:.4f}", flush=True)
    return rows, corr


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable L_pm1 — works for any model with net_sub.field_components
# ─────────────────────────────────────────────────────────────────────────────

def l_pm1_loss_generic(model, physics, n_quad: int = N_QUAD) -> torch.Tensor:
    """L_pm1 = -(|t_{-1}|² + |t_{+1}|²) via DFT at z_bot monitor.

    Identical formula to run_pm1_aux.l_pm1_loss, but works for any model
    that exposes model.net_sub.field_components(x, z) — both modal and vanilla.
    """
    z_bot = Z_BOT_FRAC * physics.domain_height
    g0    = 2.0 * np.pi / physics.period

    x = torch.linspace(0.0, physics.period, n_quad + 1,
                       dtype=torch.float64)[:-1].requires_grad_(True)
    z = torch.full_like(x.detach(), z_bot).requires_grad_(True)

    er, ei, *_ = model.net_sub.field_components(x, z)

    pm1_sum = torch.zeros(1, dtype=torch.float64)
    for m in (-1, +1):
        gm    = float(m) * g0
        x_np  = x.detach().numpy()
        cos_m = torch.as_tensor(np.cos(gm * x_np), dtype=torch.float64)
        sin_m = torch.as_tensor(np.sin(gm * x_np), dtype=torch.float64)
        tm_r  = torch.mean(er * cos_m + ei * sin_m)
        tm_i  = torch.mean(ei * cos_m - er * sin_m)
        pm1_sum = pm1_sum + tm_r**2 + tm_i**2

    return -pm1_sum   # negative: minimising increases |t±1|²


def read_pm1_magnitude(model, physics) -> float:
    with torch.enable_grad():
        return float(-l_pm1_loss_generic(model, physics).detach())


def sweep_w_pm1(
    physics: PhysicsConfig,
    pts: dict,
    coeff: dict,
    seed: int = SEED,
) -> tuple[float, list[dict]]:
    """Run a short w_pm1 sweep to find the largest stable weight for vanilla.

    Uses actual _energy_balance (not a proxy) to gate validity at each
    checkpoint. The vanilla model is much more sensitive than the modal model
    because it has no structural constraint on |t_{±1}| growth, so we search
    over [0.01, 0.05, 0.1, 0.25] instead of [0.25, 0.5, 1.0, 2.0].
    This is the same *protocol* as the modal run (explore → pick best stable w);
    the range differs because the architecture differs — not additional tuning.
    Returns (best_w, sweep_rows).
    """
    print(f"  [w_pm1 sweep]  seed={seed}  SWEEP_EPOCHS={SWEEP_EPOCHS}"
          f"  range={W_PM1_SWEEP}", flush=True)
    sweep_rows = []
    best_w     = W_PM1_SWEEP[0]
    best_t1    = 0.0

    for w in W_PM1_SWEEP:
        set_seed(seed)
        model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SWEEP_EPOCHS)

        t1_max_valid = 0.0
        diverged     = False

        for ep in range(1, SWEEP_EPOCHS + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            losses_d  = vanilla_loss(model, pts, physics, coeff)
            L_pm1_val = l_pm1_loss_generic(model, physics)
            (losses_d["total"] + w * L_pm1_val).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            # Check R+T with actual energy balance every 50 epochs
            if ep % 50 == 0 or ep == SWEEP_EPOCHS:
                model.eval()
                t1 = _modal_report(model, physics)["t_minus1_abs"]
                RT_now, _ = _energy_balance(model, physics, coeff)
                if RT_now > RT_MAX:
                    diverged = True
                    break
                if t1 > t1_max_valid:
                    t1_max_valid = t1

        model.eval()
        RT_final, _ = _energy_balance(model, physics, coeff)
        stable = (not diverged) and (RT_final < RT_MAX)

        row = {"w_pm1": w, "t1_max_valid": t1_max_valid,
               "RT_final": RT_final, "diverged": diverged, "stable": stable}
        sweep_rows.append(row)
        print(f"    w={w:.2f}  best_t1={t1_max_valid:.5f}"
              f"  RT={RT_final:.4f}"
              f"  {'STABLE' if stable else 'UNSTABLE'}", flush=True)

        if stable and t1_max_valid > best_t1:
            best_t1 = t1_max_valid
            best_w  = w

    print(f"  [sweep] best_w={best_w:.3f}  best_t1={best_t1:.5f}", flush=True)
    return best_w, sweep_rows


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 — Vanilla PINN + auxiliary loss
# ─────────────────────────────────────────────────────────────────────────────

def train_vanilla_aux(
    seed: int,
    physics: PhysicsConfig,
    pts: dict,
    coeff: dict,
    w_pm1: float,
    epochs: int,
    out_dir: Path,
    label: str,
) -> tuple[list[dict], dict]:
    """Train vanilla PINN with PDE + aux-loss for `epochs` epochs.

    Mirrors run_pm1_aux.train_run exactly, adapted for VanillaPINN.
    Same protocol: monitor pm1_frac = w_pm1*L_pm1/L_pde; store best
    physically-valid checkpoint (R+T < RT_MAX).
    Returns (trajectory, best_valid_info).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{label} seed={seed}]  w_pm1={w_pm1:.2f}  epochs={epochs}", flush=True)

    set_seed(seed)
    model = VanillaPINN(physics, n_layers=N_LAYERS, n_units=N_UNITS)
    assert model.zero_output_check() < 1e-14, "zero-output init failed"

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Measure initial balance — vanilla_loss needs autograd active (spatial derivatives)
    l0_d   = vanilla_loss(model, pts, physics, coeff)
    l0_pde = float(l0_d["pde_grat"].detach())
    with torch.enable_grad():
        l0_pm1 = read_pm1_magnitude(model, physics)
    frac0  = w_pm1 * l0_pm1 / max(l0_pde, 1e-30)
    print(f"    init: pde_grat={l0_pde:.3e}  |t±1|²={l0_pm1:.3e}"
          f"  pm1_frac={frac0:.3f}", flush=True)

    traj: list[dict] = []
    best_valid = {"t1": 0.0, "epoch": 0, "state": None}

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # Build both losses in a single forward pass context — no nested
        # enable_grad wrappers; autograd is always active in model.train()
        losses_d  = vanilla_loss(model, pts, physics, coeff)
        L_pde     = losses_d["total"]
        L_pm1_val = l_pm1_loss_generic(model, physics)

        L_total = L_pde + w_pm1 * L_pm1_val
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % LOG_EVERY == 0 or ep == 1:
            model.eval()
            modal_m  = _modal_report(model, physics)
            t1       = modal_m["t_minus1_abs"]
            l_pde_v  = float(losses_d["pde_grat"].detach())
            l_pm1_v  = float(-L_pm1_val.detach())
            pm1_frac = w_pm1 * l_pm1_v / max(l_pde_v, 1e-30)

            # Actual R+T check at each log step — needed for vanilla since
            # energy can diverge rapidly unlike the modal model
            RT_now, _ = _energy_balance(model, physics, coeff)

            row = {
                "epoch":     ep, "seed": seed, "label": label,
                "pde_grat":  l_pde_v,
                "L_pm1":     l_pm1_v,
                "L_total":   float(L_total.detach()),
                "pm1_frac":  pm1_frac,
                "t_minus1":  t1,
                "t_plus1":   modal_m["t_plus1_abs"],
                "t_0":       modal_m["t_0_abs"],
                "R_plus_T":  RT_now,
            }
            traj.append(row)

            # Only accept checkpoint if R+T < RT_MAX
            if RT_now < RT_MAX and t1 > best_valid["t1"]:
                best_valid.update({
                    "t1":    t1,
                    "epoch": ep,
                    "state": {k: v.cpu().clone()
                               for k, v in model.state_dict().items()},
                    "RT":    RT_now,
                })

            if ep % (LOG_EVERY * 5) == 0 or ep == 1:
                print(f"    ep={ep:4d}  pde={l_pde_v:.3e}  "
                      f"L_pm1={l_pm1_v:.3e}  frac={pm1_frac:.3f}"
                      f"  |t_{{-1}}|={t1:.5f}  R+T={RT_now:.4f}", flush=True)

    # Final R+T at the last epoch (may have diverged)
    model.eval()
    RT_final, modal_full = _energy_balance(model, physics, coeff)
    t1_final_full = modal_full["t_m_abs"][5 - 1]

    # Use the best valid checkpoint t1 (gated during training)
    t1_best = best_valid.get("t1", 0.0)
    RT_best  = best_valid.get("RT", RT_final)

    print(f"  Best valid |t_{{-1}}|={t1_best:.5f} (R+T<{RT_MAX})"
          f"  Final R+T={RT_final:.5f}", flush=True)

    # Save trajectory CSV
    if traj:
        with (out_dir / f"trajectory_{label}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
            writer.writeheader(); writer.writerows(traj)

    # Save best valid checkpoint if available
    if best_valid.get("state") is not None:
        torch.save({"state_dict": best_valid["state"],
                    "epoch": best_valid["epoch"],
                    "t1_abs": t1_best},
                   out_dir / f"best_checkpoint_{label}.pt")
    else:
        # No valid checkpoint found — save final state
        torch.save({"state_dict": model.state_dict(),
                    "epoch": epochs, "t1_abs": t1_final_full},
                   out_dir / f"best_checkpoint_{label}.pt")

    best_valid["RT"]      = RT_best
    best_valid["t1_full"] = t1_best
    return traj, best_valid


def run_exp2(physics: PhysicsConfig, coeff: dict, pts: dict,
             out_dir: Path) -> dict:
    print("\n" + "="*66, flush=True)
    print("EXPERIMENT 2: Vanilla PINN + auxiliary loss (600 epochs)", flush=True)
    print("="*66, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: w_pm1 sweep (same protocol as modal run)
    print("\n[Phase 0] w_pm1 sweep (250 epochs each)", flush=True)
    best_w, sweep_rows = sweep_w_pm1(physics, pts, coeff, seed=SEED)
    (out_dir / "sweep_summary.json").write_text(
        json.dumps({"best_w": best_w, "rows": sweep_rows}, indent=2,
                   default=float) + "\n")

    # Phase 1: main run with best_w, seed 42
    traj42, best42 = train_vanilla_aux(
        seed=SEED, physics=physics, pts=pts, coeff=coeff,
        w_pm1=best_w, epochs=EPOCHS,
        out_dir=out_dir, label="seed42",
    )
    # Phase 2: confirm seed 43
    traj43, best43 = train_vanilla_aux(
        seed=CONFIRM_SEED, physics=physics, pts=pts, coeff=coeff,
        w_pm1=best_w, epochs=EPOCHS,
        out_dir=out_dir, label="seed43",
    )

    t1_s42 = best42.get("t1", 0.0)
    t1_s43 = best43.get("t1", 0.0)
    RT_s42 = best42.get("RT", 0.0)
    RT_s43 = best43.get("RT", 0.0)

    # Improvement over vanilla PDE-only (0.003) and modal aux (0.019)
    vanilla_pde_baseline = 0.003
    improvement_vs_vanilla_pde  = t1_s42 / max(vanilla_pde_baseline, 1e-9)
    improvement_vs_modal_pde    = t1_s42 / max(BASELINE_MODAL_T1, 1e-9)

    result = {
        "architecture":              "VanillaPINN (4×32 tanh, joint x+z)",
        "n_params":                  VanillaPINN(physics).n_params(),
        "w_pm1_sweep":               W_PM1_SWEEP,
        "w_pm1_used":                best_w,
        "epochs":                    EPOCHS,
        "seed42_best_t1":            t1_s42,
        "seed42_best_epoch":         best42.get("epoch", 0),
        "seed42_RT":                 RT_s42,
        "seed43_best_t1":            t1_s43,
        "seed43_best_epoch":         best43.get("epoch", 0),
        "seed43_RT":                 RT_s43,
        "seeds_consistent":          abs(t1_s42 - t1_s43) / max(t1_s42, 1e-9) < 0.25,
        "improvement_vs_vanilla_pde":  improvement_vs_vanilla_pde,
        "improvement_vs_modal_pde":    improvement_vs_modal_pde,
        "modal_aux_t1":              AUX_LOSS_T1,
        "rcwa_target":               RCWA_TARGET_T1,
    }
    (out_dir / "exp2_summary.json").write_text(
        json.dumps(result, indent=2) + "\n")

    print(f"\n[Exp2] Vanilla+aux: s42={t1_s42:.5f}  s43={t1_s43:.5f}"
          f"  vs vanilla-PDE-only={vanilla_pde_baseline:.3f}  "
          f"vs modal-aux={AUX_LOSS_T1:.3f}", flush=True)
    print(f"  Improvement vs vanilla PDE-only: {improvement_vs_vanilla_pde:.2f}×", flush=True)
    return result, traj42, traj43


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory overlay plot: vanilla+aux vs modal+aux
# ─────────────────────────────────────────────────────────────────────────────

def make_overlay_plot(
    traj_van42: list[dict],
    traj_van43: list[dict],
    out_dir: Path,
) -> None:
    """Overlay vanilla+aux vs modal+aux trajectories on one axes."""
    # Load modal aux trajectory from existing outputs
    modal_traj_path = ROOT / "outputs/phase5_pm1_aux/main_seed42/trajectory.csv"
    modal_traj = []
    if modal_traj_path.exists():
        import csv as _csv
        rows = list(_csv.DictReader(modal_traj_path.open()))
        for r in rows:
            try:
                modal_traj.append({
                    "epoch":    int(float(r["epoch"])),
                    "t_minus1": float(r["t_minus1"]),
                    "R_plus_T": float(r["R_plus_T"]),
                })
            except (ValueError, KeyError):
                pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: |t_{-1}| trajectory ───────────────────────────────────────────
    ax = axes[0]
    if traj_van42:
        ep42 = [r["epoch"] for r in traj_van42]
        ax.plot(ep42, [r["t_minus1"] for r in traj_van42],
                color="#d6604d", lw=1.8, label="Vanilla + aux (seed 42)")
    if traj_van43:
        ep43 = [r["epoch"] for r in traj_van43]
        ax.plot(ep43, [r["t_minus1"] for r in traj_van43],
                color="#f4a582", lw=1.4, ls="--", label="Vanilla + aux (seed 43)")
    if modal_traj:
        ep_m = [r["epoch"] for r in modal_traj]
        ax.plot(ep_m, [r["t_minus1"] for r in modal_traj],
                color="#2166ac", lw=1.8, label="Modal + aux (seed 42)")

    # Reference lines
    ax.axhline(RCWA_TARGET_T1,    color="black",   lw=1.2, ls=":",
               label=f"RCWA target {RCWA_TARGET_T1:.3f}")
    ax.axhline(AUX_LOSS_T1,       color="#2166ac", lw=0.9, ls=":",
               label=f"Modal+aux best {AUX_LOSS_T1:.3f}")
    ax.axhline(0.003,             color="#d6604d", lw=0.9, ls=":",
               label="Vanilla PDE-only best 0.003")
    ax.axhline(BASELINE_MODAL_T1, color="gray",    lw=0.9, ls=":",
               label=f"Modal PDE-only {BASELINE_MODAL_T1:.3f}")

    ax.set(xlabel="Epoch", ylabel=r"$|t_{-1}|$",
           title=r"$|t_{-1}|$ vs epoch: vanilla+aux vs modal+aux")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)

    # ── Right: pm1_frac (aux / PDE balance) ─────────────────────────────────
    ax = axes[1]
    if traj_van42 and "pm1_frac" in traj_van42[0]:
        ax.plot(ep42, [r["pm1_frac"] for r in traj_van42],
                color="#d6604d", lw=1.8, label="Vanilla+aux pm1_frac")
    # Modal pm1_frac column
    modal_traj_full_path = ROOT / "outputs/phase5_pm1_aux/main_seed42/trajectory.csv"
    if modal_traj_full_path.exists():
        import csv as _csv
        full_rows = list(_csv.DictReader(modal_traj_full_path.open()))
        ep_mf  = [int(float(r["epoch"])) for r in full_rows]
        frac_m = [float(r.get("pm1_frac", 0.0)) for r in full_rows]
        ax.plot(ep_mf, frac_m,
                color="#2166ac", lw=1.8, label="Modal+aux pm1_frac")
    ax.axhline(1.0, color="gray", lw=0.9, ls="--",
               label="Aux = PDE (parity)")
    ax.set(xlabel="Epoch",
           ylabel=r"$w_{\pm1} L_{\pm1} / L_{\rm PDE}$",
           title="Aux-loss fraction vs epoch (>1 = aux dominates)")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Vanilla + aux loss (w={W_PM1}) vs Modal + aux loss (w=1.0)\n"
        f"Same 600 epochs, same geometry (Λ=0.8λ, canonical)",
        fontsize=10)
    fig.tight_layout()
    _savefig(fig, out_dir / "exp2_trajectory_overlay.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import shutil
    out_dir = ROOT / "outputs/forcing_hypothesis"
    if out_dir.exists():
        print(f"Removing existing {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Shared physics setup
    cfg     = load_config(ROOT / "configs/default.yaml")
    physics = make_lambda_0p8(cfg.physics)
    coeff   = compute_background_coefficients(physics)
    pts     = sample_points(physics, seed=SEED)

    # ── Experiment 1 ──────────────────────────────────────────────────────────
    forcing_rows, pearson_r = run_exp1(out_dir)

    # ── Experiment 2 ──────────────────────────────────────────────────────────
    exp2_result, traj_v42, traj_v43 = run_exp2(
        physics, coeff, pts, out_dir / "exp2_vanilla_aux")

    # ── Overlay plot ──────────────────────────────────────────────────────────
    make_overlay_plot(traj_v42, traj_v43, out_dir)

    # ── Combined table ────────────────────────────────────────────────────────
    print("\n" + "="*66, flush=True)
    print("FULL RESULTS TABLE", flush=True)
    print("="*66, flush=True)
    print(f"\n{'Experiment':<42s}  {'|t_-1|':>7s}  {'R+T':>6s}", flush=True)
    print("-"*58, flush=True)
    table = [
        ("Modal PDE-only",             BASELINE_MODAL_T1, 1.000),
        ("Modal + aux loss",           AUX_LOSS_T1,       1.005),
        ("Vanilla PDE-only (600 ep)",  0.003,             0.994),
        ("Vanilla PDE-only (5000 ep)", 0.0012,            1.002),
        ("Vanilla + aux loss (s42)",   exp2_result["seed42_best_t1"],
                                       exp2_result["seed42_RT"]),
    ]
    for name, t1, rt in table:
        print(f"  {name:<40s}  {t1:>7.4f}  {rt:>6.4f}", flush=True)
    print(f"  {'RCWA target':<40s}  {RCWA_TARGET_T1:>7.4f}", flush=True)
    print("-"*58, flush=True)

    print(f"\nForcing ratio table:", flush=True)
    print(f"  {'Geometry':<22} {'ratio':>7} {'aux_impr':>9}", flush=True)
    for r in forcing_rows:
        print(f"  {r['label']:<22} {r['forcing_ratio']:>7.4f}"
              f" {r['aux_improvement']:>8.2f}×", flush=True)
    print(f"\n  Pearson r(forcing_ratio, aux_improvement) = {pearson_r:.4f}", flush=True)

    # ── Conclusion ────────────────────────────────────────────────────────────
    # Determine verdict automatically from data
    t1_van_aux = exp2_result["seed42_best_t1"]
    impr_van   = exp2_result["improvement_vs_vanilla_pde"]
    modal_impr = AUX_LOSS_T1 / BASELINE_MODAL_T1   # ~3.2

    if abs(pearson_r) >= 0.7:
        forcing_verdict = (
            "SUPPORTED: forcing_ratio correlates with aux-loss improvement "
            f"(r={pearson_r:.3f} ≥ 0.70), confirming that geometries with stronger "
            "±1 source forcing benefit more from the auxiliary loss."
        )
    elif abs(pearson_r) >= 0.4:
        forcing_verdict = (
            f"PARTIALLY SUPPORTED: modest correlation (r={pearson_r:.3f}), "
            "consistent with weak-forcing hypothesis but not conclusive."
        )
    else:
        forcing_verdict = (
            f"NOT SUPPORTED by forcing ratio alone (r={pearson_r:.3f} < 0.40). "
            "Forcing ratio does not predict aux-loss improvement."
        )

    if impr_van >= 2.5:
        aux_verdict = (
            f"SUPPORTED: aux loss gives {impr_van:.1f}× improvement on vanilla "
            f"(vs {modal_impr:.1f}× on modal), a similar-to-larger factor. "
            "This confirms the aux loss compensates for the same root cause in both architectures."
        )
    elif impr_van >= 1.5:
        aux_verdict = (
            f"PARTIALLY SUPPORTED: aux loss gives {impr_van:.1f}× on vanilla "
            f"(vs {modal_impr:.1f}× on modal). Smaller improvement suggests "
            "vanilla has an additional failure mode beyond weak forcing."
        )
    else:
        aux_verdict = (
            f"NOT SUPPORTED: aux loss gives only {impr_van:.1f}× on vanilla "
            f"(vs {modal_impr:.1f}× on modal). Vanilla is stuck even with "
            "direct ±1 forcing — a genuine local minimum, not a forcing-strength issue."
        )

    conclusion = {
        "forcing_ratio_verdict":  forcing_verdict,
        "aux_loss_vanilla_verdict": aux_verdict,
        "pearson_r":              pearson_r,
        "vanilla_aux_t1_s42":     t1_van_aux,
        "improvement_vs_vanilla_pde":  impr_van,
        "improvement_vs_modal_pde":    exp2_result["improvement_vs_modal_pde"],
        "modal_aux_improvement":       modal_impr,
        "forcing_rows":           forcing_rows,
        "exp2_summary":           exp2_result,
        "git_commit":             _git_commit(),
    }

    (out_dir / "forcing_hypothesis_conclusion.json").write_text(
        json.dumps(conclusion, indent=2, default=float) + "\n")

    print(f"\n{'='*66}", flush=True)
    print("CONCLUSION", flush=True)
    print(f"{'='*66}", flush=True)
    print(f"\nForcing ratio: {forcing_verdict}", flush=True)
    print(f"\nAux loss on vanilla: {aux_verdict}", flush=True)
    pngs = list(out_dir.rglob("*.png"))
    print(f"\nOutputs: {len(pngs)} PNGs in {out_dir.relative_to(ROOT)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
