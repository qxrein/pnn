#!/usr/bin/env python3
"""Regenerate all paper figures with print-friendly styling.

Produces:
  outputs/paper_figures/
    fig1_pointwise_error.png          (Fig 1 — field error map)
    fig2_subspace_analysis.png        (Fig 2 — ±1 subspace, 2-panel)
    fig3_modal_amplitudes.png         (Fig 3 — modal bar chart)
    fig_trajectory.png                (supplementary — training curves)
    fig_generalization.png            (supplementary — 4-geometry aux results)
    fig_obstruction_sweep.png         (supplementary — 45-geometry obstruction)

Design principles
-----------------
- IEEE/Nature dual palette: distinguishable in full colour AND greyscale.
  Primary pair: solid black (#000000) + 60% grey (#666666)
  Secondary pair: slate blue (#2166ac) + burnt orange (#d6604d)
  These four are perceptually ordered in greyscale (black→grey→dark blue→orange).
- All bars carry hatch patterns so they are readable when printed in B&W.
- Line styles cycled: solid, dashed, dotted, dash-dot.
- Minimum linewidth 1.5 pt; minimum marker size 5 pt.
- Font sizes: title 11, axis label 10, tick 9, legend 8.
- DPI 300 for production; 180 for quick preview.
- All figures sized for a two-column IEEE/OSA layout:
    single-column : 3.4 in wide
    double-column : 7.0 in wide
    max height    : 3.5 in (single-panel) or 2.5 in per row (multi-panel)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Global print-friendly rcParams
# ─────────────────────────────────────────────────────────────────────────────

PRINT_RC = {
    # font
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    # lines
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    # grid
    "axes.grid":         True,
    "grid.alpha":        0.30,
    "grid.linewidth":    0.5,
    # save
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
}
matplotlib.rcParams.update(PRINT_RC)

# ── Dual-mode palette (colour + greyscale safe) ───────────────────────────────
C_BLACK   = "#000000"   # RCWA / reference   — greyscale: 0.00
C_GREY    = "#666666"   # baseline           — greyscale: 0.40
C_BLUE    = "#2166ac"   # PINN main          — greyscale: 0.27
C_ORANGE  = "#d6604d"   # auxiliary / aux    — greyscale: 0.45
C_GREEN   = "#4dac26"   # target             — greyscale: 0.35

# hatch patterns: bar with hatch → readable in B&W
H_RCWA  = ""        # solid fill for reference
H_PINN  = "////"    # forward hatch for PINN
H_BASE  = "xxxx"    # cross hatch for baseline
H_AUX   = "++++"    # plus hatch for aux

LS = ["-", "--", ":", "-."]   # line-style cycle


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  saved  {path.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_traj(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_sensitivity_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (index, singular_value, combined_pm1, t0_sensitivity, active_bool)."""
    rows = list(csv.DictReader(path.open()))
    idx    = np.array([int(r["singular_index"]) for r in rows])
    sigma  = np.array([float(r["singular_value"]) for r in rows])
    pm1    = np.array([float(r["combined_pm1_sensitivity"]) for r in rows])
    t0     = np.array([float(r["t0_sensitivity"]) for r in rows])
    active = np.array([r["active"] == "True" for r in rows])
    return idx, sigma, pm1, t0, active


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Pointwise field error  |E_PINN - E_RCWA|
# ─────────────────────────────────────────────────────────────────────────────

def fig1_pointwise_error(out_dir: Path) -> None:
    print("\n[Fig 1] Pointwise field error")
    npz_path = ROOT / "outputs/paper_assets/field_plots/field_comparison.npz"
    if not npz_path.exists():
        print("  field_comparison.npz not found — skipping Fig 1")
        return

    data   = np.load(npz_path)
    x, z   = data["x"], data["z"]
    diff   = data["Ey_diff"]          # |E_PINN - E_RCWA|
    X, Z   = np.meshgrid(x, z)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    im = ax.pcolormesh(X, Z, diff, cmap="Greys", shading="auto",
                       vmin=0, vmax=diff.max())
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|E_y^{\rm PINN} - E_y^{\rm RCWA}|$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Mark grating region
    cfg = _load_json(ROOT / "configs/default.yaml") if False else None
    ridge_z_min, ridge_z_max = 1.2, 1.4   # canonical geometry
    ax.axhline(ridge_z_min, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(ridge_z_max, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.text(0.02, (ridge_z_min + ridge_z_max) / 2, "grating",
            color="white", fontsize=7, va="center",
            transform=ax.get_yaxis_transform())

    ax.set(xlabel=r"$x\,/\,\lambda$", ylabel=r"$z\,/\,\lambda$",
           title=r"$|E_y^{\rm PINN} - E_y^{\rm RCWA}|$")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out_dir / "fig1_pointwise_error.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — ±1-sensitive subspace analysis  (two-panel, informative)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_subspace_analysis(out_dir: Path) -> None:
    print("\n[Fig 2] ±1-sensitive subspace analysis")
    ga_dir = ROOT / "outputs/phase5_gradient_alignment"

    sigma_all = np.load(ga_dir / "singular_values.npy")          # (1386,) full spectrum
    ga_sum    = _load_json(ga_dir / "gradient_alignment_summary.json")
    svd_meta  = ga_sum["svd"]
    phase2    = ga_sum["phase2"]

    rank      = int(svd_meta["rank"])                 # 216
    pm1_rank  = int(phase2["pm1_subspace_rank"])      # 4
    pm1_sigma = np.array(phase2["pm1_sigma_vals"])    # [5.02 x4]
    sigma_max_pm1 = float(pm1_sigma.max())            # 5.02

    idx, sigma_sens, pm1_s, t0_s, active = _load_sensitivity_csv(
        ga_dir / "modal_singular_direction_sensitivity.csv")

    # Only the active directions carry non-trivial sensitivity
    n_active = int(active.sum())   # 216

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # ── Left panel: singular-value spectrum ──────────────────────────────────
    ax = axes[0]
    x_all = np.arange(1, len(sigma_all) + 1)
    tiny  = np.finfo(float).tiny

    # Active subspace (first `rank` values)
    ax.semilogy(x_all[:rank], np.maximum(sigma_all[:rank], tiny),
                color=C_BLUE, lw=1.2, ls="-", label="Active subspace")
    # Null subspace
    ax.semilogy(x_all[rank:], np.maximum(sigma_all[rank:], tiny),
                color=C_GREY, lw=0.8, ls="-", alpha=0.5, label="Null subspace")
    # Rank boundary
    ax.axvline(rank, color=C_BLACK, lw=1.2, ls="--",
               label=f"Rank boundary ({rank})")
    # ±1 subspace σ markers
    for i, sv in enumerate(pm1_sigma):
        label = r"$\pm1$ subspace $\sigma$" if i == 0 else None
        ax.scatter([pm1_rank - i + 2], [sv], marker="*", s=80,
                   color=C_ORANGE, zorder=5, label=label)

    ax.set(xlabel="Singular direction index",
           ylabel=r"Singular value $\sigma_i$",
           title="Jacobian singular-value spectrum")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, len(sigma_all) + 10)

    # Annotate σ_max(±1)
    ax.annotate(
        rf"$\sigma_{{\max}}^{{\pm1}}={sigma_max_pm1:.2f}$",
        xy=(pm1_rank, sigma_max_pm1),
        xytext=(50, sigma_max_pm1 * 3),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ORANGE),
        color=C_ORANGE,
    )

    # ── Right panel: per-direction ±1 sensitivity ────────────────────────────
    ax = axes[1]

    # Active directions — split into ±1-sensitive (top pm1_rank) and rest
    idx_act   = idx[active]
    pm1_act   = pm1_s[active]
    t0_act    = t0_s[active]
    # Sort active by index for proper x-axis
    order     = np.argsort(idx_act)
    idx_act   = idx_act[order]
    pm1_act   = pm1_act[order]
    t0_act    = t0_act[order]

    # Null directions
    idx_null  = idx[~active]
    pm1_null  = pm1_s[~active]

    # Plot zeroth-order sensitivity (active, grey)
    ax.semilogy(idx_act + 1, np.maximum(t0_act, tiny),
                color=C_GREY, lw=1.0, ls=LS[1], alpha=0.75,
                label=r"$|t_0|$ sensitivity (active)")

    # Plot ±1 sensitivity (active, blue)
    ax.semilogy(idx_act + 1, np.maximum(pm1_act, tiny),
                color=C_BLUE, lw=1.4, ls=LS[0],
                label=r"$|t_{\pm1}|$ sensitivity (active)")

    # Null-space ±1 sensitivity floor
    if len(pm1_null):
        ax.semilogy(idx_null + 1, np.maximum(pm1_null, tiny),
                    ".", ms=1.5, color=C_GREY, alpha=0.35,
                    label="Null-space (≈0)")

    # Rank boundary
    ax.axvline(rank, color=C_BLACK, lw=1.2, ls="--", label=f"Rank boundary ({rank})")

    # Highlight the ±1-sensitive subspace (top pm1_rank directions)
    # Find which active directions have the highest pm1 sensitivity
    top_pm1_idx  = idx_act[np.argsort(pm1_act)[-pm1_rank:]]
    top_pm1_vals = pm1_act[np.argsort(pm1_act)[-pm1_rank:]]
    ax.scatter(top_pm1_idx + 1, np.maximum(top_pm1_vals, tiny),
               marker="*", s=80, color=C_ORANGE, zorder=5,
               label=rf"$\pm1$ subspace (rank {pm1_rank})")

    # Annotation box
    ax.text(0.97, 0.97,
            rf"$\|P_{{\pm1}}\,g\|/\|g\| < 10^{{-6}}$" + "\nat initialization",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=C_BLACK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.7))

    ax.set(xlabel="Singular direction index",
           ylabel=r"Sensitivity $s_{\pm1}$",
           title=r"$\pm1$ sensitivity per singular direction")
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(r"$\pm1$-sensitive subspace analysis",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig2_subspace_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Modal amplitude comparison: PINN vs RCWA
# ─────────────────────────────────────────────────────────────────────────────

def fig3_modal_amplitudes(out_dir: Path) -> None:
    print("\n[Fig 3] Modal amplitude comparison")

    csv_path = ROOT / "outputs/paper_assets/modal_plots/modal_comparison.csv"
    if not csv_path.exists():
        print("  modal_comparison.csv not found — skipping Fig 3")
        return

    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass

    orders      = [int(r["m"])          for r in rows]
    t_pinn      = [float(r["t_pinn_abs"]) for r in rows]
    t_rcwa      = [float(r["t_rcwa_abs"]) for r in rows]
    amp_err     = [float(r["amp_error"])  for r in rows]

    n        = len(orders)
    x_pos    = np.arange(n)
    bar_w    = 0.38

    fig, ax  = plt.subplots(figsize=(3.4, 3.0))

    # RCWA bars — black fill, no hatch (reference)
    b1 = ax.bar(x_pos - bar_w / 2, t_rcwa, bar_w,
                color=C_BLACK, hatch=H_RCWA,
                label="RCWA (reference)", zorder=3,
                edgecolor=C_BLACK, linewidth=0.6)

    # PINN bars — blue fill + forward hatch
    b2 = ax.bar(x_pos + bar_w / 2, t_pinn, bar_w,
                color=C_BLUE, hatch=H_PINN,
                label="PINN (ep 600)", zorder=3,
                edgecolor=C_BLACK, linewidth=0.6, alpha=0.85)

    # Annotate the ±1 error gap with a bracket/arrow
    for m_val, label in [(-1, r"$m{=}{-1}$"), (1, r"$m{=}{+1}$")]:
        idx = orders.index(m_val)
        rcwa_h = t_rcwa[idx]
        pinn_h = t_pinn[idx]
        err    = amp_err[idx]
        # small bracket from PINN top to RCWA top
        x_mid  = x_pos[idx] + bar_w / 2
        if rcwa_h > pinn_h:
            ax.annotate(
                "",
                xy=(x_mid, rcwa_h + 0.003),
                xytext=(x_mid, pinn_h + 0.001),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=C_ORANGE,
                    lw=1.2,
                    shrinkA=0, shrinkB=0,
                ),
                zorder=5,
            )
            ax.text(x_mid + 0.22, (rcwa_h + pinn_h) / 2,
                    rf"$\Delta={err:.3f}$",
                    fontsize=6.5, color=C_ORANGE, va="center", zorder=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in orders])
    ax.set(xlabel="Diffraction order $m$",
           ylabel=r"$|t_m|$",
           title="Transmitted modal amplitudes: PINN vs RCWA")

    # Highlight zeroth order recovery
    idx0   = orders.index(0)
    ax.annotate(rf"$|t_0|\approx{t_pinn[idx0]:.2f}$",
                xy=(x_pos[idx0], t_pinn[idx0]),
                xytext=(x_pos[idx0] - 0.8, t_pinn[idx0] * 0.85),
                fontsize=7, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_BLUE))

    ax.set_ylim(0, max(t_rcwa) * 1.30)
    ax.legend(loc="upper right", fontsize=7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    # Grid on y only (cleaner for bar chart)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out_dir / "fig3_modal_amplitudes.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 (supplementary) — Training trajectory: |t±1| and R+T
# ─────────────────────────────────────────────────────────────────────────────

def fig_trajectory(out_dir: Path) -> None:
    print("\n[Fig supp] Training trajectory")

    traj42 = _load_traj(ROOT / "outputs/phase5_pm1_aux/main_seed42/trajectory.csv")
    # seed 43 from paper_assets if available, else confirm_seed dir
    traj43_paths = [
        ROOT / "outputs/paper_assets/seed43_run/trajectory.csv",
        ROOT / "outputs/phase5_pm1_aux/confirm_seed43/trajectory.csv",
    ]
    traj43 = next((_load_traj(p) for p in traj43_paths if p.exists()), [])

    if not traj42:
        print("  no trajectory data — skipping")
        return

    TARGET  = 0.04941
    BASE    = 0.00638
    BEST    = 0.01901

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # ── Left: |t_{-1}| vs epoch ──────────────────────────────────────────────
    ax = axes[0]
    ep42 = [r["epoch"] for r in traj42]
    t42  = [r["t_minus1"] for r in traj42]
    ax.plot(ep42, t42, color=C_BLUE, lw=1.8, ls=LS[0], label="Seed 42")
    if traj43:
        ep43 = [r["epoch"] for r in traj43]
        ax.plot(ep43, [r["t_minus1"] for r in traj43],
                color=C_GREY, lw=1.5, ls=LS[1], label="Seed 43")

    ax.axhline(TARGET, color=C_BLACK, lw=1.2, ls=LS[2],
               label=f"RCWA target {TARGET:.4f}")
    ax.axhline(BASE,   color=C_GREY,  lw=1.0, ls=LS[3],
               label=f"PDE-only {BASE:.4f}")
    ax.axhline(BEST,   color=C_ORANGE, lw=1.2, ls=LS[1],
               label=f"Best valid {BEST:.4f}")

    ax.set(xlabel="Epoch",
           ylabel=r"$|t_{-1}|$",
           title=r"Transmitted $\pm1$ amplitude vs epoch")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, TARGET * 1.25)

    # ── Right: R+T vs epoch ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(ep42, [r["R_plus_T"] for r in traj42],
            color=C_BLUE, lw=1.8, ls=LS[0], label="Seed 42")
    if traj43:
        ax.plot(ep43, [r["R_plus_T"] for r in traj43],
                color=C_GREY, lw=1.5, ls=LS[1], label="Seed 43")

    ax.axhline(1.000, color=C_BLACK, lw=1.2, ls=LS[2], label=r"$R+T=1$ (ideal)")
    ax.axhline(1.050, color=C_ORANGE, lw=1.0, ls=LS[3], label=r"$R+T=1.05$ gate")

    ax.set(xlabel="Epoch",
           ylabel=r"$R + T$",
           title="Energy balance vs epoch")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0.99, 1.08)

    fig.suptitle(r"Auxiliary $\pm1$ loss training dynamics (w=1.0, 600 epochs)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 (supplementary) — Generalization across 4 geometries
# ─────────────────────────────────────────────────────────────────────────────

def fig_generalization(out_dir: Path) -> None:
    print("\n[Fig supp] Generalization")

    summ_path = ROOT / "outputs/aux_generalization/aux_generalization_summary.json"
    if not summ_path.exists():
        print("  aux_generalization_summary.json not found — skipping")
        return

    summ    = _load_json(summ_path)
    results = summ["results"]

    # Short labels — pure ASCII/LaTeX, no Unicode subscripts
    labels     = ["G1\n$\\Lambda$=1.5$\\lambda$", "G2\n$n$=2.0", "G3\n$d_c$=0.25", "G4\n$\\theta$=10$^\\circ$"]
    baseline   = [r["baseline_t1"] for r in results]
    best_valid = [r["seed42_t1_best_valid"] for r in results]
    factors    = [r["improvement_factor_s42"] for r in results]
    works      = [r["mitigation_works"] for r in results]

    n     = len(labels)
    x_pos = np.arange(n)
    bw    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ── Left: absolute |t_{-1}| ───────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x_pos - bw / 2, baseline,   bw, color=C_GREY,  hatch=H_BASE,
           edgecolor=C_BLACK, lw=0.6, label="PDE-only baseline", zorder=3)
    ax.bar(x_pos + bw / 2, best_valid, bw, color=C_BLUE,  hatch=H_PINN,
           edgecolor=C_BLACK, lw=0.6, alpha=0.85, label="Aux loss best valid", zorder=3)

    # RCWA target line
    ax.axhline(0.049, color=C_BLACK, lw=1.2, ls=LS[2], label=r"RCWA $|t_{-1}|=0.049$")

    # Mark geometries where mitigation works
    for i, w in enumerate(works):
        marker = "OK" if w else "--"
        colour = C_GREEN if w else C_ORANGE
        ax.text(x_pos[i], max(best_valid[i], baseline[i]) + 0.003,
                marker, ha="center", va="bottom", fontsize=10, color=colour)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel=r"$|t_{-1}|$", title=r"$|t_{-1}|$ by geometry")
    ax.legend(fontsize=7)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ── Right: improvement factor ──────────────────────────────────────────────
    ax = axes[1]
    bar_colors = [C_GREEN if w else C_ORANGE for w in works]
    bar_hatch  = [H_PINN  if w else H_BASE   for w in works]
    for i, (f, c, h, w) in enumerate(zip(factors, bar_colors, bar_hatch, works)):
        ax.bar(x_pos[i], f, 0.55, color=c, hatch=h,
               edgecolor=C_BLACK, lw=0.6, alpha=0.85, zorder=3,
               label=("Works (>2×)" if (w and i == 1) else
                      "Marginal (<2×)" if (not w and i == 0) else None))

    ax.axhline(2.0, color=C_BLACK, lw=1.2, ls=LS[2], label="2× threshold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel="Improvement factor over PDE-only",
           title="Aux loss improvement factor")
    ax.legend(fontsize=7)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Annotate exact factors
    for i, f in enumerate(factors):
        ax.text(x_pos[i], f + 0.2, f"{f:.1f}×",
                ha="center", va="bottom", fontsize=7.5, color=C_BLACK)

    fig.suptitle(
        r"Auxiliary $\pm1$ loss: generalization across geometries",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig_generalization.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 (supplementary) — Obstruction sweep summary
# ─────────────────────────────────────────────────────────────────────────────

def fig_obstruction_sweep(out_dir: Path) -> None:
    print("\n[Fig supp] Obstruction sweep")

    sweep_path = ROOT / "outputs/obstruction_sweep/sweep_results.json"
    if not sweep_path.exists():
        print("  sweep_results.json not found — skipping")
        return

    sweep   = _load_json(sweep_path)
    results = sweep["results"]

    g_norms   = np.array([r["g_norm"]   for r in results])
    frac_pm1  = np.array([r["frac_pm1"] for r in results])
    n_ridge   = np.array([r["n_ridge"]  for r in results])
    theta     = np.array([r["theta_deg"] for r in results])
    period    = np.array([r["period_over_lambda"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    # ── Left: frac_pm1 histogram ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(frac_pm1, bins=20, range=(0, 1e-4),
            color=C_BLUE, edgecolor=C_BLACK, lw=0.6)
    ax.set(xlabel=r"$\|P_{\pm1}\,g\|\,/\,\|g\|$",
           ylabel="Count",
           title=f"All 45 geometries:\nfrac_pm1 ≡ 0")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(0.95, 0.95, "45 / 45\nobstructed",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=C_ORANGE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_ORANGE, lw=0.8))

    # ── Middle: g_norm distribution ───────────────────────────────────────────
    ax = axes[1]
    ax.hist(np.log10(g_norms + 1e-30), bins=15,
            color=C_GREY, edgecolor=C_BLACK, lw=0.6)
    ax.set(xlabel=r"$\log_{10}\|g\|$",
           ylabel="Count",
           title=r"Gradient norm $\|g\|$ distribution")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    # ── Right: g_norm vs n_ridge (shows obstruction independent of contrast) ──
    ax = axes[2]
    sc = ax.scatter(n_ridge, np.log10(g_norms + 1e-30),
                    c=theta, cmap="Greys",
                    edgecolors=C_BLACK, lw=0.4, s=30, zorder=3,
                    vmin=0, vmax=45)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label(r"$\theta_{\rm inc}$ [°]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set(xlabel=r"$n_{\rm ridge}$",
           ylabel=r"$\log_{10}\|g\|$",
           title=r"$\|g\|$ vs index contrast")
    ax.yaxis.grid(True, lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    fig.suptitle(
        r"Gradient-alignment obstruction: 45 geometries, $\|P_{\pm1}\,g\|/\|g\|\equiv 0$",
        fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig_obstruction_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = ROOT / "outputs/paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    fig1_pointwise_error(out_dir)
    fig2_subspace_analysis(out_dir)
    fig3_modal_amplitudes(out_dir)
    fig_trajectory(out_dir)
    fig_generalization(out_dir)
    fig_obstruction_sweep(out_dir)

    # List all saved files
    saved = sorted(out_dir.glob("*.png"))
    print(f"\nDone — {len(saved)} figures saved to outputs/paper_figures/")
    for p in saved:
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name:45s}  {size_kb:4d} KB")


if __name__ == "__main__":
    main()
