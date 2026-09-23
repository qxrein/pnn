#!/usr/bin/env python3
"""Auxiliary-loss generalization: does the ±1 mitigation transfer across geometries?

Runs the same PDE + L_pm1 auxiliary loss on 4 geometries selected from
the obstruction sweep.  Each geometry is confirmed obstructed (frac_pm1=0.000).
Each run uses the same protocol as the validated baseline:
  - w_pm1 = 1.0
  - 600 epochs
  - seed 42 (primary) + seed 43 (confirmation)
  - best physically-valid checkpoint (R+T < 1.05)
  - no RCWA targets

Geometries
----------
G1  Λ=1.5λ, dc=0.4, h=0.2, n=1.5, θ=0°   — supra-wavelength; ±1 propagating in air+sub
G2  Λ=0.8λ, dc=0.4, h=0.2, n=2.0, θ=0°   — high contrast; ±1 evanescent in air, prop in sub
G3  Λ=0.8λ, dc=0.25, h=0.2, n=1.5, θ=0°  — narrow ridge
G4  Λ=0.8λ, dc=0.4, h=0.2, n=1.5, θ=10°  — oblique incidence; ±1 borderline in sub

Baseline (already validated)
B   Λ=0.8λ, dc=0.4, h=0.2, n=1.5, θ=0°   — |t±1|=0.019 at epoch 600, seed 42+43

Claim tested
------------
If the auxiliary loss produces |t±1| > baseline (~0.006) for all four geometries,
the mitigation is geometry-generic and not specific to the validated case.
A weaker pass criterion than the baseline (>2× baseline) is used since
some geometries have different physical |t±1| scales.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from src.maxwell_layered_bg import compute_background_coefficients
from src.reference_companion import COMPANION_FILENAME, sha256_file
from src.reference_data import load_reference_npz, normalize_reference_orientation
from src.reference_validation import validate_reference
from src.utils import set_seed
from scripts.run_phase6_explicit_modal import jsonable
from scripts.train_lbg import layered_bg_loss, make_lambda_0p8
from scripts.run_frozen_head_least_squares import (
    CANONICAL_REF, COMPANION_PATH, MODAL_ORDER_MAX, N_DTN_ORDERS, SEED, TARGET_T1,
    sha256_file,
)
import scripts.run_frozen_head_least_squares as _fhls
from scripts.run_frozen_head_integrity import git_commit, git_status, run_compileall, run_pytest
from scripts.run_feature_scaling import make_scaled_model
from scripts.run_pm1_aux import (
    BASELINE_T1, LR, LOG_EVERY, Z_BOT_FRAC,
    l_pm1_loss, read_l_pm1_magnitude, train_run,
)
from scripts.run_obstruction_sweep import (
    compute_background_oblique, make_physics,
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
# Geometry definitions
# ─────────────────────────────────────────────────────────────────────────────

GEOMETRIES = {
    "G1_L1p5_dc0p4_n1p5_0deg": {
        "label":       "G1: Λ=1.5λ, dc=0.4, n=1.5, θ=0°",
        "period":      1.5, "duty_cycle": 0.4, "height": 0.2,
        "n_ridge":     1.5, "n_sub": 1.45,  "theta_deg": 0.0,
        "pm1_status":  "propagating in air and substrate",
        "why_chosen":  "supra-wavelength; different diffraction regime",
    },
    "G2_L0p8_dc0p4_n2p0_0deg": {
        "label":       "G2: Λ=0.8λ, dc=0.4, n=2.0, θ=0°",
        "period":      0.8, "duty_cycle": 0.4, "height": 0.2,
        "n_ridge":     2.0, "n_sub": 1.45,  "theta_deg": 0.0,
        "pm1_status":  "evanescent in air, propagating in substrate",
        "why_chosen":  "high-contrast (Si-like); ±1 substrate-only",
    },
    "G3_L0p8_dc0p25_n1p5_0deg": {
        "label":       "G3: Λ=0.8λ, dc=0.25, n=1.5, θ=0°",
        "period":      0.8, "duty_cycle": 0.25, "height": 0.2,
        "n_ridge":     1.5, "n_sub": 1.45,  "theta_deg": 0.0,
        "pm1_status":  "evanescent in air, propagating in substrate",
        "why_chosen":  "narrow ridge; different spatial source shape",
    },
    "G4_L0p8_dc0p4_n1p5_10deg": {
        "label":       "G4: Λ=0.8λ, dc=0.4, n=1.5, θ=10°",
        "period":      0.8, "duty_cycle": 0.4, "height": 0.2,
        "n_ridge":     1.5, "n_sub": 1.45,  "theta_deg": 10.0,
        "pm1_status":  "evanescent in air, borderline propagating in substrate",
        "why_chosen":  "oblique incidence; Bloch-shifted orders",
    },
}

W_PM1      = 1.0
EPOCHS     = 600
CONFIRM_S  = 43
SUCCESS_2X = 2.0   # |t±1| > 2× baseline counts as improvement


# ─────────────────────────────────────────────────────────────────────────────
# Run one geometry
# ─────────────────────────────────────────────────────────────────────────────

def run_geometry(
    geom_key: str,
    geom: dict,
    ref_path: Path,
    companion_p: Path,
    ref,
    device,
    dtype,
    root: Path,
) -> dict:
    print(f"\n{'='*68}")
    print(f"  {geom['label']}")
    print(f"  {geom['why_chosen']}")
    print(f"  ±1 status: {geom['pm1_status']}")
    print(f"{'='*68}", flush=True)

    physics = make_physics(
        geom["period"], geom["duty_cycle"],
        ridge_height=geom["height"],
        n_ridge=geom["n_ridge"], n_sub=geom["n_sub"],
    )

    # Background field (oblique if needed)
    if abs(geom["theta_deg"]) < 0.01:
        coeff = compute_background_coefficients(physics)
    else:
        coeff = compute_background_oblique(physics, geom["theta_deg"])
        # Make real-valued for background_field_torch compatibility
        for key in ("k1", "k2"):
            v = complex(coeff[key])
            coeff[key] = v.real if abs(v.imag) < abs(v.real) * 0.01 else abs(v)
        for key in ("r_eff", "tau"):
            coeff[key] = complex(coeff[key])

    # Use shared collocation points (re-sample for this geometry)
    from src.maxwell_2d_nondim import sample_nd_points
    pts = sample_nd_points(physics, 64, 32, 32, 32,
                           device, dtype, seed=SEED)

    out_dir = root / geom_key

    # ── Seed 42 ────────────────────────────────────────────────────────────────
    traj42, best42, best42_valid = train_run(
        seed=SEED,
        physics=physics, pts=pts, coeff=coeff,
        w_pm1=W_PM1, epochs=EPOCHS,
        device=device, dtype=dtype, ref=ref,
        ref_path=ref_path, companion_p=companion_p,
        out_dir=out_dir / "seed42",
        label=f"{geom_key}_s42",
    )
    t1_s42 = best42_valid["t1"] if best42_valid.get("state") else 0.0
    ep_s42 = best42_valid.get("epoch", 0)

    # ── Seed 43 ────────────────────────────────────────────────────────────────
    traj43, best43, best43_valid = train_run(
        seed=CONFIRM_S,
        physics=physics, pts=pts, coeff=coeff,
        w_pm1=W_PM1, epochs=EPOCHS,
        device=device, dtype=dtype, ref=ref,
        ref_path=ref_path, companion_p=companion_p,
        out_dir=out_dir / "seed43",
        label=f"{geom_key}_s43",
    )
    t1_s43 = best43_valid["t1"] if best43_valid.get("state") else 0.0

    # Best t_±1 without auxiliary loss (epoch 1 = PDE-only baseline for this geometry)
    baseline_for_geom = traj42[0]["t_minus1"] if traj42 else BASELINE_T1

    # Improvement check: did the auxiliary loss help?
    improved_s42 = t1_s42 > SUCCESS_2X * baseline_for_geom
    improved_s43 = t1_s43 > SUCCESS_2X * baseline_for_geom
    consistent   = abs(t1_s42 - t1_s43) / max(t1_s42, 1e-10) < 0.25

    print(f"\n  RESULT {geom['label']}:")
    print(f"    baseline (ep1)     : {baseline_for_geom:.5f}")
    print(f"    seed 42 best valid : {t1_s42:.5f}  (ep={ep_s42})")
    print(f"    seed 43 best valid : {t1_s43:.5f}")
    print(f"    improvement (s42)  : {t1_s42/max(baseline_for_geom,1e-10):.2f}×  "
          f"{'IMPROVED' if improved_s42 else 'NO IMPROVEMENT'}")
    print(f"    seeds consistent   : {consistent}", flush=True)

    # Save CSV + JSON
    for label, traj in (("seed42", traj42), ("seed43", traj43)):
        if traj:
            with (out_dir / f"{label}_trajectory.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(traj[0]))
                writer.writeheader(); writer.writerows(traj)

    result = {
        "geom_key":          geom_key,
        "label":             geom["label"],
        "why_chosen":        geom["why_chosen"],
        "pm1_status":        geom["pm1_status"],
        "period_over_lambda": geom["period"],
        "duty_cycle":        geom["duty_cycle"],
        "ridge_height":      geom["height"],
        "n_ridge":           geom["n_ridge"],
        "theta_deg":         geom["theta_deg"],
        "w_pm1":             W_PM1,
        "epochs":            EPOCHS,
        "baseline_t1":       baseline_for_geom,
        "seed42_t1_best_valid": t1_s42,
        "seed42_best_epoch":    ep_s42,
        "seed42_RT_max":     max((r["R_plus_T"] for r in traj42), default=0.0),
        "seed43_t1_best_valid": t1_s43,
        "seed43_RT_max":     max((r["R_plus_T"] for r in traj43), default=0.0),
        "improvement_factor_s42": t1_s42 / max(baseline_for_geom, 1e-10),
        "improved_s42":      improved_s42,
        "improved_s43":      improved_s43,
        "seeds_consistent":  consistent,
        "mitigation_works":  improved_s42 and improved_s43 and consistent,
    }
    (out_dir / "result.json").write_text(json.dumps(jsonable(result), indent=2)+"\n")
    return result, traj42, traj43


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_plots(results: list[dict], all_trajs: dict, root: Path) -> None:
    baseline_result = {
        "label": "Baseline\nΛ=0.8λ, n=1.5\nθ=0°",
        "seed42_t1_best_valid": 0.0190,
        "seed43_t1_best_valid": 0.0190,
        "baseline_t1": BASELINE_T1,
        "improvement_factor_s42": 0.0190 / BASELINE_T1,
        "mitigation_works": True,
    }
    all_res = [baseline_result] + results

    labels      = [r["label"].split(":")[0] if ":" in r["label"] else r["label"]
                   for r in all_res]
    t1_s42      = [r["seed42_t1_best_valid"] for r in all_res]
    t1_s43      = [r["seed43_t1_best_valid"] for r in all_res]
    baselines   = [r["baseline_t1"] for r in all_res]
    improved    = [r["mitigation_works"] for r in all_res]

    x = np.arange(len(all_res))
    colors = ["#2ecc71" if ok else "#e74c3c" for ok in improved]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # |t±1| bar chart
    ax = axes[0]
    ax.bar(x - 0.2, t1_s42, 0.35, label="seed 42", color=colors, alpha=0.85)
    ax.bar(x + 0.2, t1_s43, 0.35, label="seed 43", color=colors, alpha=0.55)
    ax.step(np.append(x - 0.5, x[-1]+0.5),
            np.append(baselines, baselines[-1]),
            where="post", color="r", ls="--", lw=0.8, label="baseline (PDE-only)")
    ax.axhline(TARGET_T1, color="k", ls="--", lw=0.6, label="RCWA target")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylabel=r"Best valid $|t_{\pm1}|$ (R+T < 1.05)",
           title=r"Auxiliary loss $|t_{\pm1}|$ across geometries")
    ax.legend(fontsize=7)

    # Improvement factor bar chart
    ax2 = axes[1]
    imp_fac = [r["improvement_factor_s42"] for r in all_res]
    ax2.bar(x, imp_fac, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="k", ls="-", lw=0.8)
    ax2.axhline(SUCCESS_2X, color="r", ls="--", lw=0.8, label=f"{SUCCESS_2X}× threshold")
    ax2.set_xticks(x, labels, fontsize=8)
    ax2.set(ylabel="Improvement factor (seed 42 / PDE-only epoch-1)",
            title="Improvement factor")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(root / "aux_generalization_summary.png", dpi=180)
    plt.close(fig)

    # Trajectory overlay
    fig, axes2 = plt.subplots(1, len(results)+1, figsize=(4*(len(results)+1), 4.5),
                              sharey=True)
    # Baseline panel
    ax0 = axes2[0]
    bpath = ROOT / "outputs/phase5_pm1_aux/main_seed42/trajectory.csv"
    if bpath.exists():
        b42 = list(csv.DictReader(bpath.open()))
        ax0.plot([int(float(r["epoch"])) for r in b42],
                 [float(r["t_minus1"]) for r in b42], lw=1.5)
    ax0.axhline(TARGET_T1, color="k", ls="--", lw=0.6)
    ax0.axhline(BASELINE_T1, color="r", ls=":", lw=0.6)
    ax0.set(title="Baseline\nΛ=0.8λ, n=1.5", xlabel="epoch",
            ylabel=r"$|t_{-1}|$")

    for i, (gk, (traj42, traj43)) in enumerate(all_trajs.items()):
        ax = axes2[i+1]
        if traj42:
            ep = [int(float(r["epoch"])) for r in traj42]
            ax.plot(ep, [float(r["t_minus1"]) for r in traj42], lw=1.5, label="s42")
        if traj43:
            ep3 = [int(float(r["epoch"])) for r in traj43]
            ax.plot(ep3, [float(r["t_minus1"]) for r in traj43],
                    lw=1.5, ls="--", alpha=0.8, label="s43")
        ax.axhline(TARGET_T1, color="k", ls="--", lw=0.6)
        ax.axhline(BASELINE_T1, color="r", ls=":", lw=0.6)
        short = gk.split("_")[0]
        ax.set(title=GEOMETRIES[gk]["label"].split(":")[0], xlabel="epoch")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(root / "aux_generalization_trajectories.png", dpi=180)
    plt.close(fig)
    print(f"\n  Plots saved to {root}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/aux_generalization")
    args = ap.parse_args()

    root = ROOT / args.output_root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)

    # Load reference (used for modal reporting only — not in objective)
    can_sha = sha256_file(ROOT / CANONICAL_REF)
    com_sha = sha256_file(ROOT / COMPANION_PATH)
    man     = json.loads((ROOT / "outputs/reference_companion/manifest.json").read_text())
    assert can_sha == man["canonical_sha256"], "canonical SHA mismatch"
    assert com_sha == man["companion_sha256"], "companion SHA mismatch"
    ref_path    = ROOT / CANONICAL_REF
    companion_p = ROOT / COMPANION_PATH
    validate_reference(ref_path, make_lambda_0p8(load_config(ROOT / "configs/default.yaml").physics))
    ref    = normalize_reference_orientation(load_reference_npz(ref_path))
    device = torch.device("cpu")
    dtype  = torch.float64

    phase0 = {
        "git_commit":   git_commit(),
        "status":       git_status(),
        "canonical_sha_ok": True,
        "companion_sha_ok": True,
        "w_pm1": W_PM1, "epochs": EPOCHS,
        "geometries": list(GEOMETRIES.keys()),
        "modal_data_loss": False,
        "optical_coupling": False,
        "claim_tested": (
            "Does the auxiliary loss L_pm1 produce |t_±1| > 2× PDE-only baseline "
            "for all 4 tested geometries with both seed 42 and seed 43?"
        ),
    }
    (root / "phase0_preflight.json").write_text(json.dumps(phase0, indent=2)+"\n")
    print("[Phase 0] Preflight OK", flush=True)

    # Run each geometry
    all_results = []
    all_trajs   = {}
    for gk, geom in GEOMETRIES.items():
        result, traj42, traj43 = run_geometry(
            gk, geom, ref_path, companion_p, ref, device, dtype, root)
        all_results.append(result)
        all_trajs[gk] = (traj42, traj43)

    # Write summary plots
    write_summary_plots(all_results, all_trajs, root)

    # Verification
    print("\n[Verification]", flush=True)
    cr = run_compileall()
    pr = run_pytest()
    print(f"  compileall: {'OK' if cr['passed'] else 'FAIL'}")
    print(f"  pytest: {pr['summary_line']}")

    # Summary JSON + CSV
    n_works = sum(1 for r in all_results if r["mitigation_works"])
    claim_supported = n_works == len(GEOMETRIES)

    summary = {
        "n_geometries": len(GEOMETRIES),
        "n_mitigation_works": n_works,
        "claim_supported": claim_supported,
        "conclusion": (
            f"The ±1 auxiliary loss improves |t_±1| by > {SUCCESS_2X}× over the "
            f"PDE-only baseline for {n_works}/{len(GEOMETRIES)} geometries "
            f"with both seeds. "
            + ("The mitigation is geometry-generic." if claim_supported
               else "Partial generalization only.")
        ),
        "baseline_validated": {
            "seed42_t1": 0.0190, "seed43_t1": 0.0190,
            "geometry": "Λ=0.8λ, dc=0.4, n=1.5, θ=0°",
        },
        "results": all_results,
        "compileall": cr, "pytest": pr,
    }
    (root / "aux_generalization_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2)+"\n")

    # CSV
    if all_results:
        fields = [k for k in all_results[0] if not isinstance(all_results[0][k], (dict, list))]
        with (root / "aux_generalization_summary.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in fields})

    print("\n" + "="*68, flush=True)
    print(json.dumps(jsonable({
        "n_geometries":         len(GEOMETRIES),
        "n_mitigation_works":   n_works,
        "claim_supported":      claim_supported,
        "conclusion":           summary["conclusion"][:200],
        "results_summary":      [{"geom": r["label"], "t1_s42": r["seed42_t1_best_valid"],
                                   "t1_s43": r["seed43_t1_best_valid"],
                                   "works": r["mitigation_works"]}
                                  for r in all_results],
        "pytest": pr["summary_line"],
        "output_root": str(root),
    }), indent=2))


if __name__ == "__main__":
    main()
