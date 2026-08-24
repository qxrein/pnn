#!/usr/bin/env python3
"""Controlled PDE-only free-space vs layered-background Phase 5 comparison."""
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
from src.field_comparison import compare_fields, compare_modal_with_rcwa, extract_modal_amplitudes
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_feature_variants import Maxwell2DDD_Variant
from src.maxwell_layered_bg import compute_background_coefficients
from src.modal_dtn import modal_dtn_loss
from src.reference_data import interpolate_reference_to_grid, load_reference_npz, normalize_reference_orientation
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed
from scripts.train_lbg import evaluate, free_space_loss, layered_bg_loss


def _git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _jsonable(value):
    if isinstance(value, dict): return {k: _jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, complex): return {"real": value.real, "imag": value.imag}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def _model(physics, args, device, dtype):
    return Maxwell2DDD_Variant(physics, args.feature_variant, args.hidden_layers,
                               args.hidden_width, args.fourier_levels).to(device=device, dtype=dtype)


def _evaluate_modal(fields, ref_er, ref_ei, physics, formulation, reference, n_harmonics):
    dual = compare_fields(fields["E_scat_real"], fields["E_scat_imag"], ref_er, ref_ei,
                          fields["z"], physics, formulation=formulation,
                          region_mask="external_only", field_representation="scattered",
                          reference_field_representation="total")
    x1d, z1d = fields["x"][0, :], fields["z"][:, 0]
    total = dual["pinn_E_total_r"] + 1j * dual["pinn_E_total_i"]
    modal = extract_modal_amplitudes(total, x1d, z1d, physics, n_orders=1,
                                     formulation=formulation, field_representation="total")
    modal_compare = compare_modal_with_rcwa(modal, str(reference), n_harmonics, physics=physics)
    return dual, modal, modal_compare


def _modal_row(epoch, losses, modal):
    values = {m: i for i, m in enumerate(modal["orders"])}
    return {
        "epoch": epoch, "pde_loss": losses["pde"], "total_loss": losses["total"],
        "energy_balance": modal["energy_check"],
        "r0_real": complex(modal["r_m_complex"][values[0]]).real,
        "r0_imag": complex(modal["r_m_complex"][values[0]]).imag,
        "t0_real": complex(modal["t_m_complex"][values[0]]).real,
        "t0_imag": complex(modal["t_m_complex"][values[0]]).imag,
        "tm1_real": complex(modal["t_m_complex"][values[-1]]).real,
        "tm1_imag": complex(modal["t_m_complex"][values[-1]]).imag,
        "tp1_real": complex(modal["t_m_complex"][values[1]]).real,
        "tp1_imag": complex(modal["t_m_complex"][values[1]]).imag,
    }


def _save_figures(directory, fields, reference_er, reference_ei, modal_history, label):
    directory.mkdir(parents=True, exist_ok=True)
    total = fields["E_real"] + 1j * fields["E_imag"]
    ref = reference_er + 1j * reference_ei
    extent = [fields["x"].min(), fields["x"].max(), fields["z"].max(), fields["z"].min()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, data, title in zip(axes, [np.abs(total), np.abs(ref), np.abs(total-ref)],
                               ["PINN |E_total|", "RCWA |E_total|", "|PINN − RCWA|"]):
        im = ax.imshow(data, extent=extent, origin="upper", aspect="auto", cmap="viridis")
        ax.set_title(f"{label}: {title}"); ax.set_xlabel("x"); ax.set_ylabel("z")
        fig.colorbar(im, ax=ax)
    fig.savefig(directory / "field_comparison.png", dpi=150); plt.close(fig)
    epochs = [r["epoch"] for r in modal_history]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in ("r0", "t0", "tm1", "tp1"):
        vals = [abs(complex(r[f"{name}_real"], r[f"{name}_imag"])) for r in modal_history]
        ax.plot(epochs, vals, marker="o", label=name)
    ax.set(xlabel="epoch", ylabel="modal amplitude magnitude", title=f"{label}: modal amplitudes")
    ax.legend(); fig.savefig(directory / "modal_amplitudes_vs_epoch.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, [r["energy_balance"] for r in modal_history], marker="o")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="epoch", ylabel="R + T", title=f"{label}: energy balance")
    fig.savefig(directory / "energy_balance_vs_epoch.png", dpi=150); plt.close(fig)


def _run_one(directory, name, physics, args, pts, device, dtype, coeff, reference, ref_er, ref_ei, n_harmonics, config_hash):
    directory.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    model = _model(physics, args, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = directory / "best_checkpoint.pt"
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history, modal_history = [], []
    formulation = "free_space" if name == "free_space" else "layered_bg"

    def loss_fn():
        if formulation == "free_space":
            losses = free_space_loss(model, pts, physics, args.w_pde, args.w_e, args.w_h,
                                     args.w_top, args.w_bot)
            if args.use_dtn:
                losses["top"] = modal_dtn_loss(model.net_air, pts["x_top"], 0.0, physics,
                                                 "top", physics.n_air, args.n_dtn_orders)
                losses["bottom"] = modal_dtn_loss(model.net_sub, pts["x_bot"], physics.domain_height,
                                                    physics, "bottom", physics.n_substrate,
                                                    args.n_dtn_orders)
                losses["total"] = (args.w_pde * losses["pde"]
                                   + args.w_e * (losses["E_int1"] + losses["E_int2"])
                                   + args.w_h * (losses["H_int1"] + losses["H_int2"])
                                   + args.w_top * losses["top"] + args.w_bot * losses["bottom"])
            return losses
        return layered_bg_loss(model, pts, physics, coeff, args.w_pde, args.w_e, args.w_h,
                               args.w_top, args.w_bot, use_dtn=args.use_dtn, n_dtn_orders=args.n_dtn_orders,
                               rcwa_amps=None, w_modal=0.0, n_modal_orders=args.n_modal_orders)

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        row = {k: float(v.detach()) for k, v in losses.items()}; row["epoch"] = epoch
        if row["total"] < best_loss:
            best_loss, best_epoch = row["total"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0 or epoch == 1:
            history.append(row)
        if epoch % args.modal_log_interval == 0 or epoch == args.epochs:
            fields = evaluate(model, physics, device, dtype, formulation, coeff if formulation == "layered_bg" else None)
            _, modal, _ = _evaluate_modal(fields, ref_er, ref_ei, physics, formulation, reference, n_harmonics)
            modal_history.append(_modal_row(epoch, row, modal))
            print(f"[{name}] epoch={epoch} pde={row['pde']:.3e} total={row['total']:.3e} R+T={modal['energy_check']:.6f}")
    torch.save({"state_dict": best_state, "epoch": best_epoch, "config_hash": config_hash}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.eval()
    with torch.enable_grad():
        reproduced_losses = {k: float(v.detach()) for k, v in loss_fn().items()}
    fields = evaluate(model, physics, device, dtype, formulation, coeff if formulation == "layered_bg" else None)
    dual, modal, modal_compare = _evaluate_modal(fields, ref_er, ref_ei, physics, formulation, reference, n_harmonics)
    _save_figures(directory, fields, ref_er, ref_ei, modal_history, name)
    with (directory / "modal_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(modal_history[0])); writer.writeheader(); writer.writerows(modal_history)
    result = {
        "valid": False,
        "invalid_reason": "Phase 5 requires manual threshold review; no physical validation is inferred from training loss.",
        "formulation": formulation, "boundary_type": "dtn" if args.use_dtn else "robin", "modal_loss_weight": 0.0,
        "checkpoint": str(checkpoint), "checkpoint_epoch": best_epoch,
        "checkpoint_reloaded": True, "best_training_loss": best_loss,
        "reproduced_checkpoint_losses": reproduced_losses,
        "dual_comparison": _jsonable(dual), "modal_pinn": _jsonable(modal),
        "modal_rcwa_vs_pinn": _jsonable(modal_compare), "modal_history": modal_history,
    }
    (directory / "result.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-per-region", type=int, default=256)
    parser.add_argument("--n-interface", type=int, default=128)
    parser.add_argument("--n-bc", type=int, default=128)
    parser.add_argument("--modal-log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-layers", type=int, default=4); parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--fourier-levels", type=int, default=4); parser.add_argument("--feature-variant", default="global_k0")
    parser.add_argument("--w-pde", type=float, default=1.0); parser.add_argument("--w-e", type=float, default=500.0)
    parser.add_argument("--w-h", type=float, default=500.0); parser.add_argument("--w-top", type=float, default=200.0); parser.add_argument("--w-bot", type=float, default=100.0)
    parser.add_argument("--n-dtn-orders", type=int, default=8); parser.add_argument("--n-modal-orders", type=int, default=3)
    parser.add_argument("--use-dtn", action="store_true")
    parser.add_argument("--output-dir", default="outputs/phase5_ab")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute(): out = ROOT / out
    if out.exists(): raise FileExistsError(f"Refusing to overwrite existing {out}")
    config = load_config(ROOT / "configs/default.yaml")
    physics = config.physics
    physics.period = 0.8 * physics.wavelength; physics.ridge_width = 0.4 * physics.period
    device = resolve_device("cpu"); dtype = resolve_training_dtype(config.training.dtype, device)
    reference = ROOT / "outputs/reference_lambda_0p8.npz"; ref = normalize_reference_orientation(load_reference_npz(reference))
    xg, zg = np.meshgrid(np.linspace(0, physics.period, physics.nx_visualization), np.linspace(0, physics.domain_height, physics.nz_visualization))
    ref_er, ref_ei, _ = interpolate_reference_to_grid(ref, xg[0], zg[:, 0])
    coeff = compute_background_coefficients(physics); n_harmonics = 75
    configuration = {**vars(args), "case": "lambda_0p8", "reference": str(reference), "device": "cpu", "dtype": str(dtype), "boundary_type": "dtn" if args.use_dtn else "robin", "modal_loss_weight": 0.0}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True); set_seed(args.seed)
    pts = sample_nd_points(physics, args.n_per_region, args.n_interface, args.n_bc, args.n_bc, device, dtype, seed=args.seed)
    free = _run_one(out / "free_space", "free_space", physics, args, pts, device, dtype, coeff, reference, ref_er, ref_ei, n_harmonics, config_hash)
    layered = _run_one(out / "layered_bg", "layered_bg", physics, args, pts, device, dtype, coeff, reference, ref_er, ref_ei, n_harmonics, config_hash)
    report = {"valid": False, "invalid_reason": "Physics thresholds have not been met or reviewed.", "configuration": configuration,
              "configuration_hash": config_hash, "test_commit_hash": _git_hash(),
              "monitor_positions": {"top": 0.16, "bottom": 1.84}, "reference_planes": {"reflection": 0.0, "transmission": 1.4},
              "free_space": free, "layered_bg": layered}
    (out / "phase5_ab_report.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    rows = []
    for name, result in [("free_space", free), ("layered_bg", layered)]:
        d, m = result["dual_comparison"], result["modal_pinn"]
        rows.append({"formulation": name, "checkpoint_epoch": result["checkpoint_epoch"], "total_complex_l2": d["total/complex_l2"],
                     "scattered_complex_l2": d["scattered/complex_l2"], "magnitude_l2": d["total/magnitude_l2"],
                     "phase_rmse_deg": d["total/phase_rmse_deg"], "R_total": m["R_total"], "T_total": m["T_total"], "energy_balance": m["energy_check"]})
    with (out / "phase5_ab_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return 0


if __name__ == "__main__": raise SystemExit(main())
