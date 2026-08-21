#!/usr/bin/env python3
"""Boundary diagnostic: compare PINN predictions vs. prescribed boundary targets.

Loads a trained checkpoint and evaluates:
- Top boundary (z=0): target = E_inc = 1 + 0i
- Bottom boundary (z=domain_height): Robin residual (∂E/∂z + i k_sub E)
- Periodic boundary: left vs right mismatch

Saves per-boundary figures and a JSON report.

Usage::

    .venv/bin/python scripts/boundary_diagnostic.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --output-dir outputs/diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.boundary_conditions import boundary_diagnostic
from src.config import load_config
from src.evaluate import load_checkpoint
from src.utils import detach_numpy, resolve_device, resolve_training_dtype


def main():
    parser = argparse.ArgumentParser(description="Boundary condition diagnostic")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--n-points", type=int, default=512)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.training.device)
    dtype = resolve_training_dtype(config.training.dtype, device)
    model, meta = load_checkpoint(ckpt_path, device)
    physics = config.physics

    N = args.n_points
    z_top_np = np.zeros(N)
    x_top_np = np.linspace(0.0, physics.period, N)
    z_bot_np = np.full(N, physics.domain_height)
    x_bot_np = np.linspace(0.0, physics.period, N)

    def _t(arr):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    # ---- Top boundary ----
    diag_top = boundary_diagnostic(
        model, _t(x_top_np), _t(z_top_np), physics, "top"
    )
    pred_r_top = detach_numpy(diag_top["pred_real"])
    pred_i_top = detach_numpy(diag_top["pred_imag"])
    tgt_r_top  = detach_numpy(diag_top["target_real"])
    tgt_i_top  = detach_numpy(diag_top["target_imag"])

    top_mse_real = float(np.mean((pred_r_top - tgt_r_top)**2))
    top_mse_imag = float(np.mean((pred_i_top - tgt_i_top)**2))
    top_mse = top_mse_real + top_mse_imag

    # ---- Bottom boundary (Robin residual) ----
    from src.derivatives import first_derivative
    k_sub = physics.k0 * physics.n_substrate
    xr = _t(x_bot_np).detach().clone().requires_grad_(True)
    zr = _t(z_bot_np).detach().clone().requires_grad_(True)
    model.train()  # need grad
    e_re_bot, e_im_bot = model.field_components(xr, zr)
    dEr_dz = first_derivative(e_re_bot, zr)
    dEi_dz = first_derivative(e_im_bot, zr)
    robin_r = detach_numpy(dEr_dz - k_sub * e_im_bot)
    robin_i = detach_numpy(dEi_dz + k_sub * e_re_bot)
    bot_mse = float(np.mean(robin_r**2 + robin_i**2))
    model.eval()

    # ---- Periodic boundary ----
    z_per_np = np.linspace(0.0, physics.domain_height, N)
    x_left_np = np.zeros(N)
    x_right_np = np.full(N, physics.period)
    with torch.no_grad():
        e_left  = detach_numpy(model(_t(x_left_np), _t(z_per_np)))
        e_right = detach_numpy(model(_t(x_right_np), _t(z_per_np)))
    per_mse = float(np.mean((e_left - e_right)**2))

    # ---- Convention printout ----
    print("\n=== Coordinate and convention diagnostics ===")
    print(f"  x range:              [0, {physics.period}] (physical)")
    print(f"  z range:              [0, {physics.domain_height}] (physical, z=0 is top)")
    print(f"  k0*x range:           [0, {physics.k0 * physics.period:.4f}] (nondim)")
    print(f"  k0*z range:           [0, {physics.k0 * physics.domain_height:.4f}] (nondim)")
    print(f"  wavelength:           {physics.wavelength}")
    print(f"  k0:                   {physics.k0:.6f}")
    print(f"  k_substrate:          {k_sub:.6f}")
    print(f"  Grating ridge z:      [{physics.ridge_z_min}, {physics.ridge_z_max}]")
    print(f"  Grating ridge x:      [{physics.ridge_x_min}, {physics.ridge_x_max}]")
    print(f"  Time convention:      exp(+iωt) suppressed")
    print(f"  Propagation:          E_inc = exp(-i k0 z)  (downward, +z)")
    print(f"  At z=0:               E_inc = 1 + 0i")
    print()
    print("=== Boundary loss diagnostics ===")
    print(f"  Top BC MSE (total):   {top_mse:.6e}")
    print(f"    Re component:       {top_mse_real:.6e}  (target = cos(k0*z) ≈ 1.0 at z≈0)")
    print(f"    Im component:       {top_mse_imag:.6e}  (target = -sin(k0*z) ≈ 0.0 at z≈0)")
    print(f"  Bottom Robin MSE:     {bot_mse:.6e}  (∂E/∂z + i k_sub E should = 0)")
    print(f"  Periodic MSE:         {per_mse:.6e}  (E(0,z) = E(period,z))")
    print()

    # ---- Figures ----
    # Top BC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x_top_np, tgt_r_top, "b-", label="Target Re{E}")
    axes[0].plot(x_top_np, pred_r_top, "r--", label="PINN Re{E}")
    axes[0].set_xlabel("x"); axes[0].set_title(f"Top BC  Re{{E}}  MSE={top_mse_real:.3e}")
    axes[0].legend()
    axes[1].plot(x_top_np, tgt_i_top, "b-", label="Target Im{E}")
    axes[1].plot(x_top_np, pred_i_top, "r--", label="PINN Im{E}")
    axes[1].set_xlabel("x"); axes[1].set_title(f"Top BC  Im{{E}}  MSE={top_mse_imag:.3e}")
    axes[1].legend()
    fig.suptitle("Top boundary (z=0): PINN vs. target")
    fig.savefig(output_dir / "diag_top_bc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bottom Robin residual
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x_bot_np, robin_r, "r-")
    axes[0].set_xlabel("x"); axes[0].set_title("Robin residual Re: ∂E_r/∂z − k_sub E_i")
    axes[1].plot(x_bot_np, robin_i, "r-")
    axes[1].set_xlabel("x"); axes[1].set_title("Robin residual Im: ∂E_i/∂z + k_sub E_r")
    fig.suptitle(f"Bottom BC Robin residual  MSE={bot_mse:.3e}")
    fig.savefig(output_dir / "diag_bottom_bc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Periodic boundary mismatch
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(z_per_np, e_left[:, 0], "b-", label="x=0  Re{E}")
    axes[0].plot(z_per_np, e_right[:, 0], "r--", label="x=period  Re{E}")
    axes[0].set_xlabel("z"); axes[0].legend()
    axes[1].plot(z_per_np, e_left[:, 1], "b-", label="x=0  Im{E}")
    axes[1].plot(z_per_np, e_right[:, 1], "r--", label="x=period  Im{E}")
    axes[1].set_xlabel("z"); axes[1].legend()
    fig.suptitle(f"Periodic BC  MSE={per_mse:.3e}")
    fig.savefig(output_dir / "diag_periodic_bc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- JSON report ----
    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": meta.get("epoch"),
        "top_bc_mse": top_mse,
        "top_bc_mse_real": top_mse_real,
        "top_bc_mse_imag": top_mse_imag,
        "bottom_robin_mse": bot_mse,
        "periodic_mse": per_mse,
        "k0": physics.k0,
        "k_substrate": float(k_sub),
        "convention": "E_inc = exp(-i k0 z), time: exp(+iωt) suppressed",
    }
    report_path = output_dir / "boundary_diagnostic.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")
    print(f"  Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()
