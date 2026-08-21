#!/usr/bin/env python3
"""Train a 2-D first-order Maxwell PINN for the binary diffraction grating.

Uses the validated first-order Maxwell formulation from benchmark_maxwell_1d.py.
The 1-D layered benchmark (test_maxwell_1d.py) showed complex-field error < 0.02%,
confirming this formulation is reliable.

Physics
-------
TE polarisation, scalar 2-D problem (E_y, H̃_x, H̃_z).
Network outputs: [Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z].
PDE:
    ∂Er/∂z -  k0*Hi_x = 0
    ∂Ei/∂z +  k0*Hr_x = 0
    ∂Er/∂x +  k0*Hi_z = 0
    ∂Ei/∂x -  k0*Hr_z = 0
    ∂Hr_x/∂z - ∂Hr_z/∂x - k0*εr*Ei = 0
    ∂Hi_x/∂z - ∂Hi_z/∂x + k0*εr*Er = 0

Validation
----------
After training, compare against the RCWA reference field (outputs/reference_grating.npz)
using relative complex L2 error, magnitude error, and phase RMSE.

Usage::

    .venv/bin/python scripts/train_maxwell_grating.py \\
        --config configs/default.yaml

    .venv/bin/python scripts/train_maxwell_grating.py \\
        --config configs/default.yaml \\
        --reference outputs/reference_grating.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, save_config_snapshot
from src.geometry import epsilon_r_grid
from src.maxwell_2d import (
    Maxwell2DMLP,
    maxwell_2d_total_loss,
    sample_collocation_points,
)
from src.reference_data import run_comparison
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed
from src.benchmarks import evaluate_benchmark_errors


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_maxwell_grating(config, output_dir: Path, device_str: str = "auto"):
    set_seed(config.training.seed)
    device = resolve_device(device_str or config.training.device)
    dtype  = resolve_training_dtype(config.training.dtype, device)
    physics = config.physics

    print(f"  Device: {device}  dtype: {dtype}")
    print(f"  k0 = {physics.k0:.6f}")
    print(f"  Domain: x ∈ [0, {physics.period}],  z ∈ [0, {physics.domain_height}]")
    print(f"  Grating ridge: x ∈ [{physics.ridge_x_min:.3f}, {physics.ridge_x_max:.3f}],  "
          f"z ∈ [{physics.ridge_z_min:.3f}, {physics.ridge_z_max:.3f}]")

    model = Maxwell2DMLP(
        physics,
        hidden_layers=config.model.hidden_layers,
        hidden_width=config.model.hidden_width,
        num_fourier_levels=max(1, config.model.num_fourier_features // 4),
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Weights — PDE residual is the primary objective
    w_pde = config.loss_weights.pde
    w_top = config.loss_weights.top
    w_bot = config.loss_weights.bottom
    w_per = config.loss_weights.periodic
    print(f"  Loss weights: pde={w_pde}  top={w_top}  bot={w_bot}  per={w_per}")

    # Sample collocation points once (resample every few epochs for generality)
    pts = sample_collocation_points(
        physics,
        n_interior=config.sampling.n_interior,
        n_top=config.sampling.n_top,
        n_bottom=config.sampling.n_bottom,
        n_periodic=config.sampling.n_periodic,
        device=device, dtype=dtype,
        seed=config.training.seed,
    )
    print(f"  Interior points: {pts['x_int'].shape[0]}  "
          f"Top: {pts['x_top'].shape[0]}  "
          f"Bot: {pts['x_bot'].shape[0]}  "
          f"Per: {pts['x_left'].shape[0]}")

    opt   = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.training.epochs)

    history = []
    best_loss = float("inf")
    best_state = None
    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resample_interval = 500  # resample interior points every N epochs

    for ep in range(1, config.training.epochs + 1):

        # Resample interior points periodically to improve coverage
        if ep % resample_interval == 0:
            pts = sample_collocation_points(
                physics,
                n_interior=config.sampling.n_interior,
                n_top=config.sampling.n_top,
                n_bottom=config.sampling.n_bottom,
                n_periodic=config.sampling.n_periodic,
                device=device, dtype=dtype,
                seed=config.training.seed + ep,
            )

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = maxwell_2d_total_loss(model, pts, physics, w_pde, w_top, w_bot, w_per)
        losses["total"].backward()
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        opt.step()
        sched.step()

        if ep % config.training.validation_interval == 0 or ep == 1:
            row = {
                "epoch":    ep,
                "pde":      float(losses["pde"].detach()),
                "top":      float(losses["top"].detach()),
                "bottom":   float(losses["bottom"].detach()),
                "periodic": float(losses["periodic"].detach()),
                "total":    float(losses["total"].detach()),
            }
            history.append(row)
            if ep % (config.training.validation_interval * 5) == 0:
                print(f"  ep={ep:6d}  pde={row['pde']:.3e}  "
                      f"top={row['top']:.3e}  bot={row['bottom']:.3e}  "
                      f"per={row['periodic']:.3e}  total={row['total']:.3e}")
            if row["total"] < best_loss:
                best_loss = row["total"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % config.training.checkpoint_interval == 0:
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "config": config}, ckpt_dir / f"maxwell_ep{ep:06d}.pt")

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = ckpt_dir / "maxwell_best.pt"
    torch.save({"epoch": config.training.epochs, "model_state_dict": model.state_dict(),
                "config": config, "best_loss": best_loss}, ckpt_path)
    print(f"  Best total loss: {best_loss:.4e}")
    print(f"  Checkpoint: {ckpt_path}")

    # Save history
    hist_path = Path(config.paths.history_file).parent / "maxwell_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    return model, history, ckpt_path


# ---------------------------------------------------------------------------
# Evaluation on grid
# ---------------------------------------------------------------------------


def evaluate_maxwell_grating(model, config, device, dtype):
    """Evaluate on the visualization grid and return field arrays."""
    physics = config.physics
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        out = model.forward(x_flat, z_flat)

    shape = x_grid.shape
    Er  = detach_numpy(out[:, 0].reshape(shape))
    Ei  = detach_numpy(out[:, 1].reshape(shape))
    Hrx = detach_numpy(out[:, 2].reshape(shape))
    Hix = detach_numpy(out[:, 3].reshape(shape))
    Hrz = detach_numpy(out[:, 4].reshape(shape))
    Hiz = detach_numpy(out[:, 5].reshape(shape))

    magnitude = np.sqrt(Er**2 + Ei**2)
    phase     = np.arctan2(Ei, Er)
    intensity = magnitude**2
    eps_np    = detach_numpy(eps_grid)
    x_np      = detach_numpy(x_grid)
    z_np      = detach_numpy(z_grid)

    return {
        "x": x_np, "z": z_np, "eps_r": eps_np,
        "E_real": Er, "E_imag": Ei,
        "Hr_x": Hrx, "Hi_x": Hix,
        "Hr_z": Hrz, "Hi_z": Hiz,
        "magnitude": magnitude, "phase": phase,
        "intensity": intensity,
        "intensity_normalized": intensity / (intensity.max() + 1e-30),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(fields: dict, output_dir: Path, history: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = fields["x"], fields["z"]
    ext  = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]

    def _save(fig, name):
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Permittivity
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fields["eps_r"], extent=ext, aspect="auto", cmap="viridis")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("εr(x,z)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="εr"); _save(fig, "mx_01_eps")

    # E real
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fields["E_real"], extent=ext, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("Re{E_y}")
    plt.colorbar(im, ax=ax, fraction=0.046); _save(fig, "mx_02_E_real")

    # E imag
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fields["E_imag"], extent=ext, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("Im{E_y}")
    plt.colorbar(im, ax=ax, fraction=0.046); _save(fig, "mx_03_E_imag")

    # Magnitude
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fields["magnitude"], extent=ext, aspect="auto", cmap="viridis")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("|E_y|")
    plt.colorbar(im, ax=ax, fraction=0.046, label="|E|"); _save(fig, "mx_04_magnitude")

    # Phase
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fields["phase"], extent=ext, aspect="auto", cmap="twilight")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("Phase (rad)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="rad"); _save(fig, "mx_05_phase")

    # H_x magnitude
    Hmag_x = np.sqrt(fields["Hr_x"]**2 + fields["Hi_x"]**2)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(Hmag_x, extent=ext, aspect="auto", cmap="viridis")
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title("|H̃_x|")
    plt.colorbar(im, ax=ax, fraction=0.046); _save(fig, "mx_06_Hx_magnitude")

    # Training history
    if history:
        df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ("total", "pde", "top", "bottom", "periodic"):
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=col)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Maxwell 2-D training"); ax.legend()
        _save(fig, "mx_07_history")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 2-D Maxwell PINN for grating")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--reference",  type=str, default=None,
                        help="Path to RCWA reference NPZ for validation")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--n-interior", type=int, default=None,
                        help="Override interior sampling points")
    parser.add_argument("--n-bc",       type=int, default=None,
                        help="Override boundary/periodic sampling points")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    # CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.n_interior is not None:
        config.sampling.n_interior = args.n_interior
    if args.n_bc is not None:
        config.sampling.n_top = args.n_bc
        config.sampling.n_bottom = args.n_bc
        config.sampling.n_periodic = args.n_bc

    out = Path(args.output_dir) if args.output_dir else Path(config.paths.figure_dir) / "maxwell"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device_str = args.device or config.training.device

    print("\n=== 2-D Maxwell PINN — binary diffraction grating ===")
    print(f"  Formulation: first-order TE (E_y, H̃_x, H̃_z)")
    print(f"  Convention:  exp(+iωt) suppressed, E_inc = exp(-ik0 z)")
    print(f"  Epochs: {config.training.epochs}")

    device = resolve_device(device_str)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    # Train
    model, history, ckpt_path = train_maxwell_grating(config, out, device_str)

    # Evaluate on grid
    print("\n  Evaluating on visualization grid...")
    fields = evaluate_maxwell_grating(model, config, device, dtype)

    # Save NPZ
    npz_path = out / "maxwell_results.npz"
    np.savez_compressed(npz_path, **fields)
    print(f"  Results: {npz_path}")

    # Save figures
    save_figures(fields, out, history)
    print(f"  Figures: {out}")

    # Metrics
    metrics: dict = {
        "top_bc_E_mse": float(np.mean((fields["E_real"][:, 0] - 1.0)**2 + fields["E_imag"][:, 0]**2)),
        "top_bc_H_mse": float(np.mean((fields["Hr_x"][:, 0] - config.physics.n_air)**2 + fields["Hi_x"][:, 0]**2)),
        "best_total_loss": min(row["total"] for row in history) if history else float("nan"),
        "epochs": config.training.epochs,
    }
    print(f"\n  Top BC E MSE: {metrics['top_bc_E_mse']:.4e}")
    print(f"  Top BC H MSE: {metrics['top_bc_H_mse']:.4e}")

    # Reference comparison
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        print(f"\n  Reference validation: {ref_path}")
        ref_metrics = run_comparison(
            reference_path=ref_path,
            pinn_x=fields["x"][0, :],
            pinn_z=fields["z"][:, 0],
            pinn_E_real=fields["E_real"],
            pinn_E_imag=fields["E_imag"],
            output_dir=out,
            save_figure=True,
        )
        if ref_metrics:
            # Also compute phase-sensitive metrics
            from src.reference_data import load_reference_npz, interpolate_reference_to_grid, normalize_reference_orientation
            ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
            ref_Er, ref_Ei, _ = interpolate_reference_to_grid(
                ref_data, fields["x"][0, :], fields["z"][:, 0]
            )
            valid = np.isfinite(ref_Er) & np.isfinite(ref_Ei)
            pinn_Er = fields["E_real"][valid]
            pinn_Ei = fields["E_imag"][valid]
            ref_Er_v = ref_Er[valid]
            ref_Ei_v = ref_Ei[valid]
            full_errs = evaluate_benchmark_errors(pinn_Er, pinn_Ei, ref_Er_v, ref_Ei_v)
            print(f"  Reference comparison:")
            for k, v in full_errs.items():
                print(f"    {k}: {v:.4e}")
            metrics["reference_comparison"] = full_errs
        else:
            print("  Reference file not found — skipped.")

    # Save metrics
    metrics_path = out / "maxwell_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
