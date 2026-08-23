#!/usr/bin/env python3
"""Train a 2-D first-order Maxwell PINN for the binary diffraction grating.

Formulation options
-------------------
--scattered   Use scattered-field formulation (recommended for gratings).
              Network predicts E_scat, H̃_scat only.
              Top BC allows reflected orders (outgoing Robin).

Default (--no-scattered) trains total-field for diagnostic comparison only.

Staged verification
-------------------
--case homogeneous   2D domain, constant εr=1, plane-wave target.
--case layered       2D domain, horizontal layers only, matches 1D benchmark.
--case grating       Full binary grating (default).

Usage::

    # Scattered-field grating (recommended)
    .venv/bin/python scripts/train_maxwell_grating.py \\
        --config configs/default.yaml \\
        --scattered \\
        --reference outputs/reference_grating.npz

    # Diagnostic: homogeneous plane wave
    .venv/bin/python scripts/train_maxwell_grating.py \\
        --case homogeneous --epochs 3000

    # Diagnostic: 2D layered (must match 1D Maxwell benchmark)
    .venv/bin/python scripts/train_maxwell_grating.py \\
        --case layered --epochs 5000
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

from src.benchmarks import evaluate_benchmark_errors
from src.config import PhysicsConfig, load_config
from src.geometry import epsilon_r_grid
from src.maxwell_2d import (
    Maxwell2DMLP,
    evaluate_losses_from_checkpoint,
    incident_E_H,
    maxwell_2d_total_loss,
    sample_collocation_points,
)
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
    run_comparison,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    config,
    output_dir: Path,
    device_str: str,
    hidden_layers: int,
    hidden_width: int,
    num_fourier_levels: int,
    scattered: bool,
    n_interface: int,
) -> tuple[Maxwell2DMLP, list[dict], Path]:

    set_seed(config.training.seed)
    device = resolve_device(device_str)
    dtype  = resolve_training_dtype(config.training.dtype, device)
    physics = config.physics

    model = Maxwell2DMLP(
        physics,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        num_fourier_levels=num_fourier_levels,
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    w_pde = config.loss_weights.pde
    w_top = config.loss_weights.top
    w_bot = config.loss_weights.bottom
    w_per = config.loss_weights.periodic

    print(f"  Parameters: {n_params:,}")
    print(f"  Formulation: {'scattered-field' if scattered else 'total-field'}")
    print(f"  Loss weights: pde={w_pde}  top={w_top}  bot={w_bot}  per={w_per}")

    # Initial sampling
    def _resample(seed_offset=0):
        return sample_collocation_points(
            physics,
            n_interior=config.sampling.n_interior,
            n_top=config.sampling.n_top,
            n_bottom=config.sampling.n_bottom,
            n_periodic=config.sampling.n_periodic,
            device=device, dtype=dtype,
            seed=config.training.seed + seed_offset,
            n_interface=n_interface,
        )

    pts = _resample(0)
    print(f"  Interior: {pts['x_int'].shape[0]}  ridge+near: included  "
          f"Top: {pts['x_top'].shape[0]}  Bot: {pts['x_bot'].shape[0]}  "
          f"Per: {pts['x_left'].shape[0]}"
          + (f"  Intf: {pts['x_intf'].shape[0]}" if 'x_intf' in pts else ""))

    opt   = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.training.epochs)

    history: list[dict] = []
    best_total = float("inf")
    best_state = None
    best_epoch = 0
    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, config.training.epochs + 1):
        if ep % 500 == 0:
            pts = _resample(ep)

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = maxwell_2d_total_loss(model, pts, physics, w_pde, w_top, w_bot, w_per, scattered)
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
            total_val = row["total"]
            if total_val < best_total:
                best_total = total_val
                best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if ep % (config.training.validation_interval * 5) == 0:
                print(f"  ep={ep:6d}  pde={row['pde']:.3e}  "
                      f"top={row['top']:.3e}  bot={row['bottom']:.3e}  "
                      f"per={row['periodic']:.3e}  total={total_val:.3e}"
                      + ("  *best*" if ep == best_epoch else ""))

    # Restore and save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ckpt_dir / "maxwell_best.pt"
    torch.save({
        "epoch":              config.training.epochs,
        "best_epoch":         best_epoch,
        "model_state_dict":   model.state_dict(),
        "config":             config,
        "best_loss":          best_total,
        "num_fourier_levels": num_fourier_levels,
        "hidden_layers":      hidden_layers,
        "hidden_width":       hidden_width,
        "scattered":          scattered,
    }, ckpt_path)
    print(f"  Best weighted total loss: {best_total:.4e}  at epoch {best_epoch}")
    print(f"  Checkpoint saved: {ckpt_path}")

    # ---- Checkpoint consistency check ----
    print("\n  === Checkpoint consistency audit ===")
    # Resample a fresh set with the training seed to get reproducible points
    pts_audit = _resample(0)
    reload_losses = evaluate_losses_from_checkpoint(
        ckpt_path, pts_audit, physics, w_pde, w_top, w_bot, w_per, scattered
    )
    print(f"  best_epoch stored:       {reload_losses['best_epoch']}")
    print(f"  best_loss stored:        {reload_losses['saved_best_loss']:.4e}")
    print(f"  reloaded eval_total:     {reload_losses['eval_total']:.4e}  "
          f"(weighted, same pts as training)")
    print(f"  reloaded eval_pde:       {reload_losses['eval_pde']:.4e}  (unweighted)")
    print(f"  reloaded eval_top:       {reload_losses['eval_top']:.4e}  (unweighted)")
    print(f"  reloaded eval_bottom:    {reload_losses['eval_bottom']:.4e}  (unweighted)")
    print(f"  reloaded eval_periodic:  {reload_losses['eval_periodic']:.4e}")
    # The weighted total computed here must approximately match best_total
    # (small difference expected due to resampling)
    ratio = reload_losses['eval_total'] / (best_total + 1e-30)
    print(f"  eval_total / best_total: {ratio:.3f}  (≈1.0 expected, may differ slightly due to resample)")

    hist_path = Path(config.paths.history_file).parent / "maxwell_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)

    return model, history, ckpt_path, reload_losses


# ---------------------------------------------------------------------------
# Evaluation on grid
# ---------------------------------------------------------------------------


def evaluate_on_grid(model, config, device, dtype, scattered=False):
    """Evaluate model on visualization grid. If scattered, add incident field."""
    physics = config.physics
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        out = model.forward(x_flat, z_flat)

    shape = x_grid.shape
    Er_net = detach_numpy(out[:, 0].reshape(shape))
    Ei_net = detach_numpy(out[:, 1].reshape(shape))

    if scattered:
        # Add incident field to get total field
        z_np    = detach_numpy(z_flat).reshape(shape)
        k0, n   = physics.k0, physics.n_air
        Er_inc  = np.cos(k0 * z_np)
        Ei_inc  = -np.sin(k0 * z_np)
        Er_total = Er_net + Er_inc
        Ei_total = Ei_net + Ei_inc
    else:
        Er_total = Er_net
        Ei_total = Ei_net

    magnitude = np.sqrt(Er_total**2 + Ei_total**2)
    x_np = detach_numpy(x_grid)
    z_np = detach_numpy(z_grid)

    return {
        "x": x_np, "z": z_np,
        "eps_r":   detach_numpy(eps_grid),
        "E_real":  Er_total, "E_imag":  Ei_total,
        "E_scat_real": Er_net,  "E_scat_imag": Ei_net,
        "magnitude": magnitude,
        "phase":     np.arctan2(Ei_total, Er_total),
        "intensity": magnitude**2,
        "intensity_normalized": magnitude**2 / (magnitude.max()**2 + 1e-30),
    }


# ---------------------------------------------------------------------------
# Staged verification cases
# ---------------------------------------------------------------------------


def make_homogeneous_physics(config):
    """Return a PhysicsConfig with no grating (uniform εr=1)."""
    from dataclasses import replace
    p = config.physics
    return PhysicsConfig(
        wavelength=p.wavelength, n_air=p.n_air,
        n_ridge=p.n_air, n_substrate=p.n_air,  # all air
        period=p.period, ridge_width=0.0, ridge_height=0.0,
        domain_height=p.domain_height, ridge_base_fraction=1.0,
        interface_margin=p.interface_margin,
        nx_visualization=p.nx_visualization, nz_visualization=p.nz_visualization,
    )


def make_layered_physics(config):
    """Return a PhysicsConfig matching the 1-D layered benchmark geometry."""
    p = config.physics
    return PhysicsConfig(
        wavelength=p.wavelength, n_air=p.n_air,
        n_ridge=p.n_ridge, n_substrate=p.n_substrate,
        period=p.period, ridge_width=p.period,   # full-width "ridge" = slab
        ridge_height=p.ridge_height,
        domain_height=p.domain_height,
        ridge_base_fraction=p.ridge_base_fraction,
        interface_margin=p.interface_margin,
        nx_visualization=p.nx_visualization, nz_visualization=p.nz_visualization,
    )


def check_homogeneous_case(config, device, dtype, n_pts=512):
    """Verify that the 2D Maxwell residual is zero for an exact plane wave in air.

    This checks the PDE implementation and sign convention before training.
    """
    print("\n  [Case A] Homogeneous check — PDE residual of exact plane wave")
    phys_hom = make_homogeneous_physics(config)
    k0 = phys_hom.k0

    class ExactPW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x, z):
            Er  =  torch.cos(k0 * z) + self.dummy*0 + x*0
            Ei  = -torch.sin(k0 * z) + self.dummy*0 + x*0
            n   = phys_hom.n_air
            Hrx =  n * Er;  Hix = n * Ei
            Hrz = torch.zeros_like(Er) + self.dummy*0 + x*0
            Hiz = torch.zeros_like(Er) + self.dummy*0 + x*0
            return torch.stack([Er, Ei, Hrx, Hix, Hrz, Hiz], dim=-1)
        def field_components(self, x, z):
            out = self.forward(x, z)
            return tuple(out[:, i] for i in range(6))

    model = ExactPW()
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(0.1, phys_hom.period - 0.1, n_pts), dtype=dtype)
    z = torch.as_tensor(rng.uniform(0.1, phys_hom.domain_height - 0.1, n_pts), dtype=dtype)
    from src.maxwell_2d import maxwell_2d_pde_residual
    residuals = maxwell_2d_pde_residual(model, x, z, phys_hom)
    mses = [float(torch.mean(r**2).detach()) for r in residuals]
    max_mse = max(mses)
    status = "PASS" if max_mse < 1e-8 else "FAIL"
    print(f"  Max equation MSE: {max_mse:.2e}  [{status}]")
    if max_mse >= 1e-8:
        for i, m in enumerate(mses):
            print(f"    Eq {i}: {m:.2e}")
    return max_mse < 1e-8


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(fields, output_dir, history, tag="mx"):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = fields["x"], fields["z"]
    ext = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]

    def _s(fig, name):
        fig.savefig(output_dir / f"{tag}_{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{tag}_{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    for key, title, cmap in [
        ("eps_r",     "εr(x,z)",   "viridis"),
        ("E_real",    "Re{E_y}",   "RdBu_r"),
        ("E_imag",    "Im{E_y}",   "RdBu_r"),
        ("magnitude", "|E_y|",     "viridis"),
        ("phase",     "Phase(rad)","twilight"),
    ]:
        if key not in fields:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(fields[key], extent=ext, aspect="auto", cmap=cmap)
        ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        _s(fig, key)

    if history:
        df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ("total", "pde", "top", "bottom"):
            if col in df.columns:
                ax.semilogy(df["epoch"], df[col], label=col)
        ax.set_xlabel("Epoch"); ax.legend(); ax.set_title("Training loss")
        _s(fig, "history")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 2-D Maxwell PINN")
    parser.add_argument("--config",        type=str, default="configs/default.yaml")
    parser.add_argument("--reference",     type=str, default=None)
    parser.add_argument("--device",        type=str, default="cpu")
    parser.add_argument("--output-dir",    type=str, default=None)
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--n-interior",    type=int, default=2048)
    parser.add_argument("--n-bc",          type=int, default=256)
    parser.add_argument("--n-interface",   type=int, default=256,
                        help="Extra collocation points near grating boundary")
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width",  type=int, default=64)
    parser.add_argument("--fourier-levels",type=int, default=4)
    parser.add_argument("--scattered",     action="store_true",
                        help="Use scattered-field formulation (recommended for grating)")
    parser.add_argument("--case",          type=str, default="grating",
                        choices=["grating", "homogeneous", "layered"],
                        help="Which geometry to train on")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    config.sampling.n_interior = args.n_interior
    config.sampling.n_top      = args.n_bc
    config.sampling.n_bottom   = args.n_bc
    config.sampling.n_periodic = args.n_bc

    # Choose geometry
    if args.case == "homogeneous":
        config.physics = make_homogeneous_physics(config)
        tag = "hom"
    elif args.case == "layered":
        config.physics = make_layered_physics(config)
        tag = "lay"
    else:
        tag = "grating"

    out = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "maxwell"
    out.mkdir(parents=True, exist_ok=True)
    device_str = args.device

    print(f"\n=== 2-D Maxwell PINN  [{args.case}]  "
          f"{'scattered-field' if args.scattered else 'total-field'} ===")
    print(f"  Formulation: first-order TE (E_y, H̃_x, H̃_z)")
    print(f"  Convention:  E_inc = exp(-ik0 z),  H̃_x = +n*E for forward wave")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Architecture: {args.hidden_layers}×{args.hidden_width}  "
          f"fourier_levels={args.fourier_levels}")

    device = resolve_device(device_str)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    # Step 0: verify PDE implementation on exact plane wave
    pde_check_passed = check_homogeneous_case(config, device, dtype)
    if not pde_check_passed:
        print("  ERROR: 2D PDE implementation failed homogeneous check. Aborting.")
        return 1

    # Step 1: train
    model, history, ckpt_path, audit = train(
        config, out, device_str,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        num_fourier_levels=args.fourier_levels,
        scattered=args.scattered,
        n_interface=args.n_interface,
    )

    # Step 2: evaluate on grid
    print("\n  Evaluating on visualization grid...")
    fields = evaluate_on_grid(model, config, device, dtype, scattered=args.scattered)

    npz_path = out / f"{tag}_results.npz"
    np.savez_compressed(npz_path, **fields)
    save_figures(fields, out, history, tag=tag)

    # Step 3: report checkpoint audit
    print(f"\n=== Metric summary ===")
    print(f"  best_epoch:           {audit['best_epoch']}")
    print(f"  best weighted total:  {audit['saved_best_loss']:.4e}  (from training)")
    print(f"  reloaded eval total:  {audit['eval_total']:.4e}  (from fresh load, training pts)")
    print(f"  reloaded eval pde:    {audit['eval_pde']:.4e}  (unweighted)")
    print(f"  reloaded eval top:    {audit['eval_top']:.4e}  (unweighted)")
    print(f"  reloaded eval bot:    {audit['eval_bottom']:.4e}  (unweighted)")

    # Grid-evaluated top-BC MSE (different from training — different pts)
    z0_row = 0  # first row in grid is z=0
    grid_top_E_mse = float(np.mean((fields["E_real"][z0_row, :] - 1.0)**2 +
                                    fields["E_imag"][z0_row, :]**2))
    # Hr_x is not stored in grid fields dict; skip H grid MSE
    grid_top_H_mse = float("nan")  # computed from training pts in audit above
    print(f"  grid top E MSE:       {grid_top_E_mse:.4e}  "
          f"(viz grid pts — different from training pts)")
    print(f"  grid top H MSE:       {grid_top_H_mse:.4e}  (viz grid pts)")

    # Layered case: compare against 1D benchmark
    if args.case == "layered":
        print(f"\n  Comparing 2D result against 1D analytical solution...")
        from src.benchmarks import LayeredMediumBenchmark, evaluate_benchmark_errors
        p = config.physics
        bm = LayeredMediumBenchmark(
            n_air=p.n_air, n_slab=p.n_ridge, n_sub=p.n_substrate,
            k0=p.k0, z_slab_top=p.ridge_z_min, z_slab_bot=p.ridge_z_max,
            domain_height=p.domain_height,
        )
        # Average field over x (should be x-independent)
        pinn_Er = fields["E_real"].mean(axis=1)
        pinn_Ei = fields["E_imag"].mean(axis=1)
        z_np = fields["z"][:, 0]
        ref_Er, ref_Ei = bm.analytical_field_np(np.zeros_like(z_np), z_np)
        errs = evaluate_benchmark_errors(pinn_Er, pinn_Ei, ref_Er, ref_Ei)
        print(f"  2D-vs-1D comparison (x-averaged vs analytical):")
        for k, v in errs.items():
            print(f"    {k}: {v:.4e}")

    # Step 4: reference comparison (grating case)
    ref_metrics: dict = {}
    if args.reference and args.case == "grating":
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        if ref_path.exists():
            print(f"\n  Reference validation (RCWA): {ref_path}")
            ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
            pinn_x1d = fields["x"][0, :]
            pinn_z1d = fields["z"][:, 0]
            ref_Er, ref_Ei, _ = interpolate_reference_to_grid(ref_data, pinn_x1d, pinn_z1d)
            valid = np.isfinite(ref_Er) & np.isfinite(ref_Ei)
            full_errs = evaluate_benchmark_errors(
                fields["E_real"][valid], fields["E_imag"][valid],
                ref_Er[valid], ref_Ei[valid],
            )
            print(f"  Comparison (PINN total field vs RCWA):")
            for k, v in full_errs.items():
                print(f"    {k}: {v:.4e}")
            ref_metrics = full_errs
        else:
            print(f"  Reference file not found: {ref_path}")

    # Save JSON metrics
    out_metrics = {
        "case":           args.case,
        "scattered":      args.scattered,
        "best_epoch":     audit["best_epoch"],
        "best_loss":      audit["saved_best_loss"],
        "checkpoint_audit": audit,
        "grid_top_E_mse": grid_top_E_mse,
        "grid_top_H_mse": grid_top_H_mse,
        "reference":      ref_metrics,
    }
    metrics_path = out / f"{tag}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(out_metrics, f, indent=2, default=str)
    print(f"\n  Metrics: {metrics_path}")
    print(f"  Figures: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
