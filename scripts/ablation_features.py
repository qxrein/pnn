#!/usr/bin/env python3
"""Feature-frequency ablation study for the 2-D Maxwell DD-PINN.

Runs all four feature variants under IDENTICAL conditions (same seed, same
collocation points, same optimizer, same epochs, same reference) and reports
a comparison table.

Variants:
    A. normalized          -- xn, zn ∈ [-1,1],  freq = 2*pi/h (overcomplete)
    B. global_k0           -- xbar=k0*x, zbar=k0*z  (same freq for all subdomains)
    C. local_material_k    -- xi_z=k_j*(z-zlo)  (exact match to medium oscillation)
    D. local_material_plus_grating_x  -- xi_z=k_j*(z-zlo), xi_x=2*pi*x/Lambda

PDE residuals always computed w.r.t. global (xbar, zbar) — chain rule holds.

Usage::

    .venv/bin/python scripts/ablation_features.py \\
        --case grating --epochs 5000 --reference outputs/reference_grating.npz

    .venv/bin/python scripts/ablation_features.py \\
        --case lambda_0p8 --epochs 5000 --gen-reference
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks import evaluate_benchmark_errors
from src.config import PhysicsConfig, load_config
from src.geometry import epsilon_r_grid
from src.maxwell_2d_nondim import (
    maxwell_2d_nd_bottom_bc,
    maxwell_2d_nd_interface_loss,
    maxwell_2d_nd_top_bc,
    sample_nd_points,
)
from src.maxwell_feature_variants import FEATURE_VARIANTS, Maxwell2DDD_Variant
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# PDE residual using variant model
# ---------------------------------------------------------------------------


def _g(field, wrt, ones):
    g = torch.autograd.grad(field, wrt, grad_outputs=ones,
                             create_graph=True, retain_graph=True, allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(wrt)


def variant_pde_residual(
    subnet,
    x: torch.Tensor,
    z: torch.Tensor,
    physics: PhysicsConfig,
    eps_val: float,
    scattered: bool = True,
) -> tuple:
    """Nondim Maxwell residuals via autograd on global xbar, zbar."""
    k0 = physics.k0
    xbar = (x * k0).detach().clone().requires_grad_(True)
    zbar = (z * k0).detach().clone().requires_grad_(True)
    # Physical coords from nondim (autograd through conversion)
    x_phys = xbar / k0
    z_phys = zbar / k0

    Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z = subnet.field_components(x_phys, z_phys)
    ones = torch.ones_like(Er)

    dEr_dz  = _g(Er,   zbar, ones); dEi_dz  = _g(Ei,   zbar, ones)
    dHrx_dz = _g(Hr_x, zbar, ones); dHix_dz = _g(Hi_x, zbar, ones)
    dEr_dx  = _g(Er,   xbar, ones); dEi_dx  = _g(Ei,   xbar, ones)
    dHrz_dx = _g(Hr_z, xbar, ones); dHiz_dx = _g(Hi_z, xbar, ones)

    res = [dEr_dz - Hi_x, dEi_dz + Hr_x,
           dEr_dx + Hi_z, dEi_dx - Hr_z,
           dHrx_dz - dHrz_dx - eps_val * Ei,
           dHix_dz - dHiz_dx + eps_val * Er]

    if scattered:
        Er_inc = torch.cos(zbar.detach())
        Ei_inc = -torch.sin(zbar.detach())
        d = eps_val - 1.0
        res[4] = res[4] - d * Ei_inc
        res[5] = res[5] + d * Er_inc

    return tuple(res)


def variant_total_loss(
    model: Maxwell2DDD_Variant,
    pts: dict,
    physics: PhysicsConfig,
    w_pde=1.0, w_E=500.0, w_H=500.0, w_top=200.0, w_bot=100.0,
    scattered=True,
) -> dict:
    zero = torch.zeros(1, dtype=torch.float64)
    grat_thick = physics.ridge_z_max - physics.ridge_z_min

    def _pde(key, subnet, eps):
        xk = pts.get(f"x_{key}"); zk = pts.get(f"z_{key}")
        if xk is None or len(xk) == 0: return zero
        res = variant_pde_residual(subnet, xk, zk, physics, eps, scattered)
        return sum(torch.mean(r**2) for r in res) / len(res)

    L_air  = _pde("air",  model.net_air,  physics.n_air**2)
    L_grat = _pde("grat", model.net_grat, physics.n_ridge**2) if grat_thick > 1e-6 else zero
    L_sub  = _pde("sub",  model.net_sub,  physics.n_substrate**2)
    L_pde  = L_air + L_grat + L_sub

    if grat_thick > 1e-6:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat, physics.ridge_z_min, pts["x_int1"])
        LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,  physics.ridge_z_max, pts["x_int2"])
    else:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air, model.net_sub, physics.ridge_z_min, pts.get("x_int1", pts["x_top"]))
        LE2 = zero; LH2 = zero

    L_top = maxwell_2d_nd_top_bc(model.net_air, pts["x_top"], physics)
    L_bot = maxwell_2d_nd_bottom_bc(model.net_sub, pts["x_bot"], physics)

    total = w_pde*L_pde + w_E*(LE1+LE2) + w_H*(LH1+LH2) + w_top*L_top + w_bot*L_bot
    return {
        "pde": L_pde, "pde_air": L_air, "pde_grat": L_grat, "pde_sub": L_sub,
        "E_int1": LE1, "H_int1": LH1, "E_int2": LE2, "H_int2": LH2,
        "top": L_top, "bottom": L_bot, "total": total,
    }


# ---------------------------------------------------------------------------
# Training (shared for all variants)
# ---------------------------------------------------------------------------


def train_variant(
    variant: str,
    physics: PhysicsConfig,
    pts_train: dict,    # fixed shared collocation points
    device, dtype,
    hidden_layers: int, hidden_width: int, num_fourier_levels: int,
    epochs: int, lr: float, seed: int,
    w_pde=1.0, w_E=500.0, w_H=500.0, w_top=200.0, w_bot=100.0,
) -> tuple[Maxwell2DDD_Variant, list[dict]]:
    set_seed(seed)
    model = Maxwell2DDD_Variant(physics, variant, hidden_layers, hidden_width, num_fourier_levels)
    model = model.to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{variant}]  params={n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_total = float("inf")
    best_state = None
    best_epoch = 0
    pts = pts_train  # use fixed shared points throughout

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = variant_total_loss(model, pts, physics, w_pde, w_E, w_H, w_top, w_bot)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % 100 == 0 or ep == 1:
            tv = float(losses["total"].detach())
            row = {
                "epoch": ep,
                "pde_air":  float(losses["pde_air"].detach()),
                "pde_grat": float(losses["pde_grat"].detach()),
                "pde_sub":  float(losses["pde_sub"].detach()),
                "E_int":    float((losses["E_int1"]+losses["E_int2"]).detach()),
                "H_int":    float((losses["H_int1"]+losses["H_int2"]).detach()),
                "top":      float(losses["top"].detach()),
                "bottom":   float(losses["bottom"].detach()),
                "total":    tv,
            }
            history.append(row)
            if tv < best_total:
                best_total = tv; best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            r = history[-1]
            print(f"    ep={ep:5d}  pde_air={r['pde_air']:.2e}  pde_grat={r['pde_grat']:.2e}  "
                  f"pde_sub={r['pde_sub']:.2e}  E_int={r['E_int']:.2e}  "
                  f"top={r['top']:.2e}  bot={r['bottom']:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{variant}]  best={best_total:.4e}  epoch={best_epoch}")
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_variant(model: Maxwell2DDD_Variant, physics: PhysicsConfig, device, dtype) -> dict:
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1); z_flat = z_grid.reshape(-1)
    model.eval()
    with torch.no_grad():
        Er_s = torch.zeros(x_flat.shape[0], dtype=dtype, device=device)
        Ei_s = torch.zeros_like(Er_s)
        p = physics
        for mask, net in [
            (z_flat <= p.ridge_z_min, model.net_air),
            ((z_flat > p.ridge_z_min) & (z_flat <= p.ridge_z_max), model.net_grat),
            (z_flat > p.ridge_z_max, model.net_sub),
        ]:
            if mask.any():
                out = net.forward(x_flat[mask], z_flat[mask])
                Er_s[mask] = out[:, 0]; Ei_s[mask] = out[:, 1]

    shape = x_grid.shape
    Er_net = detach_numpy(Er_s.reshape(shape)); Ei_net = detach_numpy(Ei_s.reshape(shape))
    z_np = detach_numpy(z_grid); k0 = physics.k0
    Er = Er_net + np.cos(k0*z_np); Ei = Ei_net - np.sin(k0*z_np)
    mag = np.sqrt(Er**2 + Ei**2)
    return {
        "x": detach_numpy(x_grid), "z": z_np, "eps_r": detach_numpy(eps_grid),
        "E_real": Er, "E_imag": Ei, "magnitude": mag,
        "phase": np.arctan2(Ei, Er),
    }


# ---------------------------------------------------------------------------
# Normalised diffraction efficiencies
# ---------------------------------------------------------------------------


def normalised_diffraction(E_total_2d, x1d, z1d, physics, n_orders=3):
    from scripts.audit_maxwell_2d import compute_normalised_diffraction
    return compute_normalised_diffraction(E_total_2d, x1d, z1d, physics, n_orders)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature-frequency ablation")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--case",           default="grating",
                        choices=["grating", "lambda_0p8"])
    parser.add_argument("--reference",      default=None)
    parser.add_argument("--gen-reference",  action="store_true")
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--epochs",         type=int, default=5000)
    parser.add_argument("--hidden-layers",  type=int, default=4)
    parser.add_argument("--hidden-width",   type=int, default=64)
    parser.add_argument("--fourier-levels", type=int, default=4)
    parser.add_argument("--n-per-region",   type=int, default=1024)
    parser.add_argument("--n-interface",    type=int, default=256)
    parser.add_argument("--n-bc",           type=int, default=256)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--w-pde",   type=float, default=1.0)
    parser.add_argument("--w-E",     type=float, default=500.0)
    parser.add_argument("--w-H",     type=float, default=500.0)
    parser.add_argument("--w-top",   type=float, default=200.0)
    parser.add_argument("--w-bot",   type=float, default=100.0)
    parser.add_argument("--variants", nargs="+", default=list(FEATURE_VARIANTS),
                        choices=list(FEATURE_VARIANTS))
    parser.add_argument("--output-dir", default="outputs/ablation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    # Physics
    base = config.physics
    if args.case == "lambda_0p8":
        p = PhysicsConfig(
            wavelength=base.wavelength, n_air=base.n_air,
            n_ridge=base.n_ridge, n_substrate=base.n_substrate,
            period=0.8*base.wavelength, ridge_width=0.4*0.8*base.wavelength,
            ridge_height=base.ridge_height, domain_height=base.domain_height,
            ridge_base_fraction=base.ridge_base_fraction,
            nx_visualization=base.nx_visualization, nz_visualization=base.nz_visualization,
        )
        print(f"  Lambda=0.8λ: period={p.period:.4f}  kx_1={2*np.pi/p.period:.4f}")
        kx1 = 2*np.pi/p.period; kz1sq = p.k0**2 - kx1**2
        print(f"  ±1 orders: {'PROPAGATING kz=' + f'{np.sqrt(kz1sq):.4f}' if kz1sq>0 else 'EVANESCENT'}")
    else:
        p = base

    # Reference
    ref_path = None
    if args.gen_reference or args.case == "lambda_0p8":
        from scripts.generate_reference import solve_rcwa
        rfile = ROOT / f"outputs/reference_{args.case}.npz"
        print(f"\n  Generating RCWA reference → {rfile}")
        x_r, z_r, Er_r, Ei_r, _amps_r = solve_rcwa(p, N_harmonics=75)
        np.savez(rfile, x=x_r, z=z_r, E_real=Er_r, E_imag=Ei_r)
        ref_path = rfile
        print(f"  Saved: {ref_path}")
    elif args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path

    print(f"\n=== Feature-frequency ablation  [{args.case}] ===")
    print(f"  Variants: {args.variants}")
    print(f"  Epochs={args.epochs}  arch={args.hidden_layers}×{args.hidden_width}  "
          f"fourier_levels={args.fourier_levels}  seed={args.seed}")
    print(f"  ALL VARIANTS use identical seed, collocation pts, optimizer, epochs")

    # Sample FIXED shared collocation points (same for all variants)
    set_seed(args.seed)
    pts = sample_nd_points(
        p, args.n_per_region, args.n_interface, args.n_bc, args.n_bc,
        device, dtype, seed=args.seed,
    )
    print(f"  Interior: {pts['x_air'].shape[0]}/region  Interface: {pts['x_int1'].shape[0]}  BC: {pts['x_top'].shape[0]}")

    # Feature frequency documentation
    print(f"\n  Feature frequency analysis:")
    k0 = p.k0
    for name, z_lo, z_hi, k_j in [
        ("air",     0.0,         p.ridge_z_min,    k0*p.n_air),
        ("grating", p.ridge_z_min, p.ridge_z_max,  k0*p.n_ridge),
        ("sub",     p.ridge_z_max, p.domain_height, k0*p.n_substrate),
    ]:
        h = z_hi - z_lo
        if h < 1e-6: continue
        freq_norm   = 2*np.pi/h
        freq_k0     = k0
        freq_kj     = k_j
        freq_grat_x = 2*np.pi/p.period
        print(f"  [{name:8s}] h={h:.3f}λ  k_j={k_j:.3f}  "
              f"normalized={freq_norm:.2f}  global_k0={freq_k0:.2f}  "
              f"local_kj={freq_kj:.2f}  grating_x={freq_grat_x:.2f}")

    # Run each variant
    results = {}
    for variant in args.variants:
        print(f"\n  --- Variant: {variant} ---")
        model, history = train_variant(
            variant, p, pts, device, dtype,
            args.hidden_layers, args.hidden_width, args.fourier_levels,
            args.epochs, args.lr, args.seed,
            args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot,
        )

        fields = evaluate_variant(model, p, device, dtype)

        # Per-region final PDE residuals
        final = history[-1] if history else {}

        # Reference comparison
        ref_errs = {}
        diff_metrics = {}
        if ref_path and ref_path.exists():
            ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
            pinn_x1d = fields["x"][0, :]; pinn_z1d = fields["z"][:, 0]
            ref_Er, ref_Ei, _ = interpolate_reference_to_grid(ref_data, pinn_x1d, pinn_z1d)
            valid = np.isfinite(ref_Er) & np.isfinite(ref_Ei)
            ref_errs = evaluate_benchmark_errors(
                fields["E_real"][valid], fields["E_imag"][valid],
                ref_Er[valid], ref_Ei[valid],
            )
            # Normalised diffraction
            E_c = fields["E_real"] + 1j*fields["E_imag"]
            diff_metrics = normalised_diffraction(E_c, pinn_x1d, pinn_z1d, p, n_orders=3)

        results[variant] = {
            "final_losses": final,
            "ref_errors": ref_errs,
            "diffraction": {k: float(v) for k, v in diff_metrics.items()
                            if isinstance(v, (int, float, np.floating))},
        }

        # Save field figure
        np.savez_compressed(out / f"{variant}_fields.npz",
                            x=fields["x"], z=fields["z"],
                            E_real=fields["E_real"], E_imag=fields["E_imag"],
                            magnitude=fields["magnitude"])

    # Comparison table
    print("\n" + "="*90)
    print("ABLATION COMPARISON TABLE")
    print("="*90)
    header = f"{'Variant':35}  {'complex_L2':>12}  {'phase_rmse':>12}  {'mag_L2':>10}  {'pde_grat':>10}  {'R+T':>8}"
    print(header)
    print("-" * 90)
    for v in args.variants:
        r = results[v]
        e = r["ref_errors"]
        l = r["final_losses"]
        d = r["diffraction"]
        cl2  = e.get("relative_complex_l2", float("nan"))
        prms = e.get("phase_rmse_deg", float("nan"))
        ml2  = e.get("relative_l2_magnitude", float("nan"))
        pg   = l.get("pde_grat", float("nan"))
        rt   = d.get("energy_check", float("nan"))
        print(f"  {v:33}  {cl2:12.4e}  {prms:12.2f}°  {ml2:10.4e}  {pg:10.4e}  {rt:8.4f}")
    print("="*90)

    print("\nDetailed per-variant losses (final epoch):")
    for v in args.variants:
        l = results[v]["final_losses"]
        print(f"\n  [{v}]")
        for k in ("pde_air","pde_grat","pde_sub","E_int","H_int","top","bottom","total"):
            if k in l: print(f"    {k}: {l[k]:.4e}")

    if ref_path:
        print("\nDiffraction efficiencies (normalised, scattered only):")
        print(f"  {'Variant':35}  {'R0':>8}  {'T0':>8}  {'R+T':>8}")
        for v in args.variants:
            d = results[v]["diffraction"]
            print(f"  {v:35}  {d.get('R0',float('nan')):8.4f}  {d.get('T0',float('nan')):8.4f}  {d.get('energy_check',float('nan')):8.4f}")

    # Save JSON
    mp = out / f"ablation_{args.case}.json"
    with mp.open("w") as f:
        json.dump({"case": args.case, "args": vars(args), "results": results}, f, indent=2, default=str)
    print(f"\n  Report: {mp}")

    # Training loss figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, key, title in [(axes[0],"pde_grat","PDE grating"),
                            (axes[1],"top","Top BC"),
                            (axes[2],"total","Total loss")]:
        for v in args.variants:
            h = results[v]["final_losses"]
            # Re-run short training to get history — skip for now, use final values
            pass
        ax.set_title(title); ax.set_xlabel("Epoch")
    plt.savefig(out / f"ablation_{args.case}_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
