#!/usr/bin/env python3
"""Layered-background scattered-field 2-D Maxwell PINN.

Compares:
  free_space  -- original: E_total = E_inc + E_scat, source = (eps-1)*E_inc
  layered_bg  -- new:      E_total = E_bg  + E_scat, source = (eps-eps_bg)*E_bg

Key difference: source is ZERO in substrate for layered_bg.
Controlled comparison uses identical seed, points, optimizer, epochs.

Usage::

    .venv/bin/python scripts/train_lbg.py \\
        --case lambda_0p8 --epochs 5000 --device cpu

    .venv/bin/python scripts/train_lbg.py \\
        --case grating --epochs 8000 --device cpu \\
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks import evaluate_benchmark_errors
from src.config import PhysicsConfig, load_config
from src.field_comparison import (
    compare_fields,
    compare_modal_with_rcwa,
    extract_modal_amplitudes,
    flat_contrast_test,
    print_comparison_summary,
    load_rcwa_amplitudes,
)
from src.geometry import epsilon_r_grid
from src.maxwell_2d_nondim import (
    maxwell_2d_nd_bottom_bc,
    maxwell_2d_nd_interface_loss,
    maxwell_2d_nd_pde_loss_subdomain,
    maxwell_2d_nd_top_bc,
    sample_nd_points,
)
from src.maxwell_feature_variants import Maxwell2DDD_Variant
from src.maxwell_layered_bg import (
    background_field_np,
    background_field_torch,
    compute_background_coefficients,
    compute_source_map,
    delta_eps_tensor,
    lbg_bottom_bc,
    lbg_top_bc,
    maxwell_2d_lbg_pde_residual,
    reconstruct_total_field,
)
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def make_lambda_0p8(base: PhysicsConfig) -> PhysicsConfig:
    new_period = 0.8 * base.wavelength
    return PhysicsConfig(
        wavelength=base.wavelength, n_air=base.n_air,
        n_ridge=base.n_ridge, n_substrate=base.n_substrate,
        period=new_period, ridge_width=0.4 * new_period,
        ridge_height=base.ridge_height, domain_height=base.domain_height,
        ridge_base_fraction=base.ridge_base_fraction,
        nx_visualization=base.nx_visualization, nz_visualization=base.nz_visualization,
    )


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _g(f, w, o):
    g = torch.autograd.grad(f, w, grad_outputs=o, create_graph=True,
                             retain_graph=True, allow_unused=True)[0]
    return g if g is not None else torch.zeros_like(w)


def free_space_loss(model, pts, physics, w_pde, w_E, w_H, w_top, w_bot):
    """Original scattered-field loss (source = (eps-1)*E_inc)."""
    zero = torch.zeros(1, dtype=torch.float64)
    p = physics
    gt = p.ridge_z_max - p.ridge_z_min

    def _pde(k, net, eps):
        xk = pts.get(f"x_{k}"); zk = pts.get(f"z_{k}")
        if xk is None or not len(xk): return zero
        # Use variant_pde_residual which calls field_components(x_phys, z_phys)
        from scripts.ablation_features import variant_pde_residual
        res = variant_pde_residual(net, xk, zk, physics, eps, scattered=True)
        return sum(torch.mean(r**2) for r in res) / len(res)

    La = _pde("air",  model.net_air,  p.n_air**2)
    Lg = _pde("grat", model.net_grat, p.n_ridge**2) if gt > 1e-6 else zero
    Ls = _pde("sub",  model.net_sub,  p.n_substrate**2)
    Lp = La + Lg + Ls

    if gt > 1e-6:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat, p.ridge_z_min, pts["x_int1"])
        LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,  p.ridge_z_max, pts["x_int2"])
    else:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air, model.net_sub, p.ridge_z_min, pts.get("x_int1", pts["x_top"]))
        LE2 = zero; LH2 = zero

    Lt = maxwell_2d_nd_top_bc(model.net_air, pts["x_top"], physics)
    Lb = maxwell_2d_nd_bottom_bc(model.net_sub, pts["x_bot"], physics)

    total = w_pde*Lp + w_E*(LE1+LE2) + w_H*(LH1+LH2) + w_top*Lt + w_bot*Lb
    return {"pde":Lp,"pde_air":La,"pde_grat":Lg,"pde_sub":Ls,
            "E_int1":LE1,"H_int1":LH1,"E_int2":LE2,"H_int2":LH2,
            "top":Lt,"bottom":Lb,"total":total}


def layered_bg_loss(model, pts, physics, coeff, w_pde, w_E, w_H, w_top, w_bot,
                    use_dtn: bool = False, n_dtn_orders: int = 8,
                    rcwa_amps: dict | None = None, w_modal: float = 0.0,
                    n_modal_orders: int = 3):
    """Layered-background scattered-field loss.

    use_dtn=True    → replace Robin BCs with modal DtN.
    rcwa_amps       → if provided, add modal data loss at boundaries.
    w_modal         → weight for modal data loss (0 = disabled).
    """
    zero = torch.zeros(1, dtype=torch.float64)
    p = physics
    gt = p.ridge_z_max - p.ridge_z_min

    def _pde(k, net, eps):
        xk = pts.get(f"x_{k}"); zk = pts.get(f"z_{k}")
        if xk is None or not len(xk): return zero
        res = maxwell_2d_lbg_pde_residual(net, xk, zk, physics, eps, coeff)
        return sum(torch.mean(r**2) for r in res) / len(res)

    La = _pde("air",  model.net_air,  p.n_air**2)
    Lg = _pde("grat", model.net_grat, p.n_ridge**2) if gt > 1e-6 else zero
    Ls = _pde("sub",  model.net_sub,  p.n_substrate**2)
    Lp = La + Lg + Ls

    if gt > 1e-6:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air,  model.net_grat, p.ridge_z_min, pts["x_int1"])
        LE2, LH2 = maxwell_2d_nd_interface_loss(model.net_grat, model.net_sub,  p.ridge_z_max, pts["x_int2"])
    else:
        LE1, LH1 = maxwell_2d_nd_interface_loss(model.net_air, model.net_sub, p.ridge_z_min, pts.get("x_int1", pts["x_top"]))
        LE2 = zero; LH2 = zero

    Lt = lbg_top_bc(model.net_air, pts["x_top"], physics,
                    use_dtn=use_dtn, n_dtn_orders=n_dtn_orders)
    Lb = lbg_bottom_bc(model.net_sub, pts["x_bot"], physics, coeff,
                       use_dtn=use_dtn, n_dtn_orders=n_dtn_orders)

    # Modal data loss (optional)
    L_modal = zero
    if rcwa_amps is not None and w_modal > 0:
        from src.field_comparison import modal_data_loss
        # Use same monitor planes as extract_modal_amplitudes
        z_top_monitor = 0.08 * p.domain_height
        z_bot_monitor = 0.92 * p.domain_height
        # Transmission: bottom monitor (deep in substrate, no evanescent contamination)
        L_modal_bot = modal_data_loss(
            model.net_sub, z_bot_monitor, physics,
            rcwa_amps["c_trans"], rcwa_amps["N_harmonics"],
            "bottom", p.n_substrate,
            n_data_orders=n_modal_orders,
            weight_propagating=1.0, weight_evanescent=0.0,
        )
        # Reflection: top monitor (in air, away from grating)
        L_modal_top = modal_data_loss(
            model.net_air, z_top_monitor, physics,
            rcwa_amps["c_refl"], rcwa_amps["N_harmonics"],
            "top", p.n_air,
            n_data_orders=n_modal_orders,
            weight_propagating=1.0, weight_evanescent=0.0,
        )
        L_modal = L_modal_bot + L_modal_top

    total = w_pde*Lp + w_E*(LE1+LE2) + w_H*(LH1+LH2) + w_top*Lt + w_bot*Lb + w_modal*L_modal
    return {"pde":Lp,"pde_air":La,"pde_grat":Lg,"pde_sub":Ls,
            "E_int1":LE1,"H_int1":LH1,"E_int2":LE2,"H_int2":LH2,
            "top":Lt,"bottom":Lb,"modal":L_modal,"total":total}


# ---------------------------------------------------------------------------
# Training loop (shared)
# ---------------------------------------------------------------------------


def train(model, pts, physics, loss_fn, epochs, lr, seed, clip=1.0,
          modal_warmup_epochs: int = 0, loss_fn_modal=None,
          best_from_epoch: int = 1):
    """Training loop with optional modal warm-up and epoch-gated best tracking."""
    set_seed(seed)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_total = float("inf"); best_state = None; best_epoch = 0
    history = []

    for ep in range(1, epochs+1):
        current_loss_fn = (
            loss_fn_modal if (loss_fn_modal is not None and ep > modal_warmup_epochs)
            else loss_fn
        )

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = current_loss_fn(model, pts, physics)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); sched.step()

        if ep % 100 == 0 or ep == 1:
            tv = float(losses["total"].detach())
            row = {k: float(v.detach()) for k, v in losses.items()}
            row["epoch"] = ep
            history.append(row)
            # Track best only from best_from_epoch onward
            if ep >= best_from_epoch and tv < best_total:
                best_total = tv; best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            r = history[-1]
            modal_str = f"  modal={r.get('modal', 0):.2e}" if r.get('modal', 0) > 0 else ""
            warmup_str = " [warmup]" if (loss_fn_modal is not None and ep <= modal_warmup_epochs) else ""
            print(f"    ep={ep:5d}  pde_air={r['pde_air']:.2e}  pde_grat={r['pde_grat']:.2e}  "
                  f"pde_sub={r['pde_sub']:.2e}  E_int={r['E_int1']+r['E_int2']:.2e}  "
                  f"top={r['top']:.2e}  bot={r['bottom']:.2e}{modal_str}{warmup_str}")

    if best_state:
        model.load_state_dict(best_state)
    else:
        # No best state found in allowed range — keep last state
        pass
    print(f"  best={best_total:.4e}  epoch={best_epoch}")
    return history, best_total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, physics, device, dtype, formulation, coeff=None):
    xg, zg, epsg = epsilon_r_grid(physics, device, dtype)
    xf = xg.reshape(-1); zf = zg.reshape(-1)

    model.eval()
    with torch.no_grad():
        Er_s = torch.zeros(xf.shape[0], dtype=dtype, device=device)
        Ei_s = torch.zeros_like(Er_s)
        p = physics
        for mask, net in [
            (zf <= p.ridge_z_min, model.net_air),
            ((zf > p.ridge_z_min) & (zf <= p.ridge_z_max), model.net_grat),
            (zf > p.ridge_z_max, model.net_sub),
        ]:
            if mask.any():
                out = net.forward(xf[mask], zf[mask])
                Er_s[mask] = out[:,0]; Ei_s[mask] = out[:,1]

    shape = xg.shape
    Er_net = detach_numpy(Er_s.reshape(shape))
    Ei_net = detach_numpy(Ei_s.reshape(shape))
    z_np   = detach_numpy(zg)
    x_np   = detach_numpy(xg)
    k0     = physics.k0

    if formulation == "free_space":
        Er = Er_net + np.cos(k0*z_np)
        Ei = Ei_net - np.sin(k0*z_np)
    elif formulation == "layered_bg":
        Ebg_r, Ebg_i, _, _ = background_field_np(z_np.ravel(), coeff)
        Er = Er_net + Ebg_r.reshape(shape)
        Ei = Ei_net + Ebg_i.reshape(shape)
    else:
        raise ValueError(formulation)

    mag = np.sqrt(Er**2 + Ei**2)
    return {
        # Field arrays
        "x": x_np, "z": z_np, "eps_r": detach_numpy(epsg),
        "E_real": Er, "E_imag": Ei,
        "E_scat_real": Er_net, "E_scat_imag": Ei_net,
        "magnitude": mag, "phase": np.arctan2(Ei, Er),
        # Metadata
        "field_representation": "total",
        "field_representation_pinn_output": "scattered",
        "formulation": formulation,
        "wavelength": float(physics.wavelength),
        "period": float(physics.period),
        "n_air": float(physics.n_air),
        "n_ridge": float(physics.n_ridge),
        "n_substrate": float(physics.n_substrate),
        "k0": float(physics.k0),
        "domain_height": float(physics.domain_height),
        "z_top_monitor": float(0.08 * physics.domain_height),
        "z_bot_monitor": float(0.92 * physics.domain_height),
        "coordinate_convention": "z=0 top, z increases downward, E_inc=exp(-ik0*z)",
    }


def run_boundary_spectral_diagnostic(model, physics, n_dtn_orders: int = 8) -> dict:
    """Run spectral boundary audit for top and bottom boundaries.

    Returns per-order DtN residuals to diagnose which diffraction orders
    are not satisfying the radiation condition.
    """
    from src.modal_dtn import boundary_spectral_audit
    model.eval()
    top_audit = boundary_spectral_audit(
        model.net_air, 0.0, physics, "top", physics.n_air,
        n_dtn_orders=n_dtn_orders, N_x=256,
    )
    bot_audit = boundary_spectral_audit(
        model.net_sub, physics.domain_height, physics, "bottom", physics.n_substrate,
        n_dtn_orders=n_dtn_orders, N_x=256,
    )
    return {"top": top_audit, "bottom": bot_audit}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_comparison_figure(fields_fs, fields_lb, physics, output_dir, tag):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = fields_fs["x"], fields_fs["z"]
    ext = [x.min(), x.max(), z.max(), z.min()]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    for row_idx, (fields, title) in enumerate([(fields_fs,"Free-space"), (fields_lb,"Layered-bg")]):
        for col_idx, (key, lbl, cmap) in enumerate([
            ("E_real",    "Re{E}",   "RdBu_r"),
            ("E_imag",    "Im{E}",   "RdBu_r"),
            ("magnitude", "|E|",     "viridis"),
            ("phase",     "Phase",   "twilight"),
        ]):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(fields[key], extent=ext, aspect="auto", cmap=cmap, origin="upper")
            ax.set_title(f"{title} — {lbl}")
            ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(output_dir / f"{tag}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_source_figure(src_map, output_dir, tag):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = src_map["x"], src_map["z"]
    ext = [x.min(), x.max(), z.max(), z.min()]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    for ax, key, lbl, cmap in [
        (axes[0], "eps_r",     "εr(x,z)",           "viridis"),
        (axes[1], "eps_bg",    "εbg(x,z)",           "viridis"),
        (axes[2], "delta_eps", "δε = εr−εbg",        "RdBu_r"),
        (axes[3], "source_mag","|δε·Ebg|",           "hot"),
    ]:
        im = ax.imshow(src_map[key], extent=ext, aspect="auto", cmap=cmap, origin="upper")
        ax.set_title(lbl); ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Source map — {tag}")
    fig.savefig(output_dir / f"{tag}_source.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Normalised diffraction
# ---------------------------------------------------------------------------


def get_diffraction(fields, physics, n_orders=3):
    from scripts.audit_maxwell_2d import compute_normalised_diffraction
    E_c = fields["E_real"] + 1j*fields["E_imag"]
    x1d = fields["x"][0, :]; z1d = fields["z"][:, 0]
    return compute_normalised_diffraction(E_c, x1d, z1d, physics, n_orders)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Layered-background scattered-field ablation")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--case",           default="lambda_0p8",
                        choices=["lambda_0p8", "grating"])
    parser.add_argument("--reference",      default=None)
    parser.add_argument("--gen-reference",  action="store_true")
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--epochs",         type=int,   default=5000)
    parser.add_argument("--hidden-layers",  type=int,   default=4)
    parser.add_argument("--hidden-width",   type=int,   default=64)
    parser.add_argument("--fourier-levels", type=int,   default=4)
    parser.add_argument("--grating-levels", type=int,   default=0,
                        help="Grating-periodic x features sin/cos(m*G0*x). "
                             "0=off (default). Use >=2 for sub-wavelength period gratings.")
    parser.add_argument("--n-per-region",   type=int,   default=1024)
    parser.add_argument("--n-interface",    type=int,   default=256)
    parser.add_argument("--n-bc",           type=int,   default=256)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--w-pde",   type=float, default=1.0)
    parser.add_argument("--w-E",     type=float, default=500.0)
    parser.add_argument("--w-H",     type=float, default=500.0)
    parser.add_argument("--w-top",   type=float, default=200.0)
    parser.add_argument("--w-bot",   type=float, default=100.0)
    parser.add_argument("--w-modal", type=float, default=0.0,
                        help="Weight for modal data loss using RCWA reference amplitudes. "
                             "Requires --reference. Recommended: 50-200 for sub-wavelength gratings.")
    parser.add_argument("--n-modal-orders", type=int, default=3,
                        help="Number of grating orders to constrain with modal data loss (default 3)")
    parser.add_argument("--use-dtn", action="store_true",
                        help="Replace Robin BCs with modal DtN radiation conditions")
    parser.add_argument("--n-dtn-orders", type=int, default=8,
                        help="Number of Fourier orders each side for DtN (default 8)")
    parser.add_argument("--feature-variant", default="global_k0",
                        choices=["normalized", "global_k0", "local_material_k",
                                 "local_material_plus_grating_x"],
                        help="Feature encoding variant for x/z coordinates (default global_k0)")
    parser.add_argument("--lbg-only", action="store_true",
                        help="Skip free-space baseline, run only layered-background variant")
    parser.add_argument("--output-dir", default="outputs/lbg")
    parser.add_argument("--flat-test",  action="store_true",
                        help="Run flat-interface test (ridge contrast=0)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    config = load_config(config_path)

    out = Path(args.output_dir)
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    if args.case == "lambda_0p8":
        physics = make_lambda_0p8(config.physics)
        tag = "lambda_0p8"
    else:
        physics = config.physics
        tag = "grating"

    # Reference field
    ref_path = None
    if args.gen_reference:
        from scripts.generate_reference import solve_rcwa
        ref_path = ROOT / f"outputs/reference_{tag}.npz"
        print(f"  Generating RCWA reference → {ref_path}")
        x_r, z_r, Er_r, Ei_r, _amps_r = solve_rcwa(physics, N_harmonics=75)
        np.savez(ref_path, x=x_r, z=z_r, E_real=Er_r, E_imag=Ei_r)
    elif args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute(): ref_path = ROOT / ref_path

    # Background coefficients
    coeff = compute_background_coefficients(physics)

    # Load RCWA modal amplitudes (for modal data loss)
    rcwa_amps = None
    if args.w_modal > 0:
        if ref_path and ref_path.exists():
            try:
                rcwa_amps = load_rcwa_amplitudes(ref_path)
                print(f"  RCWA amplitudes loaded: N={rcwa_amps['N_harmonics']}  "
                      f"R+T={rcwa_amps['R_total']+rcwa_amps['T_total']:.6f}")
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Cannot load RCWA amplitudes: {e}")
                print(f"  Modal data loss disabled.")
                args.w_modal = 0.0
        else:
            print(f"  WARNING: --w-modal requires --reference. Modal data loss disabled.")
            args.w_modal = 0.0
    print(f"\n=== Layered-background ablation  [{tag}] ===")
    print(f"  r_bg = {coeff['r_eff']:.4f}  |r|={abs(coeff['r_eff']):.4f}")
    print(f"  tau  = {coeff['tau']:.4f}  |t|={abs(coeff['tau']):.4f}")
    print(f"  Energy: R+T = {coeff['reflectance']+coeff['transmittance']:.8f}")
    print(f"  Contrast source region: ridge only  (delta_eps={physics.n_ridge**2-physics.n_substrate**2:.4f})")
    print(f"  Source in substrate: 0.0  (vs free-space: {physics.n_substrate**2-1:.4f})")

    # Save source figure
    src_map = compute_source_map(physics, coeff)
    save_source_figure(src_map, out, tag)
    print(f"  Source map saved: {out}/{tag}_source.png")

    # Source magnitude by region
    delta = src_map["delta_eps"]
    sm    = src_map["source_mag"]
    Z2d   = src_map["z"]
    in_air    = Z2d <  physics.ridge_z_min
    in_grat   = (Z2d >= physics.ridge_z_min) & (Z2d <= physics.ridge_z_max)
    in_sub    = Z2d >  physics.ridge_z_max
    print(f"  Source by region: air={sm[in_air].max():.4f}  grating={sm[in_grat].max():.4f}  sub={sm[in_sub].max():.4f}")

    # Flat-interface test (delta_eps = 0)
    if args.flat_test:
        print(f"\n  --- Flat-interface test (ridge_contrast=0) ---")
        p_flat = PhysicsConfig(
            wavelength=physics.wavelength, n_air=physics.n_air,
            n_ridge=physics.n_substrate,  # no contrast
            n_substrate=physics.n_substrate,
            period=physics.period, ridge_width=physics.ridge_width,
            ridge_height=physics.ridge_height, domain_height=physics.domain_height,
            ridge_base_fraction=physics.ridge_base_fraction,
        )
        coeff_flat = compute_background_coefficients(p_flat)
        set_seed(args.seed)
        model_flat = Maxwell2DDD_Variant(p_flat, "global_k0", args.hidden_layers, args.hidden_width, args.fourier_levels)
        model_flat = model_flat.to(device=device, dtype=dtype)
        pts_flat = sample_nd_points(p_flat, args.n_per_region, args.n_interface, args.n_bc, args.n_bc, device, dtype, seed=args.seed)
        def lf_fn(m, p2, ph): return layered_bg_loss(m, p2, ph, coeff_flat, args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot)
        _, _ = train(model_flat, pts_flat, p_flat, lambda m,p2,ph: lf_fn(m,p2,ph), min(args.epochs, 2000), args.lr, args.seed)
        fields_flat = evaluate(model_flat, p_flat, device, dtype, "layered_bg", coeff_flat)
        scat_max = np.max(np.sqrt(fields_flat["E_scat_real"]**2 + fields_flat["E_scat_imag"]**2))
        print(f"  Max |E_scat| (should be ~0): {scat_max:.4e}")
        # Compare total field to E_bg
        Ebg_r, Ebg_i, _, _ = background_field_np(fields_flat["z"].ravel(), coeff_flat)
        diff = np.max(np.abs(fields_flat["E_real"].ravel() - Ebg_r) + np.abs(fields_flat["E_imag"].ravel() - Ebg_i))
        print(f"  Max |E_total - E_bg| (should be ~0): {diff:.4e}")

    # Shared collocation points (identical for both formulations)
    set_seed(args.seed)
    pts = sample_nd_points(physics, args.n_per_region, args.n_interface, args.n_bc, args.n_bc, device, dtype, seed=args.seed)

    # Model factory: use ND+grating when grating_levels>0, else Variant
    def _make_model(seed_offset=0):
        set_seed(args.seed + seed_offset)
        if args.grating_levels > 0:
            from src.maxwell_2d_nondim import Maxwell2DDD_ND
            m = Maxwell2DDD_ND(
                physics,
                hidden_layers=args.hidden_layers,
                hidden_width=args.hidden_width,
                num_fourier_levels=args.fourier_levels,
                num_grating_levels=args.grating_levels,
            )
            n_par = sum(p2.numel() for p2 in m.parameters())
            G0_over_k0 = (2 * np.pi / physics.period) / physics.k0
            print(f"  Model: ND+grating  grating_levels={args.grating_levels}  "
                  f"G0/k0={G0_over_k0:.4f}  params={n_par:,}")
        else:
            m = Maxwell2DDD_Variant(
                physics, args.feature_variant,
                args.hidden_layers, args.hidden_width, args.fourier_levels,
            )
            n_par = sum(p2.numel() for p2 in m.parameters())
            print(f"  Model: Variant({args.feature_variant})  params={n_par:,}")
        return m.to(device=device, dtype=dtype)

    # ---- Free-space variant (optional baseline) ----
    if not args.lbg_only:
        print(f"\n  === Free-space (original) ===")
        model_fs = _make_model(seed_offset=0)
        def fs_fn(m, p, ph):
            return free_space_loss(m, p, ph, args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot)
        hist_fs, best_fs = train(model_fs, pts, physics, fs_fn, args.epochs, args.lr, args.seed)
        fields_fs = evaluate(model_fs, physics, device, dtype, "free_space")
    else:
        print(f"\n  [--lbg-only: skipping free-space baseline]")
        model_fs = None; hist_fs = [{}]; best_fs = float("nan")
        fields_fs = None

    # ---- Layered-background variant (with or without DtN) ----
    bc_label = "DtN" if args.use_dtn else "Robin"
    feat_label = f"grating_levels={args.grating_levels}" if args.grating_levels > 0 else args.feature_variant
    print(f"\n  === Layered-background ({bc_label} BC, {feat_label}) ===")
    if args.use_dtn:
        G0 = 2 * np.pi / physics.period
        pm1_in_air = G0 > physics.k0 * physics.n_air
        print(f"  Modal DtN: {args.n_dtn_orders} orders each side  "
              f"period={physics.period:.4f}  pm1 in air: {'evanescent' if pm1_in_air else 'propagating'}")
    if args.grating_levels > 0:
        G0_k0 = (2 * np.pi / physics.period) / physics.k0
        print(f"  Grating features: G0/k0={G0_k0:.4f}  levels 1..{args.grating_levels}  "
              f"=> sin/cos(m*G0*x) for m=1..{args.grating_levels}")
    model_lb = _make_model(seed_offset=0)  # same seed as fs for fair comparison
    def lb_fn_no_modal(m, p, ph):
        return layered_bg_loss(m, p, ph, coeff,
                               args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot,
                               use_dtn=args.use_dtn, n_dtn_orders=args.n_dtn_orders,
                               rcwa_amps=None, w_modal=0.0,
                               n_modal_orders=args.n_modal_orders)
    def lb_fn(m, p, ph):
        return layered_bg_loss(m, p, ph, coeff,
                               args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot,
                               use_dtn=args.use_dtn, n_dtn_orders=args.n_dtn_orders,
                               rcwa_amps=rcwa_amps, w_modal=args.w_modal,
                               n_modal_orders=args.n_modal_orders)
    warmup = args.epochs // 2 if args.w_modal > 0 else 0
    if warmup > 0:
        print(f"  Staged training: {warmup} warmup + {args.epochs-warmup} modal-phase")
        print(f"  Modal loss: initial w=0.1, ramp to w={args.w_modal} over modal phase")
        print(f"  R+T rejection: episode rejected if R+T > 1.05")

        # Phase 1: PDE + DtN only, full cosine schedule
        print(f"\n  Phase 1 ({warmup} epochs, PDE+DtN):")
        hist_w, _ = train(
            model_lb, pts, physics,
            loss_fn=lb_fn_no_modal,
            epochs=warmup, lr=args.lr, seed=args.seed,
            best_from_epoch=1,
        )

        # Phase 2: PDE + DtN + ramping modal loss, fresh cosine schedule
        print(f"\n  Phase 2 ({args.epochs-warmup} epochs, PDE+DtN+Modal ramp):")
        modal_epochs = args.epochs - warmup

        def lb_fn_ramped(m, p, ph, ep=None, total_ep=None):
            """Modal weight ramps linearly from 0.1 to w_modal over modal phase."""
            if ep is not None and total_ep is not None:
                ramp = min(1.0, ep / (total_ep * 0.5))  # full weight at midpoint
                w_cur = 0.1 + (args.w_modal - 0.1) * ramp
            else:
                w_cur = args.w_modal
            return layered_bg_loss(m, p, ph, coeff,
                                   args.w_pde, args.w_E, args.w_H, args.w_top, args.w_bot,
                                   use_dtn=args.use_dtn, n_dtn_orders=args.n_dtn_orders,
                                   rcwa_amps=rcwa_amps, w_modal=w_cur,
                                   n_modal_orders=args.n_modal_orders)

        hist_m, best_lb = train(
            model_lb, pts, physics,
            loss_fn=lb_fn_ramped,
            epochs=modal_epochs,
            lr=args.lr * 0.3,
            seed=args.seed + 1,
            best_from_epoch=1,
            rt_reject_threshold=1.05,
            rcwa_amps=rcwa_amps,
            physics_for_rt=p08 if 'p08' in dir() else physics,
        )
        hist_lb = hist_w + hist_m
    else:
        hist_lb, best_lb = train(
            model_lb, pts, physics,
            loss_fn=lb_fn,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            best_from_epoch=1,
        )
    fields_lb = evaluate(model_lb, physics, device, dtype, "layered_bg", coeff)

    # ---- Spectral boundary diagnostic ----
    print(f"\n  === Spectral boundary diagnostic ({bc_label}) ===")
    spec_diag = run_boundary_spectral_diagnostic(model_lb, physics, n_dtn_orders=args.n_dtn_orders)
    print(f"  TOP boundary (air, z=0):")
    for row in spec_diag["top"]["propagating_orders"]:
        print(f"    m={row['m']:+d}  kz={row['kz_m_re']:.4f}  "
              f"|E_m|={row['E_m_abs']:.4f}  |H_m|={row['H_m_abs']:.4f}  "
              f"|H_DtN|={row['H_m_dtn_abs']:.4f}  res={row['relative_residual']:.3f}")
    print(f"  BOTTOM boundary (substrate, z={physics.domain_height:.3f}):")
    for row in spec_diag["bottom"]["propagating_orders"]:
        print(f"    m={row['m']:+d}  kz={row['kz_m_re']:.4f}  "
              f"|E_m|={row['E_m_abs']:.4f}  |H_m|={row['H_m_abs']:.4f}  "
              f"|H_DtN|={row['H_m_dtn_abs']:.4f}  res={row['relative_residual']:.3f}")
    top_max_res  = spec_diag["top"]["summary"]["max_relative_residual_propagating"]
    bot_max_res  = spec_diag["bottom"]["summary"]["max_relative_residual_propagating"]
    print(f"  max_residual_propagating: top={top_max_res:.4f}  bot={bot_max_res:.4f}")

    # ---- Comparison figure (only when both variants ran) ----
    if fields_fs is not None:
        save_comparison_figure(fields_fs, fields_lb, physics, out, tag)

    def _compare(fields, formulation, label, best_total_val):
        """Run dual total/scattered comparison and modal extraction."""
        print(f"\n  [{label}]  best_total={best_total_val:.4e}")
        model = model_fs if formulation == "free_space" else model_lb
        h = (hist_fs if formulation == "free_space" else hist_lb)[-1]
        print(f"    pde_air={h['pde_air']:.4e}  pde_grat={h['pde_grat']:.4e}  pde_sub={h['pde_sub']:.4e}")
        print(f"    E_int={h['E_int1']+h['E_int2']:.4e}  top={h['top']:.4e}  bot={h['bottom']:.4e}")

        dual = {}
        modal_pinn = {}
        modal_rcwa = {}

        if ref_path and ref_path.exists():
            ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
            x1d = fields["x"][0, :]; z1d = fields["z"][:, 0]
            ref_Er, ref_Ei, _ = interpolate_reference_to_grid(ref_data, x1d, z1d)

            # Dual comparison: PINN scatter vs RCWA total
            dual = compare_fields(
                fields["E_scat_real"], fields["E_scat_imag"],
                ref_Er, ref_Ei,
                fields["z"], physics,
                formulation=formulation,
            )
            print_comparison_summary(dual)

            # Modal extraction from PINN total field
            E_total_pinn = dual["pinn_E_total_r"] + 1j * dual["pinn_E_total_i"]
            modal_pinn = extract_modal_amplitudes(
                E_total_pinn, x1d, z1d, physics, n_orders=5,
            )

            # Modal comparison vs RCWA amplitudes stored in NPZ
            n_harm = 25 if tag == "grating" else 75
            modal_rcwa = compare_modal_with_rcwa(modal_pinn, str(ref_path), n_harm)
            print_comparison_summary(dual, rcwa_modal=modal_rcwa)

        return dual, modal_pinn, modal_rcwa

    print(f"\n{'='*70}")
    print("COMPARISON: free-space vs. layered-background")
    print(f"{'='*70}")
    if fields_fs is not None:
        dual_fs, modal_pinn_fs, modal_rcwa_fs = _compare(
            fields_fs, "free_space", "free_space", best_fs)
    else:
        dual_fs, modal_pinn_fs, modal_rcwa_fs = {}, {}, {}
    dual_lb, modal_pinn_lb, modal_rcwa_lb = _compare(
        fields_lb, "layered_bg", "layered_bg", best_lb)

    # Summary comparison table
    print(f"\n{'Metric':45} {'free_space':>15} {'layered_bg':>15}")
    print("-" * 77)
    for label_key in (
        "total/complex_l2", "total/magnitude_l2", "total/phase_rmse_deg",
        "scattered/complex_l2", "scattered/magnitude_l2", "scattered/phase_rmse_deg",
    ):
        vfs = dual_fs.get(label_key, float("nan"))
        vlb = dual_lb.get(label_key, float("nan"))
        if isinstance(vfs, float) and isinstance(vlb, float):
            print(f"  {label_key:43} {vfs:15.4f} {vlb:15.4f}")

    h_fs = hist_fs[-1]; h_lb = hist_lb[-1]
    for k in ("pde_air", "pde_grat", "pde_sub", "top", "bottom"):
        print(f"  {'pde/'+k if 'pde' not in k else k:43} "
              f"{h_fs.get(k, float('nan')):15.4e} {h_lb.get(k, float('nan')):15.4e}")

    # R+T energy check
    e_fs = modal_pinn_fs.get("energy_check", float("nan"))
    e_lb = modal_pinn_lb.get("energy_check", float("nan"))
    print(f"  {'modal_energy_check (R+T)':43} {e_fs:15.4f} {e_lb:15.4f}")
    print(f"{'='*77}")

    # JSON report
    def _serialise(d):
        """Convert numpy/complex to JSON-safe types."""
        if isinstance(d, dict):
            return {k: _serialise(v) for k, v in d.items()
                    if not isinstance(v, np.ndarray)}
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        if isinstance(d, complex):
            return {"real": float(d.real), "imag": float(d.imag)}
        if isinstance(d, (list, tuple)):
            return [_serialise(x) for x in d]
        return d

    report = {
        "case": tag, "epochs": args.epochs, "seed": args.seed,
        "boundary_condition": "dtn" if args.use_dtn else "robin",
        "n_dtn_orders": args.n_dtn_orders if args.use_dtn else None,
        "feature_variant": args.feature_variant,
        "metadata": {
            "wavelength":  float(physics.wavelength),
            "period":      float(physics.period),
            "n_air":       float(physics.n_air),
            "n_ridge":     float(physics.n_ridge),
            "n_substrate": float(physics.n_substrate),
            "k0":          float(physics.k0),
            "domain_height": float(physics.domain_height),
            "field_representation_rcwa": "total",
            "field_representation_pinn": "scattered",
            "coordinate_convention": "z=0 top, z increases downward, E_inc=exp(-ik0*z)",
            "z_top_monitor": float(0.08 * physics.domain_height),
            "z_bot_monitor": float(0.92 * physics.domain_height),
        },
        "background_coeff": {k: str(v) for k, v in coeff.items()},
        "source_region_max": {
            "air":     float(sm[in_air].max()),
            "grating": float(sm[in_grat].max()),
            "substrate": float(sm[in_sub].max()),
        },
        "spectral_boundary_diagnostic": {
            "top_max_residual_propagating":    top_max_res,
            "bottom_max_residual_propagating": bot_max_res,
            "top_propagating_orders":    spec_diag["top"]["propagating_orders"],
            "bottom_propagating_orders": spec_diag["bottom"]["propagating_orders"],
        },
        "free_space": {
            "dual_comparison": _serialise(dual_fs),
            "modal_pinn": _serialise(modal_pinn_fs),
            "modal_rcwa_vs_pinn": _serialise(modal_rcwa_fs),
            "final_losses": h_fs,
        },
        "layered_bg": {
            "dual_comparison": _serialise(dual_lb),
            "modal_pinn": _serialise(modal_pinn_lb),
            "modal_rcwa_vs_pinn": _serialise(modal_rcwa_lb),
            "final_losses": h_lb,
        },
    }
    mp = out / f"{tag}_lbg_report.json"
    with mp.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {mp}")
    print(f"  Figures: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
