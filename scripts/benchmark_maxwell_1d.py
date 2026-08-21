#!/usr/bin/env python3
"""Benchmark: second-order Helmholtz vs. first-order Maxwell PINN (1-D layered).

Trains both formulations on the same layered benchmark and reports:
    - PDE residual
    - complex L2 error
    - magnitude L2 error
    - phase RMSE (deg)
    - interface and boundary losses

Pass criterion: relative_complex_l2 < 0.10 AND relative_l2_phase_aligned < 0.05

Usage::

    .venv/bin/python scripts/benchmark_maxwell_1d.py \\
        --epochs 5000 --device cpu --output-dir outputs/benchmarks
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

from src.benchmarks import (
    LayeredMediumBenchmark,
    evaluate_benchmark_errors,
    evaluate_benchmark_errors_by_region,
    plot_complex_field_comparison,
)
from src.domain_decomp import (
    DomainDecompLayered,
    analytical_interface_errors,
    analytical_pde_residuals,
    bottom_bc_loss_dd,
    interface_field_loss,
    interface_flux_loss,
    pde_residual_subdomain,
    top_bc_loss_dd,
)
from src.maxwell_1d import (
    Maxwell1DLayered,
    MaxwellSubdomainMLP,
    analytical_H_np,
    maxwell_bottom_bc,
    maxwell_interface_loss,
    maxwell_pde_residual,
    maxwell_top_bc,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Interior sampling
# ---------------------------------------------------------------------------


def _sample(z_lo, z_hi, n, margin, seed, device, dtype):
    rng = np.random.default_rng(seed)
    lo, hi = z_lo + margin, z_hi - margin
    if lo >= hi:
        return torch.full((n,), (z_lo + z_hi) / 2.0, dtype=dtype, device=device)
    return torch.as_tensor(rng.uniform(lo, hi, n), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Helmholtz (second-order) training — same as benchmark_layered_dd
# ---------------------------------------------------------------------------


def train_helmholtz(bm, epochs, hidden_layers, hidden_width, fourier_levels,
                    device, dtype, w_pde=1.0, w_field=500.0, w_flux=500.0,
                    w_top=200.0, w_bot=100.0, margin=1e-4, seed=42):
    set_seed(seed)
    model = DomainDecompLayered(bm, hidden_layers, hidden_width, fourier_levels)
    model = model.to(device=device, dtype=dtype)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_int, n_if, n_bc = 512, 128, 128
    z_air  = _sample(0.0,           bm.z_slab_top,    n_int*1, margin, seed,   device, dtype)
    z_slab = _sample(bm.z_slab_top, bm.z_slab_bot,    n_int*2, margin, seed+1, device, dtype)
    z_sub  = _sample(bm.z_slab_bot, bm.domain_height, n_int*1, margin, seed+2, device, dtype)
    r = bm.reflection_coefficient()
    history = []

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        ra, ia = pde_residual_subdomain(model.net_air,  z_air,  bm.k0, bm.n_air**2)
        rs, is_= pde_residual_subdomain(model.net_slab, z_slab, bm.k0, bm.n_slab**2)
        ru, iu = pde_residual_subdomain(model.net_sub,  z_sub,  bm.k0, bm.n_sub**2)
        Lp = torch.mean(ra**2+ia**2) + torch.mean(rs**2+is_**2) + torch.mean(ru**2+iu**2)
        Lf = (interface_field_loss(model.net_air, model.net_slab, bm.z_slab_top, n_if, dtype, device) +
              interface_field_loss(model.net_slab, model.net_sub,  bm.z_slab_bot, n_if, dtype, device))
        Lx = (interface_flux_loss( model.net_air, model.net_slab, bm.z_slab_top, bm.k0, n_if, dtype, device) +
              interface_flux_loss( model.net_slab, model.net_sub,  bm.z_slab_bot, bm.k0, n_if, dtype, device))
        Lt = top_bc_loss_dd(model.net_air, 0.0, r, n_bc, dtype, device)
        Lb = bottom_bc_loss_dd(model.net_sub, bm.domain_height, bm.k0, bm.n_sub, n_bc, dtype, device)
        loss = w_pde*Lp + w_field*Lf + w_flux*Lx + w_top*Lt + w_bot*Lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % 500 == 0 or ep == 1:
            history.append({"epoch": ep, "pde": float(Lp.detach()), "total": float(loss.detach())})

    # LBFGS
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=10,
                                history_size=50, line_search_fn="strong_wolfe")
    def _cl():
        lbfgs.zero_grad(set_to_none=True)
        ra,ia  = pde_residual_subdomain(model.net_air,  z_air,  bm.k0, bm.n_air**2)
        rs,is_ = pde_residual_subdomain(model.net_slab, z_slab, bm.k0, bm.n_slab**2)
        ru,iu  = pde_residual_subdomain(model.net_sub,  z_sub,  bm.k0, bm.n_sub**2)
        lp = torch.mean(ra**2+ia**2)+torch.mean(rs**2+is_**2)+torch.mean(ru**2+iu**2)
        lf = (interface_field_loss(model.net_air,model.net_slab,bm.z_slab_top,n_if,dtype,device)+
              interface_field_loss(model.net_slab,model.net_sub,bm.z_slab_bot,n_if,dtype,device))
        lx = (interface_flux_loss(model.net_air,model.net_slab,bm.z_slab_top,bm.k0,n_if,dtype,device)+
              interface_flux_loss(model.net_slab,model.net_sub,bm.z_slab_bot,bm.k0,n_if,dtype,device))
        lt = top_bc_loss_dd(model.net_air,0.0,r,n_bc,dtype,device)
        lb = bottom_bc_loss_dd(model.net_sub,bm.domain_height,bm.k0,bm.n_sub,n_bc,dtype,device)
        l = w_pde*lp+w_field*lf+w_flux*lx+w_top*lt+w_bot*lb
        l.backward(); return l
    for _ in range(200):
        lbfgs.step(_cl)
    return model, history


# ---------------------------------------------------------------------------
# Maxwell (first-order) training
# ---------------------------------------------------------------------------


def train_maxwell(bm, epochs, hidden_layers, hidden_width, fourier_levels,
                  device, dtype, w_pde=1.0, w_E=500.0, w_H=500.0,
                  w_top=200.0, w_bot=100.0, margin=1e-4, seed=42):
    set_seed(seed)
    model = Maxwell1DLayered(bm, hidden_layers, hidden_width, fourier_levels)
    model = model.to(device=device, dtype=dtype)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_int, n_if, n_bc = 512, 128, 128
    z_air  = _sample(0.0,           bm.z_slab_top,    n_int*1, margin, seed,   device, dtype)
    z_slab = _sample(bm.z_slab_top, bm.z_slab_bot,    n_int*2, margin, seed+1, device, dtype)
    z_sub  = _sample(bm.z_slab_bot, bm.domain_height, n_int*1, margin, seed+2, device, dtype)
    r = bm.reflection_coefficient()
    history = []

    def _pde_loss():
        ra,ia,rha,iha = maxwell_pde_residual(model.net_air,  z_air,  bm.k0, bm.n_air**2)
        rs,is_,rhs,ihs = maxwell_pde_residual(model.net_slab, z_slab, bm.k0, bm.n_slab**2)
        ru,iu,rhu,ihu = maxwell_pde_residual(model.net_sub,  z_sub,  bm.k0, bm.n_sub**2)
        return (torch.mean(ra**2+ia**2+rha**2+iha**2) +
                torch.mean(rs**2+is_**2+rhs**2+ihs**2) +
                torch.mean(ru**2+iu**2+rhu**2+ihu**2))

    def _intf_loss():
        LE1, LH1 = maxwell_interface_loss(model.net_air,  model.net_slab, bm.z_slab_top, n_if, dtype, device)
        LE2, LH2 = maxwell_interface_loss(model.net_slab, model.net_sub,  bm.z_slab_bot, n_if, dtype, device)
        return w_E*(LE1+LE2) + w_H*(LH1+LH2)

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        Lp = _pde_loss()
        Li = _intf_loss()
        Lt = maxwell_top_bc(model.net_air, 0.0, r, bm.n_air, n_bc, dtype, device)
        Lb = maxwell_bottom_bc(model.net_sub, bm.domain_height, bm.n_sub, n_bc, dtype, device)
        loss = w_pde*Lp + Li + w_top*Lt + w_bot*Lb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % 500 == 0 or ep == 1:
            history.append({"epoch": ep, "pde": float(Lp.detach()), "total": float(loss.detach())})

    # LBFGS
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=10,
                                history_size=50, line_search_fn="strong_wolfe")
    def _cl():
        lbfgs.zero_grad(set_to_none=True)
        l = w_pde*_pde_loss() + _intf_loss()
        l += w_top * maxwell_top_bc(model.net_air, 0.0, r, bm.n_air, n_bc, dtype, device)
        l += w_bot * maxwell_bottom_bc(model.net_sub, bm.domain_height, bm.n_sub, n_bc, dtype, device)
        l.backward(); return l
    for _ in range(200):
        lbfgs.step(_cl)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, bm, model_type, output_dir, device, dtype):
    """Evaluate either Helmholtz or Maxwell model."""
    Nz   = 512
    z_np = np.linspace(0.0, bm.domain_height, Nz)
    z_t  = torch.as_tensor(z_np, dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        if model_type == "helmholtz":
            out = model.forward(z_t)
        else:
            out = model.forward_E(z_t)
    pinn_r = detach_numpy(out[:, 0])
    pinn_i = detach_numpy(out[:, 1])

    ref_r, ref_i = bm.analytical_field_np(np.zeros(Nz), z_np)
    errors = evaluate_benchmark_errors(pinn_r, pinn_i, ref_r, ref_i)
    region_errors = evaluate_benchmark_errors_by_region(pinn_r, pinn_i, ref_r, ref_i, z_np, bm)

    # Figure with H field for Maxwell
    if model_type == "maxwell":
        ref_Hr, ref_Hi = analytical_H_np(bm, z_np)
        with torch.no_grad():
            full_out = torch.zeros(Nz, 4, dtype=dtype, device=device)
            m_air  = z_t <= bm.z_slab_top
            m_slab = (z_t > bm.z_slab_top) & (z_t <= bm.z_slab_bot)
            m_sub  = z_t > bm.z_slab_bot
            if m_air.any():  full_out[m_air]  = model.net_air.forward(z_t[m_air])
            if m_slab.any(): full_out[m_slab] = model.net_slab.forward(z_t[m_slab])
            if m_sub.any():  full_out[m_sub]  = model.net_sub.forward(z_t[m_sub])
        pinn_Hr = detach_numpy(full_out[:, 2])
        pinn_Hi = detach_numpy(full_out[:, 3])

        fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
        iv = dict(color="gray", ls=":", lw=0.8)
        def _v(ax):
            ax.axvline(bm.z_slab_top, **iv); ax.axvline(bm.z_slab_bot, **iv)

        axes[0,0].plot(z_np, ref_r,  "b-", lw=1.5, label="TMM"); axes[0,0].plot(z_np, pinn_r, "r--", lw=1.2, label="PINN"); _v(axes[0,0]); axes[0,0].set_title("Re{E}"); axes[0,0].legend(fontsize=8)
        axes[0,1].plot(z_np, ref_i,  "b-", lw=1.5); axes[0,1].plot(z_np, pinn_i, "r--", lw=1.2); _v(axes[0,1]); axes[0,1].set_title("Im{E}")
        axes[0,2].plot(z_np, np.sqrt(ref_r**2+ref_i**2), "b-", lw=1.5); axes[0,2].plot(z_np, np.sqrt(pinn_r**2+pinn_i**2), "r--", lw=1.2); _v(axes[0,2]); axes[0,2].set_title("|E|")
        axes[1,0].plot(z_np, ref_Hr, "b-", lw=1.5, label="TMM"); axes[1,0].plot(z_np, pinn_Hr, "r--", lw=1.2, label="PINN"); _v(axes[1,0]); axes[1,0].set_title("Re{H̃}"); axes[1,0].legend(fontsize=8)
        axes[1,1].plot(z_np, ref_Hi, "b-", lw=1.5); axes[1,1].plot(z_np, pinn_Hi, "r--", lw=1.2); _v(axes[1,1]); axes[1,1].set_title("Im{H̃}")
        H_errs = evaluate_benchmark_errors(pinn_Hr, pinn_Hi, ref_Hr, ref_Hi)
        axes[1,2].plot(z_np, np.sqrt(ref_Hr**2+ref_Hi**2), "b-", lw=1.5); axes[1,2].plot(z_np, np.sqrt(pinn_Hr**2+pinn_Hi**2), "r--", lw=1.2); _v(axes[1,2]); axes[1,2].set_title("|H̃|")
        for ax in axes.flat:
            ax.set_xlabel("z (λ)")
        e = errors
        fig.suptitle(f"Maxwell 1D  L2_complex={e['relative_complex_l2']:.3e}  "
                     f"L2_aligned={e['relative_l2_phase_aligned']:.3e}  phase_rmse={e['phase_rmse_deg']:.1f}°\n"
                     f"H: L2_complex={H_errs['relative_complex_l2']:.3e}")
        fig.savefig(output_dir / "maxwell_1d_field.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / "maxwell_1d_field.pdf", bbox_inches="tight")
        plt.close(fig)
        errors["H_complex_l2"] = H_errs["relative_complex_l2"]
    else:
        plot_complex_field_comparison(
            z_np, pinn_r, pinn_i, ref_r, ref_i, bm,
            str(output_dir / "helmholtz_dd_field.png"),
            title=f"Helmholtz DD  complex_L2={errors['relative_complex_l2']:.3e}",
        )

    return errors, region_errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Helmholtz vs. Maxwell 1-D PINN comparison")
    parser.add_argument("--epochs",        type=int,   default=5000)
    parser.add_argument("--hidden-layers", type=int,   default=4)
    parser.add_argument("--hidden-width",  type=int,   default=64)
    parser.add_argument("--fourier-levels",type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="cpu")
    parser.add_argument("--output-dir",    type=str,   default="outputs/benchmarks")
    parser.add_argument("--skip-helmholtz",action="store_true", help="Skip Helmholtz (faster testing)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    bm = LayeredMediumBenchmark(k0=2.0 * np.pi)
    device = resolve_device(args.device)
    dtype  = resolve_training_dtype("float64", device)

    # --- Analytical verification ---
    print("\n=== Analytical solution verification ===")
    intf = analytical_interface_errors(bm)
    pde_an = analytical_pde_residuals(bm)
    r = bm.reflection_coefficient(); t = bm.transmission_coefficient()
    energy = abs(r)**2 + abs(t)**2 * bm.n_sub / bm.n_air
    print(f"  Interface errors: max = {max(v for k,v in intf.items()):.2e}  (should be ~1e-16)")
    print(f"  Analytical PDE residuals: {[(k,f'{v:.1e}') for k,v in pde_an.items() if k!='note']}")
    print(f"  Energy conservation: {energy:.10f}")
    print(f"  r = {r:.4f}  |r|={abs(r):.4f},  t = {t:.4f}  |t|={abs(t):.4f}")

    # --- Verify analytical H field ---
    print("\n=== Analytical H field verification ===")
    z_check = np.array([bm.z_slab_top - 0.01, bm.z_slab_top + 0.01,
                        bm.z_slab_bot - 0.01, bm.z_slab_bot + 0.01])
    ref_Er, ref_Ei = bm.analytical_field_np(np.zeros(4), z_check)
    ref_Hr, ref_Hi = analytical_H_np(bm, z_check)
    # Verify dE/dz = -ik0*H̃ using finite differences near each point
    dz = 1e-7
    Er_p, Ei_p = bm.analytical_field_np(np.zeros(4), z_check + dz)
    Er_m, Ei_m = bm.analytical_field_np(np.zeros(4), z_check - dz)
    dEr_dz = (Er_p - Er_m) / (2*dz)
    dEi_dz = (Ei_p - Ei_m) / (2*dz)
    # Equation 1: dE_r/dz = k0*H_i, dE_i/dz = -k0*H_r
    err1r = np.max(np.abs(dEr_dz - bm.k0 * ref_Hi))
    err1i = np.max(np.abs(dEi_dz + bm.k0 * ref_Hr))
    print(f"  Maxwell eq(1) residual: Re={err1r:.2e}  Im={err1i:.2e}  (should be ~1e-6)")
    assert err1r < 1e-4 and err1i < 1e-4, "Analytical H field fails Maxwell equation (1)"
    print(f"  Sign convention check: PASSED")

    results = {}

    # --- Helmholtz (second-order) ---
    if not args.skip_helmholtz:
        print(f"\n=== Training Helmholtz DD-PINN ({args.epochs} epochs) ===")
        helm_model, helm_hist = train_helmholtz(
            bm, args.epochs, args.hidden_layers, args.hidden_width, args.fourier_levels,
            device, dtype)
        print(f"  Final Adam epoch: pde={helm_hist[-1]['pde']:.3e}")
        helm_errors, helm_region = evaluate_model(helm_model, bm, "helmholtz", out, device, dtype)
        results["helmholtz"] = {"errors": helm_errors, "region": helm_region}
        print(f"\n  Helmholtz results:")
        for k, v in helm_errors.items():
            print(f"    {k}: {v:.4e}")
        print(f"  Per-region (Helmholtz):")
        for reg, re in helm_region.items():
            print(f"    [{reg}] complex_L2={re.get('relative_complex_l2',float('nan')):.3e}  phase_rmse={re.get('phase_rmse_deg',float('nan')):.1f}°")

    # --- Maxwell (first-order) ---
    print(f"\n=== Training Maxwell 1st-order PINN ({args.epochs} epochs) ===")
    maxw_model, maxw_hist = train_maxwell(
        bm, args.epochs, args.hidden_layers, args.hidden_width, args.fourier_levels,
        device, dtype)
    print(f"  Final Adam epoch: pde={maxw_hist[-1]['pde']:.3e}")
    maxw_errors, maxw_region = evaluate_model(maxw_model, bm, "maxwell", out, device, dtype)
    results["maxwell"] = {"errors": maxw_errors, "region": maxw_region}
    print(f"\n  Maxwell results:")
    for k, v in maxw_errors.items():
        print(f"    {k}: {v:.4e}")
    print(f"  Per-region (Maxwell):")
    for reg, re in maxw_region.items():
        print(f"    [{reg}] complex_L2={re.get('relative_complex_l2',float('nan')):.3e}  phase_rmse={re.get('phase_rmse_deg',float('nan')):.1f}°")

    # --- Comparison table ---
    print("\n=== Comparison ===")
    header = f"{'Metric':<35} {'Helmholtz':>15} {'Maxwell':>15}"
    print(header)
    print("-" * len(header))
    metrics = ["relative_complex_l2", "relative_l2_phase_aligned", "relative_l2_magnitude",
               "phase_rmse_deg", "global_phase_offset_deg"]
    for m in metrics:
        h_val = results.get("helmholtz", {}).get("errors", {}).get(m, float("nan"))
        mw_val = maxw_errors.get(m, float("nan"))
        print(f"  {m:<33} {h_val:>15.4e} {mw_val:>15.4e}")

    # --- Save JSON ---
    report_path = out / "maxwell_vs_helmholtz_report.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")

    # --- Pass criteria on Maxwell ---
    cx = maxw_errors["relative_complex_l2"]
    al = maxw_errors["relative_l2_phase_aligned"]
    mg = maxw_errors["relative_l2_magnitude"]
    passed = cx < 0.10 and al < 0.05 and mg < 0.05
    print(f"\n  Pass criteria (Maxwell):")
    print(f"    relative_complex_l2       < 0.10: {cx:.3e}  {'PASS' if cx < 0.10 else 'FAIL'}")
    print(f"    relative_l2_phase_aligned < 0.05: {al:.3e}  {'PASS' if al < 0.05 else 'FAIL'}")
    print(f"    relative_l2_magnitude     < 0.05: {mg:.3e}  {'PASS' if mg < 0.05 else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
