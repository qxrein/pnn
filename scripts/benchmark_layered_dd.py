#!/usr/bin/env python3
"""Benchmark 2b: 1-D layered-medium PINN with domain decomposition.

Physical problem
----------------
One spatial dimension z only (the problem is invariant in x).
Three uniform layers:

    Air:   z ∈ [0,          z_slab_top]   k1 = k0 * n_air
    Slab:  z ∈ [z_slab_top,  z_slab_bot]  k2 = k0 * n_slab
    Sub:   z ∈ [z_slab_bot,  domain_height] k3 = k0 * n_sub

PDE in each layer j:
    d²E_j/dz² + k_j² E_j = 0

In the physics-scaled coordinate  ξ_j = k_j*(z - z_lo_j):
    d²E_j/dξ_j² + E_j = 0

Interface conditions at z = z_int (scalar TE, μr = 1):
    E_left(z_int) = E_right(z_int)                     [field continuity]
    dE_left/dz    = dE_right/dz    [flux continuity, PHYSICAL z-derivative]

Note on local coordinates
--------------------------
Each subnet predicts E_j as a function of ξ_j = k_j*(z - z_lo_j).
The physical derivative is:
    dE_j/dz = k_j * dE_j/dξ_j

The interface_flux_loss enforces dE/dz continuity using autograd through
physical z directly, so the k_j scaling is handled correctly by the chain
rule:  d/dz → k_j * d/dξ_j.

Success criterion
-----------------
    relative_complex_l2         < 0.10   (complex field, phase-sensitive)
    relative_l2_phase_aligned   < 0.05   (after global phase removal)
    relative_l2_magnitude       < 0.05

Usage::

    .venv/bin/python scripts/benchmark_layered_dd.py \\
        --epochs 5000 --output-dir outputs/benchmarks
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

from src.benchmarks import (
    LayeredMediumBenchmark,
    evaluate_benchmark_errors,
    evaluate_benchmark_errors_by_region,
    plot_complex_field_comparison,
)
from src.domain_decomp import (
    DomainDecompLayered,
    SubdomainMLP,
    analytical_interface_errors,
    analytical_pde_residuals,
    bottom_bc_loss_dd,
    interface_field_loss,
    interface_flux_loss,
    pde_residual_subdomain,
    top_bc_loss_dd,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Interior sampling (interface-margin excluded)
# ---------------------------------------------------------------------------


def _sample_interior(
    z_lo: float,
    z_hi: float,
    n: int,
    margin: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform samples in (z_lo+margin, z_hi-margin)."""
    rng = np.random.default_rng(seed)
    lo, hi = z_lo + margin, z_hi - margin
    if lo >= hi:
        return torch.full((n,), (z_lo + z_hi) / 2.0, dtype=dtype, device=device)
    return torch.as_tensor(rng.uniform(lo, hi, n), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Coordinate and convention printout
# ---------------------------------------------------------------------------


def print_coordinate_diagnostics(bm: LayeredMediumBenchmark) -> None:
    k0, k1 = bm.k0, bm.k0 * bm.n_air
    k2, k3 = bm.k0 * bm.n_slab, bm.k0 * bm.n_sub
    h = bm.z_slab_bot - bm.z_slab_top

    print("\n=== Coordinate and convention diagnostics ===")
    print(f"  Time convention:      exp(+iωt) suppressed")
    print(f"  Propagation:          E_inc = exp(-i k0 z)  (downward, +z)")
    print(f"  At z=0:               E_inc = 1 + 0i")
    print(f"  Problem:              1-D in z (x-invariant)")
    print()
    print(f"  Domain:   z ∈ [0, {bm.domain_height}]  (units of λ)")
    print(f"  Air:      z ∈ [0,      {bm.z_slab_top}]   k1 = {k1:.6f}")
    print(f"  Slab:     z ∈ [{bm.z_slab_top},  {bm.z_slab_bot}]   k2 = {k2:.6f}  thickness = {h:.4f}λ")
    print(f"  Sub:      z ∈ [{bm.z_slab_bot},  {bm.domain_height}]   k3 = {k3:.6f}")
    print()
    print(f"  ξ_j = k_j*(z - z_lo_j)  =>  PDE: d²E/dξ² + E = 0 in each layer")
    print(f"  ξ ranges:")
    print(f"    Air:   ξ1 ∈ [0, {k1 * bm.z_slab_top:.4f}]  (k1*h_air)")
    print(f"    Slab:  ξ2 ∈ [0, {k2 * h:.4f}]  (k2*h_slab)")
    print(f"    Sub:   ξ3 ∈ [0, {k3 * (bm.domain_height - bm.z_slab_bot):.4f}]  (k3*h_sub)")
    print()
    print(f"  Interface flux condition:  dE_left/dz = dE_right/dz  (physical z)")
    print(f"  In local coords:  k_left * dE_left/dξ_left = k_right * dE_right/dξ_right")
    print(f"  The autograd chain rule handles this correctly when differentiating w.r.t. physical z.")

    # Assertions
    assert h > 0, "slab thickness must be positive"
    assert bm.z_slab_top > 0
    assert bm.z_slab_bot > bm.z_slab_top
    assert bm.z_slab_bot < bm.domain_height
    print(f"  Assertions: PASSED")


def print_analytical_diagnostics(bm: LayeredMediumBenchmark) -> tuple[dict, dict]:
    print("\n=== Analytical TMM solution verification ===")
    c = bm._tmm_coefficients()
    k1, k2, k3 = c['k1'], c['k2'], c['k3']

    # Interface errors (machine precision)
    intf = analytical_interface_errors(bm)
    print(f"  Interface BCs (machine-precision expected ~1e-15):")
    for k, v in intf.items():
        status = "OK" if v < 1e-10 else "FAIL"
        print(f"    {k:30s}: {v:.3e}  [{status}]")

    # Exact analytical PDE residuals
    pde_r = analytical_pde_residuals(bm)
    print(f"  Analytical PDE residuals (exact: d²E/dz² = -k²E  =>  residual = 0):")
    for k, v in pde_r.items():
        if k == "note":
            continue
        print(f"    {k:30s}: {v:.3e}  (floating-point zero)")

    r = bm.reflection_coefficient()
    t = bm.transmission_coefficient()
    energy = abs(r)**2 + abs(t)**2 * bm.n_sub / bm.n_air
    print(f"\n  r = {r:.6f}   |r| = {abs(r):.6f}")
    print(f"  t = {t:.6f}   |t| = {abs(t):.6f}")
    print(f"  Energy: |r|² + |t|²*(n_sub/n_air) = {energy:.10f}  (should be 1.0)")
    assert abs(energy - 1.0) < 1e-8, f"Energy not conserved: {energy}"
    print(f"  Energy conservation: PASSED")

    # Verify analytical interface errors are acceptable
    for k, v in intf.items():
        assert v < 1e-10, f"Analytical interface error too large: {k} = {v:.3e}"
    print(f"  All interface assertions: PASSED")

    return intf, pde_r


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_dd(
    bm: LayeredMediumBenchmark,
    epochs: int = 5000,
    n_int: int = 512,
    n_if:  int = 128,
    n_bc:  int = 128,
    lr: float = 5e-4,
    hidden_layers: int = 4,
    hidden_width:  int = 64,
    num_fourier_levels: int = 4,
    w_pde:   float = 1.0,
    w_field: float = 500.0,
    w_flux:  float = 500.0,
    w_top:   float = 200.0,
    w_bot:   float = 100.0,
    margin:  float = 1e-4,
    device_str: str = "cpu",
    seed: int = 42,
) -> tuple[DomainDecompLayered, list[dict]]:
    """Train the domain-decomposition PINN.

    Loss weights rationale
    ----------------------
    Interface and boundary losses are strongly penalised so the subdomains
    are coupled.  PDE weight is 1 to keep the PDE residuals on a natural scale.
    Large w_field/w_flux can make those terms near-zero while pde > 0;
    this is expected and correct — the network will converge to the PDE
    solution as training proceeds.
    """
    set_seed(seed)
    device = resolve_device(device_str)
    dtype  = resolve_training_dtype("float64", device)

    model = DomainDecompLayered(bm, hidden_layers, hidden_width, num_fourier_levels)
    model = model.to(device=device, dtype=dtype)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Interior points (interface-free)
    z_air  = _sample_interior(0.0,           bm.z_slab_top,    n_int,   margin, seed,   device, dtype)
    z_slab = _sample_interior(bm.z_slab_top, bm.z_slab_bot,    n_int*2, margin, seed+1, device, dtype)
    z_sub  = _sample_interior(bm.z_slab_bot, bm.domain_height, n_int,   margin, seed+2, device, dtype)

    r = bm.reflection_coefficient()
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # PDE losses — ξ-coordinate, scale-free: d²E/dξ² + E = 0
        rr_a, ri_a = pde_residual_subdomain(model.net_air,  z_air,  bm.k0, bm.n_air**2)
        rr_s, ri_s = pde_residual_subdomain(model.net_slab, z_slab, bm.k0, bm.n_slab**2)
        rr_u, ri_u = pde_residual_subdomain(model.net_sub,  z_sub,  bm.k0, bm.n_sub**2)
        L_pde_air  = torch.mean(rr_a**2 + ri_a**2)
        L_pde_slab = torch.mean(rr_s**2 + ri_s**2)
        L_pde_sub  = torch.mean(rr_u**2 + ri_u**2)
        L_pde = L_pde_air + L_pde_slab + L_pde_sub

        # Interface losses
        L_f1 = interface_field_loss(model.net_air,  model.net_slab, bm.z_slab_top, n_if, dtype, device)
        L_f2 = interface_field_loss(model.net_slab, model.net_sub,  bm.z_slab_bot, n_if, dtype, device)
        L_x1 = interface_flux_loss( model.net_air,  model.net_slab, bm.z_slab_top, bm.k0, n_if, dtype, device)
        L_x2 = interface_flux_loss( model.net_slab, model.net_sub,  bm.z_slab_bot, bm.k0, n_if, dtype, device)

        # Boundary losses
        L_top = top_bc_loss_dd(model.net_air, 0.0,               r,      n_bc, dtype, device)
        L_bot = bottom_bc_loss_dd(model.net_sub, bm.domain_height, bm.k0, bm.n_sub, n_bc, dtype, device)

        loss = (w_pde  * L_pde
                + w_field * (L_f1 + L_f2)
                + w_flux  * (L_x1 + L_x2)
                + w_top   * L_top
                + w_bot   * L_bot)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % 500 == 0 or ep == 1:
            history.append({
                "epoch":     ep,
                "pde_air":   float(L_pde_air.detach()),
                "pde_slab":  float(L_pde_slab.detach()),
                "pde_sub":   float(L_pde_sub.detach()),
                "field_int": float((L_f1 + L_f2).detach()),
                "flux_int":  float((L_x1 + L_x2).detach()),
                "top_bc":    float(L_top.detach()),
                "bot_bc":    float(L_bot.detach()),
                "total":     float(loss.detach()),
            })

    # LBFGS refinement
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=10,
        history_size=50, line_search_fn="strong_wolfe",
    )

    def _closure():
        lbfgs.zero_grad(set_to_none=True)
        ra, ia = pde_residual_subdomain(model.net_air,  z_air,  bm.k0, bm.n_air**2)
        rs, is_ = pde_residual_subdomain(model.net_slab, z_slab, bm.k0, bm.n_slab**2)
        ru, iu = pde_residual_subdomain(model.net_sub,  z_sub,  bm.k0, bm.n_sub**2)
        lp  = torch.mean(ra**2+ia**2) + torch.mean(rs**2+is_**2) + torch.mean(ru**2+iu**2)
        lf1 = interface_field_loss(model.net_air,  model.net_slab, bm.z_slab_top, n_if, dtype, device)
        lf2 = interface_field_loss(model.net_slab, model.net_sub,  bm.z_slab_bot, n_if, dtype, device)
        lx1 = interface_flux_loss( model.net_air,  model.net_slab, bm.z_slab_top, bm.k0, n_if, dtype, device)
        lx2 = interface_flux_loss( model.net_slab, model.net_sub,  bm.z_slab_bot, bm.k0, n_if, dtype, device)
        lt  = top_bc_loss_dd(model.net_air, 0.0, r, n_bc, dtype, device)
        lb  = bottom_bc_loss_dd(model.net_sub, bm.domain_height, bm.k0, bm.n_sub, n_bc, dtype, device)
        l   = w_pde*lp + w_field*(lf1+lf2) + w_flux*(lx1+lx2) + w_top*lt + w_bot*lb
        l.backward()
        return l

    for _ in range(200):
        lbfgs.step(_closure)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_dd(
    model: DomainDecompLayered,
    bm: LayeredMediumBenchmark,
    output_dir: Path,
    device_str: str = "cpu",
) -> dict:
    device = resolve_device(device_str)
    dtype  = resolve_training_dtype("float64", device)

    Nz   = 512
    z_np = np.linspace(0.0, bm.domain_height, Nz)
    z_t  = torch.as_tensor(z_np, dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        out = model.forward(z_t)
    pinn_r = detach_numpy(out[:, 0])
    pinn_i = detach_numpy(out[:, 1])

    ref_r, ref_i = bm.analytical_field_np(np.zeros(Nz), z_np)

    # Global errors
    errors = evaluate_benchmark_errors(pinn_r, pinn_i, ref_r, ref_i)

    # Per-region errors
    region_errors = evaluate_benchmark_errors_by_region(
        pinn_r, pinn_i, ref_r, ref_i, z_np, bm
    )

    # Final subdomain PDE + interface losses
    model.train()
    margin = 1e-4
    z_a = _sample_interior(0.0,           bm.z_slab_top,    256, margin, 99, device, dtype)
    z_s = _sample_interior(bm.z_slab_top, bm.z_slab_bot,    256, margin, 98, device, dtype)
    z_u = _sample_interior(bm.z_slab_bot, bm.domain_height, 256, margin, 97, device, dtype)
    r   = bm.reflection_coefficient()
    with torch.enable_grad():
        ra,ia = pde_residual_subdomain(model.net_air,  z_a, bm.k0, bm.n_air**2)
        rs,is_= pde_residual_subdomain(model.net_slab, z_s, bm.k0, bm.n_slab**2)
        ru,iu = pde_residual_subdomain(model.net_sub,  z_u, bm.k0, bm.n_sub**2)
        pde_air  = float(torch.mean(ra**2+ia**2).detach())
        pde_slab = float(torch.mean(rs**2+is_**2).detach())
        pde_sub  = float(torch.mean(ru**2+iu**2).detach())
        Lf1 = float(interface_field_loss(model.net_air,  model.net_slab, bm.z_slab_top, 128, dtype, device).detach())
        Lx1 = float(interface_flux_loss( model.net_air,  model.net_slab, bm.z_slab_top, bm.k0, 128, dtype, device).detach())
        Lf2 = float(interface_field_loss(model.net_slab, model.net_sub,  bm.z_slab_bot, 128, dtype, device).detach())
        Lx2 = float(interface_flux_loss( model.net_slab, model.net_sub,  bm.z_slab_bot, bm.k0, 128, dtype, device).detach())
        Lt  = float(top_bc_loss_dd(model.net_air, 0.0, r, 128, dtype, device).detach())
        Lb  = float(bottom_bc_loss_dd(model.net_sub, bm.domain_height, bm.k0, bm.n_sub, 128, dtype, device).detach())
    model.eval()

    # Figures
    output_dir.mkdir(parents=True, exist_ok=True)
    errs_label = (
        f"L2_mag={errors['relative_l2_magnitude']:.3e}  "
        f"L2_complex={errors['relative_complex_l2']:.3e}  "
        f"L2_aligned={errors['relative_l2_phase_aligned']:.3e}  "
        f"phase_rmse={errors['phase_rmse_deg']:.1f}°"
    )
    plot_complex_field_comparison(
        z_np, pinn_r, pinn_i, ref_r, ref_i, bm,
        str(output_dir / "bm2_layered_dd_field.png"),
        title=f"DD-PINN layered (1-D)\n{errs_label}",
    )

    return {
        "errors":         errors,
        "region_errors":  region_errors,
        "pde_air_mse":    pde_air,
        "pde_slab_mse":   pde_slab,
        "pde_sub_mse":    pde_sub,
        "field_int1":     Lf1,
        "flux_int1":      Lx1,
        "field_int2":     Lf2,
        "flux_int2":      Lx2,
        "top_bc":         Lt,
        "bot_bc":         Lb,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="1-D layered-medium domain-decomposition PINN benchmark"
    )
    parser.add_argument("--epochs",        type=int,   default=5000)
    parser.add_argument("--hidden-layers", type=int,   default=4)
    parser.add_argument("--hidden-width",  type=int,   default=64)
    parser.add_argument("--fourier-levels",type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="cpu")
    parser.add_argument("--output-dir",    type=str,   default="outputs/benchmarks")
    parser.add_argument("--w-pde",   type=float, default=1.0)
    parser.add_argument("--w-field", type=float, default=500.0)
    parser.add_argument("--w-flux",  type=float, default=500.0)
    parser.add_argument("--w-top",   type=float, default=200.0)
    parser.add_argument("--w-bot",   type=float, default=100.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out

    bm = LayeredMediumBenchmark(k0=2.0 * np.pi)

    # Step 1: diagnostics
    print_coordinate_diagnostics(bm)
    intf_errs, pde_residuals = print_analytical_diagnostics(bm)

    # Step 2: train
    print(f"\n=== Training DD-PINN  epochs={args.epochs} ===")
    print(f"  Architecture: {args.hidden_layers}×{args.hidden_width}  Fourier levels={args.fourier_levels}")
    print(f"  Loss weights: pde={args.w_pde}  field={args.w_field}  flux={args.w_flux}  top={args.w_top}  bot={args.w_bot}")

    model, history = train_dd(
        bm,
        epochs=args.epochs,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        num_fourier_levels=args.fourier_levels,
        w_pde=args.w_pde, w_field=args.w_field, w_flux=args.w_flux,
        w_top=args.w_top, w_bot=args.w_bot,
        device_str=args.device,
    )

    print("\n  Training history:")
    for row in history:
        print(f"    ep={row['epoch']:5d}  "
              f"pde_air={row['pde_air']:.2e}  pde_slab={row['pde_slab']:.2e}  pde_sub={row['pde_sub']:.2e}  "
              f"field={row['field_int']:.2e}  flux={row['flux_int']:.2e}  "
              f"top={row['top_bc']:.2e}  bot={row['bot_bc']:.2e}")

    # Step 3: evaluate
    results = evaluate_dd(model, bm, out, device_str=args.device)

    # Step 4: report
    print("\n=== Final report ===")
    print("\n  Analytical interface errors:")
    for k, v in intf_errs.items():
        print(f"    {k}: {v:.3e}")

    print("\n  Subdomain PDE residuals (final):")
    print(f"    pde_air_mse:  {results['pde_air_mse']:.4e}")
    print(f"    pde_slab_mse: {results['pde_slab_mse']:.4e}")
    print(f"    pde_sub_mse:  {results['pde_sub_mse']:.4e}")

    print("\n  Interface losses (final):")
    print(f"    field_int1:  {results['field_int1']:.4e}  (E continuity at z_slab_top)")
    print(f"    flux_int1:   {results['flux_int1']:.4e}   (dE/dz continuity at z_slab_top)")
    print(f"    field_int2:  {results['field_int2']:.4e}  (E continuity at z_slab_bot)")
    print(f"    flux_int2:   {results['flux_int2']:.4e}   (dE/dz continuity at z_slab_bot)")

    print("\n  Boundary losses (final):")
    print(f"    top_bc: {results['top_bc']:.4e}")
    print(f"    bot_bc: {results['bot_bc']:.4e}")

    print("\n  Global accuracy:")
    e = results['errors']
    for k, v in e.items():
        print(f"    {k}: {v:.4e}")

    print("\n  Per-region accuracy:")
    for region, re in results['region_errors'].items():
        if re:
            print(f"    [{region}]  complex_L2={re.get('relative_complex_l2', float('nan')):.3e}  "
                  f"mag={re.get('relative_l2_magnitude', float('nan')):.3e}  "
                  f"phase_rmse={re.get('phase_rmse_deg', float('nan')):.1f}°")

    print(f"\n  Phase note: global_phase_offset = {e['global_phase_offset_deg']:.2f}°  "
          f"(non-zero => global phase shift, not field-shape error)")

    # Save JSON
    report = {
        "coordinate_convention": {
            "time": "exp(+iωt) suppressed",
            "propagation": "E_inc = exp(-i k0 z)",
            "problem_dimension": "1-D in z",
            "interface_flux": "dE/dz physical (autograd chain rule handles k_j scaling)",
        },
        "analytical_interface_errors": intf_errs,
        "analytical_pde_residuals":    pde_residuals,
        "results": results,
    }
    report_path = out / "bm2_dd_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report:  {report_path}")
    print(f"  Figures: {out / 'bm2_layered_dd_field.png'}")

    # Pass criteria (all must pass)
    complex_l2 = e["relative_complex_l2"]
    aligned_l2 = e["relative_l2_phase_aligned"]
    mag_l2     = e["relative_l2_magnitude"]
    passed = (complex_l2 < 0.10) and (aligned_l2 < 0.05) and (mag_l2 < 0.05)
    print(f"\n  Pass criteria:")
    print(f"    relative_complex_l2        < 0.10: {complex_l2:.3e}  {'PASS' if complex_l2 < 0.10 else 'FAIL'}")
    print(f"    relative_l2_phase_aligned  < 0.05: {aligned_l2:.3e}  {'PASS' if aligned_l2 < 0.05 else 'FAIL'}")
    print(f"    relative_l2_magnitude      < 0.05: {mag_l2:.3e}  {'PASS' if mag_l2 < 0.05 else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
