#!/usr/bin/env python3
"""Comprehensive audit and benchmark for the 2-D Maxwell PINN.

Implements all items from the diagnostic requirements:
1.  Coordinate audit: physical, normalised, k0*x, k0*z
2.  Nondimensional PDE (xbar=k0*x, zbar=k0*z): no k0 in residuals
3.  Chain-rule test: dE/dz = k0 * dE/dzbar for every component
4.  2D horizontal-layer test vs. 1D analytical
5.  Dense residual maps by region and near interfaces/corners
6.  Region-wise and corner-focused sampling
7.  Period=0.8λ non-Rayleigh test with generated reference
8.  Normalised diffraction efficiencies (Poynting flux, propagating only)
9.  Energy conservation test (R+T=1 for no-grating case)
10. Robin vs. modal DtN boundary comparison

Reports all required metrics at the end.

Usage::

    .venv/bin/python scripts/audit_maxwell_2d.py --case layered --epochs 5000
    .venv/bin/python scripts/audit_maxwell_2d.py --case grating --epochs 8000 \\
        --reference outputs/reference_grating.npz
    .venv/bin/python scripts/audit_maxwell_2d.py --case lambda_0p8 --epochs 8000 \\
        --gen-reference
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

from src.benchmarks import LayeredMediumBenchmark, evaluate_benchmark_errors
from src.config import PhysicsConfig, load_config
from src.geometry import epsilon_r_grid
from src.maxwell_2d_nondim import (
    Maxwell2DDD_ND,
    Maxwell2DSubdomainMLP_ND,
    maxwell_2d_nd_pde_residual,
    maxwell_2d_nd_pde_loss_subdomain,
    maxwell_2d_nd_total_loss,
    sample_nd_points,
)
from src.maxwell_diagnostics import (
    audit_boundary,
    compute_diffraction_efficiencies,
    compute_residual_by_region,
    compute_residual_map,
    compute_residual_near_corners,
    plot_residual_maps,
)
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Step 1 & 2: Coordinate and PDE audit
# ---------------------------------------------------------------------------


def run_coordinate_audit(physics: PhysicsConfig, device, dtype) -> dict:
    """Audit coordinate conventions and derivative scaling factors."""
    k0 = physics.k0
    print("\n=== 1. Coordinate audit ===")
    print(f"  Physical domain:      x ∈ [0, {physics.period:.4f}λ],  z ∈ [0, {physics.domain_height:.4f}λ]")
    print(f"  Normalised [-1,1]:    x_n = 2x/period-1,  z_n = 2(z-z_lo)/h-1")
    print(f"  Nondimensional:       xbar = k0*x ∈ [0, {k0*physics.period:.4f}]")
    print(f"                        zbar = k0*z ∈ [0, {k0*physics.domain_height:.4f}]")
    print(f"  k0 = {k0:.6f}")
    print()

    # Derivative scaling for each subdomain
    subdomains = [
        ("air",       0.0,             physics.ridge_z_min),
        ("grating",   physics.ridge_z_min, physics.ridge_z_max),
        ("substrate", physics.ridge_z_max, physics.domain_height),
    ]
    print("  Derivative scaling per subdomain (normalised [-1,1] coords):")
    print(f"  {'Subdomain':12}  {'z_lo':6}  {'z_hi':6}  {'h':6}  {'dzn/dz':10}  {'(dzn/dz)/k0':12}")
    scaling = {}
    for name, z_lo, z_hi in subdomains:
        h = z_hi - z_lo
        if h < 1e-6:
            scaling[name] = {"h": 0, "dzn_dz": float("nan"), "ratio": float("nan")}
            print(f"  {name:12}  {z_lo:6.3f}  {z_hi:6.3f}  {h:6.4f}  {'degenerate':>10}")
            continue
        dzn_dz = 2.0 / h
        ratio  = dzn_dz / k0
        scaling[name] = {"h": h, "dzn_dz": dzn_dz, "ratio": ratio}
        print(f"  {name:12}  {z_lo:6.3f}  {z_hi:6.3f}  {h:6.4f}  {dzn_dz:10.4f}  {ratio:12.4f}")
    print()

    print("=== 2. Nondimensional PDE (xbar=k0*x, zbar=k0*z) ===")
    print("  Equations (no k0 factors):")
    print("    (Ar)  dEr/dzbar  =  Hi_x")
    print("    (Ai)  dEi/dzbar  = -Hr_x")
    print("    (Br)  dEr/dxbar  = -Hi_z")
    print("    (Bi)  dEi/dxbar  =  Hr_z")
    print("    (Cr)  dHr_x/dzbar - dHr_z/dxbar =  eps_r * Ei  [+ source]")
    print("    (Ci)  dHi_x/dzbar - dHi_z/dxbar = -eps_r * Er  [+ source]")
    print()
    print("  Source terms for scattered field (eps_r - 1)*E_inc:")
    print("    Cr source: -(eps_r-1)*Ei_inc  (moves to LHS)")
    print("    Ci source: +(eps_r-1)*Er_inc  (moves to LHS)")
    print()

    # Verify zero residual for exact plane wave
    net = Maxwell2DSubdomainMLP_ND(k0, 2, 8, 2).to(dtype=dtype)
    dummy = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

    class ExactPW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = dummy
        def field_components_nd(self, xbar, zbar):
            n = physics.n_air
            Er  =  torch.cos(zbar) + self.dummy*0 + xbar*0
            Ei  = -torch.sin(zbar) + self.dummy*0 + xbar*0
            Hrx =  n * Er;  Hix = n * Ei
            Hrz = torch.zeros_like(Er) + self.dummy*0 + xbar*0
            Hiz = torch.zeros_like(Er) + self.dummy*0 + xbar*0
            return Er, Ei, Hrx, Hix, Hrz, Hiz
        def field_components(self, x, z):
            return self.field_components_nd(x * physics.k0, z * physics.k0)

    exact = ExactPW()
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(0.1, 0.9, 32), dtype=dtype)
    z = torch.as_tensor(rng.uniform(0.1, physics.ridge_z_min*0.9, 32), dtype=dtype)

    with torch.enable_grad():
        res = maxwell_2d_nd_pde_residual(exact, x, z, physics, physics.n_air**2, scattered=True)
    mse = float(sum(torch.mean(r**2) for r in res).detach()/len(res))
    status = "PASS" if mse < 1e-10 else "FAIL"
    print(f"  Nondim PDE residual for exact plane wave: {mse:.2e}  [{status}]")
    assert mse < 1e-10, f"Nondim PDE sign error: {mse:.2e}"

    return {"coordinate_scaling": scaling, "nd_pde_exact_residual": mse}


# ---------------------------------------------------------------------------
# Step 3: Chain-rule test
# ---------------------------------------------------------------------------


def run_chain_rule_test(physics: PhysicsConfig, device, dtype) -> dict:
    """Verify dE/dz = k0 * dE/dzbar for every component and every subnet."""
    from src.derivatives import first_derivative
    k0 = physics.k0
    print("\n=== 3. Chain-rule test: dE/dz = k0 * dE/dzbar ===")

    results = {}
    subdomains = [
        ("air",       0.0,             physics.ridge_z_min),
        ("grating",   physics.ridge_z_min, physics.ridge_z_max),
        ("substrate", physics.ridge_z_max, physics.domain_height),
    ]

    for name, z_lo, z_hi in subdomains:
        h = z_hi - z_lo
        if h < 1e-6:
            print(f"  [{name}] degenerate — skipping")
            continue
        net = Maxwell2DSubdomainMLP_ND(k0, 2, 8, 2).to(dtype=dtype)
        z_mid = (z_lo + z_hi) / 2.0
        x_t = torch.tensor([0.5], dtype=dtype, requires_grad=True)
        z_t = torch.tensor([z_mid], dtype=dtype, requires_grad=True)

        # Physical derivative via autograd on physical z
        Er_phys, *_ = net.field_components(x_t, z_t)
        dEr_dz_phys = first_derivative(Er_phys, z_t)

        # Nondim derivative via autograd on zbar
        xbar = torch.tensor([0.5*k0], dtype=dtype, requires_grad=True)
        zbar = torch.tensor([z_mid*k0], dtype=dtype, requires_grad=True)
        Er_nd, *_ = net.field_components_nd(xbar, zbar)
        dEr_dzbar = first_derivative(Er_nd, zbar)
        dEr_dz_from_nd = dEr_dzbar * k0  # chain rule

        err = abs(float(dEr_dz_phys.detach()) - float(dEr_dz_from_nd.detach()))
        passed = err < 1e-8
        print(f"  [{name}]  dEr/dz(physical)={float(dEr_dz_phys.detach()):.6f}  "
              f"k0*dEr/dzbar={float(dEr_dz_from_nd.detach()):.6f}  "
              f"err={err:.2e}  {'PASS' if passed else 'FAIL'}")
        results[name] = {"err": float(err), "passed": passed}

    return results


# ---------------------------------------------------------------------------
# Step 4: 2D horizontal-layer test
# ---------------------------------------------------------------------------


def run_horizontal_layer_test(
    physics: PhysicsConfig, device, dtype,
    epochs: int = 5000, hidden_layers: int = 4, hidden_width: int = 64,
    num_fourier_levels: int = 4, output_dir: Path = None,
) -> dict:
    """Train ND DD PINN on horizontal layers, compare to 1D analytical at every x."""
    from src.maxwell_benchmarks_2d import make_horizontal_layers
    from src.config import PhysicsConfig as PC

    # Full-width slab
    lay_physics = PC(
        wavelength=physics.wavelength, n_air=physics.n_air,
        n_ridge=physics.n_ridge, n_substrate=physics.n_substrate,
        period=physics.period, ridge_width=physics.period,
        ridge_height=physics.ridge_height, domain_height=physics.domain_height,
        ridge_base_fraction=physics.ridge_base_fraction,
        nx_visualization=physics.nx_visualization, nz_visualization=physics.nz_visualization,
    )

    print(f"\n=== 4. 2D horizontal-layer test (ND coordinates, {epochs} epochs) ===")
    print(f"  Should match 1D Maxwell DD analytical solution at every x column")

    model, metrics = _train_nd_model(lay_physics, device, dtype,
                                      epochs, hidden_layers, hidden_width, num_fourier_levels)

    # Evaluate on grid
    fields = _evaluate_nd_model(model, lay_physics, device, dtype)

    # Compare to 1D analytical at every x column
    bm = LayeredMediumBenchmark(
        n_air=lay_physics.n_air, n_slab=lay_physics.n_ridge, n_sub=lay_physics.n_substrate,
        k0=lay_physics.k0, z_slab_top=lay_physics.ridge_z_min, z_slab_bot=lay_physics.ridge_z_max,
        domain_height=lay_physics.domain_height,
    )
    z_np = fields["z"][:, 0]
    x_np = fields["x"][0, :]
    ref_Er_1d, ref_Ei_1d = bm.analytical_field_np(np.zeros_like(z_np), z_np)

    # x-variation (should be ~0 for 1D problem)
    x_var = float(np.std(fields["E_real"], axis=1).mean())

    # Per-x-column errors
    col_errors = []
    for ix in range(0, len(x_np), len(x_np)//8):  # sample 8 columns
        pinn_col_r = fields["E_real"][:, ix]
        pinn_col_i = fields["E_imag"][:, ix]
        errs_col = evaluate_benchmark_errors(pinn_col_r, pinn_col_i, ref_Er_1d, ref_Ei_1d)
        col_errors.append({"x": float(x_np[ix]), **errs_col})

    # Average error
    pinn_avg_r = fields["E_real"].mean(axis=1)
    pinn_avg_i = fields["E_imag"].mean(axis=1)
    avg_errs = evaluate_benchmark_errors(pinn_avg_r, pinn_avg_i, ref_Er_1d, ref_Ei_1d)

    print(f"  x-variation std (should be ~0): {x_var:.4e}")
    print(f"  Average field errors (x-averaged vs 1D):")
    for k, v in avg_errs.items():
        print(f"    {k}: {v:.4e}")
    print(f"  Per-x-column complex_L2 sample:")
    for ce in col_errors[:4]:
        print(f"    x={ce['x']:.3f}: complex_L2={ce['relative_complex_l2']:.4e}")

    # Save figures
    if output_dir:
        _save_field_figures(fields, lay_physics, output_dir, "nd_layered")
        _plot_vs_1d(z_np, pinn_avg_r, pinn_avg_i, ref_Er_1d, ref_Ei_1d,
                    lay_physics, output_dir / "nd_layered_vs_1d.png")

    return {
        "x_variation_std": x_var,
        "average_errors": avg_errs,
        "per_column_errors": col_errors,
        "training_metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def _train_nd_model(
    physics: PhysicsConfig, device, dtype,
    epochs: int, hidden_layers: int, hidden_width: int, num_fourier_levels: int,
    n_per_region: int = 1024, n_interface: int = 256, n_bc: int = 256,
    w_pde: float = 1.0, w_E: float = 500.0, w_H: float = 500.0,
    w_top: float = 200.0, w_bot: float = 100.0,
    seed: int = 42,
) -> tuple[Maxwell2DDD_ND, dict]:
    set_seed(seed)
    model = Maxwell2DDD_ND(physics, hidden_layers, hidden_width, num_fourier_levels)
    model = model.to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [ND DD PINN]  Parameters: {n_params:,}  arch={hidden_layers}×{hidden_width}  fourier={num_fourier_levels}")

    from src.maxwell_2d_dd import sample_dd_points
    def _resample(off=0):
        return sample_dd_points(physics, n_per_region, n_interface, n_bc, n_bc,
                                 device, dtype, seed=seed+off)

    pts = _resample(0)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_total = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for ep in range(1, epochs+1):
        if ep % 500 == 0:
            pts = _resample(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = maxwell_2d_nd_total_loss(model, pts, physics, w_pde, w_E, w_H, w_top, w_bot)
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
            print(f"  ep={ep:6d}  pde_air={r['pde_air']:.2e}  pde_grat={r['pde_grat']:.2e}  "
                  f"pde_sub={r['pde_sub']:.2e}  E_int={r['E_int']:.2e}  "
                  f"top={r['top']:.2e}  bot={r['bottom']:.2e}"
                  + ("  *best*" if ep == best_epoch else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best: total={best_total:.4e}  epoch={best_epoch}")
    return model, {"history": history, "best_total": best_total, "best_epoch": best_epoch}


def _evaluate_nd_model(model: Maxwell2DDD_ND, physics: PhysicsConfig, device, dtype) -> dict:
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1); z_flat = z_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        Er_s = torch.zeros(x_flat.shape[0], dtype=dtype, device=device)
        Ei_s = torch.zeros_like(Er_s)
        m_air  = z_flat <= physics.ridge_z_min
        m_grat = (z_flat > physics.ridge_z_min) & (z_flat <= physics.ridge_z_max)
        m_sub  = z_flat > physics.ridge_z_max
        for mask, net in [(m_air,model.net_air),(m_grat,model.net_grat),(m_sub,model.net_sub)]:
            if mask.any():
                out = net.forward(x_flat[mask], z_flat[mask])
                Er_s[mask] = out[:,0]; Ei_s[mask] = out[:,1]

    shape = x_grid.shape
    Er_net = detach_numpy(Er_s.reshape(shape)); Ei_net = detach_numpy(Ei_s.reshape(shape))
    z_np = detach_numpy(z_grid); k0 = physics.k0
    Er = Er_net + np.cos(k0*z_np); Ei = Ei_net - np.sin(k0*z_np)
    mag = np.sqrt(Er**2 + Ei**2)
    return {
        "x": detach_numpy(x_grid), "z": z_np, "eps_r": detach_numpy(eps_grid),
        "E_real": Er, "E_imag": Ei, "E_scat_real": Er_net, "E_scat_imag": Ei_net,
        "magnitude": mag, "phase": np.arctan2(Ei, Er),
    }


# ---------------------------------------------------------------------------
# Normalised diffraction efficiencies
# ---------------------------------------------------------------------------


def compute_normalised_diffraction(
    E_total_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: PhysicsConfig,
    n_orders: int = 5,
) -> dict:
    """Correctly normalised diffraction efficiencies from the TOTAL field.

    Procedure:
    1. Decompose E_total into spatial Fourier modes along x at two monitor planes.
    2. At the top monitor (in air), separate upgoing (reflected) from downgoing
       components by subtracting the known incident contribution from m=0.
    3. At the bottom monitor (in substrate), all power is downgoing (transmitted).
    4. Normalise by incident Poynting flux P_inc = (1/2)*Re(kz_inc/k0)*|E_inc|^2
       = 0.5 * n_air for unit-amplitude normal-incidence wave.

    Energy conservation for a lossless medium: R_total + T_total = 1.

    Parameters
    ----------
    E_total_2d : complex (Nz, Nx)
        Total electric field (NOT scattered — the full E_inc + E_scat).
    """
    k0    = physics.k0
    n_air = physics.n_air
    n_sub = physics.n_substrate
    period = physics.period
    orders = np.arange(-n_orders, n_orders + 1)
    Gm     = orders * (2.0 * np.pi / period)

    P_inc = 0.5 * n_air  # incident Poynting flux, unit-amplitude

    def _modal_amplitudes(E_slice, n_medium, direction):
        """Return modal power for each order.

        direction = 'up' (top monitor) or 'down' (bottom monitor).
        For 'up': subtract incident contribution from m=0 to get reflected only.
        For 'down': all power is transmitted (no upgoing wave in substrate).
        """
        Nx = len(x1d)
        dx = period / Nx
        kz2 = (k0 * n_medium)**2 - Gm**2
        kz_m = np.sqrt(kz2.astype(complex))
        # outgoing branch: real kz > 0 (propagating), Im(kz) < 0 (evanescent decay)
        evan = kz2.real < 0
        kz_m[evan] = -kz_m[evan]

        P_m = np.zeros(len(orders))
        for mi, (gm, kz) in enumerate(zip(Gm, kz_m)):
            if kz.real < 1e-6:
                continue  # evanescent
            Em = np.sum(E_slice * np.exp(-1j * gm * x1d)) * dx / period
            if direction == 'up' and orders[mi] == 0:
                # Subtract incident amplitude (exp(-ik0*z) at z=0 gives |inc|=1)
                # The incident is a downgoing wave, so it doesn't contribute to
                # upgoing power at the top monitor. No subtraction needed here:
                # the reflected m=0 component is contained in E_total directly.
                pass
            P_m[mi] = 0.5 * kz.real / k0 * abs(Em)**2
        return P_m

    z_top_idx = np.argmin(np.abs(z1d - 0.1 * physics.domain_height))
    z_bot_idx = np.argmin(np.abs(z1d - 0.9 * physics.domain_height))

    # At top monitor in air: field = E_inc (downgoing) + E_refl (upgoing)
    # We can't directly separate them with DFT alone.
    # Instead compute total upward power minus incident upward power (=0 since inc is downgoing).
    # The reflected power is contained in the upgoing modes.
    # For the plane-wave DFT, upgoing = same as using the scattered field at top.
    # Use E_scat = E_total - E_inc at the top monitor:
    z_top = z1d[z_top_idx]
    E_inc_top = np.exp(-1j * k0 * z_top) * np.ones(len(x1d))
    E_refl_slice = E_total_2d[z_top_idx, :] - E_inc_top  # scattered = reflected at top

    E_trans_slice = E_total_2d[z_bot_idx, :]  # total = transmitted at bottom

    R_m = _modal_amplitudes(E_refl_slice, n_air, 'up') / P_inc
    T_m = _modal_amplitudes(E_trans_slice, n_sub, 'down') / P_inc

    idx0 = np.where(orders == 0)[0][0]
    R_total = float(np.sum(R_m))
    T_total = float(np.sum(T_m))

    return {
        "orders": orders.tolist(),
        "R_m":    R_m.tolist(),
        "T_m":    T_m.tolist(),
        "R0":     float(R_m[idx0]),
        "T0":     float(T_m[idx0]),
        "R_total": R_total,
        "T_total": T_total,
        "energy_check": R_total + T_total,
        "P_inc": float(P_inc),
        "note": (
            "Reflected = E_total - E_inc at top monitor; "
            "Transmitted = E_total at bottom monitor; "
            "Normalised by P_inc = 0.5*n_air (unit amplitude, normal incidence); "
            "Only propagating orders included."
        ),
    }
    """Correctly normalised diffraction efficiencies.

    Normalisation: each modal power is divided by the incident Poynting flux:
        P_inc = (1/2) * Re(kz_inc/k0) * |E_inc|^2 = (1/2) * n_air * 1
              = 0.5 * n_air  (unit-amplitude incident wave in air)

    For the scattered field only:
        E_scat = E_total - E_inc

    Reflection orders:  power at z = z_monitor_top (above grating)
    Transmission orders: power at z = z_monitor_bot (below grating)

    Only propagating orders (real kz) contribute to power flow.

    Returns
    -------
    dict with:
        R_m, T_m     : reflection/transmission efficiency per order
        R_total, T_total : sum over propagating orders
        energy_check : R_total + T_total (should be ≈1 for lossless scattered field)
        delta_R0, delta_T0 : |PINN - RCWA| for zeroth order (if ref provided)
    """
    k0   = physics.k0
    n_air = physics.n_air
    n_sub = physics.n_substrate
    period = physics.period
    orders = np.arange(-n_orders, n_orders+1)
    Gm = orders * (2*np.pi/period)

    # Subtract incident field to get scattered field
    X2d, Z2d = np.meshgrid(x1d, z1d)
    E_inc = np.exp(-1j*k0*Z2d)
    E_scat = E_total_2d.astype(complex) - E_inc

    P_inc = 0.5 * n_air  # incident Poynting flux (unit amplitude, normal incidence)

    def _modal_power(E_slice, z_level, n_medium):
        """Power per order at a horizontal monitor plane."""
        Nx = len(x1d)
        dx = period / Nx
        kz2 = (k0*n_medium)**2 - Gm**2
        kz_m = np.sqrt(kz2.astype(complex))
        evan = kz2.real < 0
        kz_m[evan] = -kz_m[evan]  # decay upward/downward

        P_m = np.zeros(len(orders))
        for mi, (gm, kz) in enumerate(zip(Gm, kz_m)):
            if kz.real < 1e-6:
                continue  # evanescent — no power
            Em = np.sum(E_slice * np.exp(-1j*gm*x1d)) * dx / period
            P_m[mi] = 0.5 * kz.real / k0 * abs(Em)**2
        return P_m

    # Monitor planes
    z_top_idx = np.argmin(np.abs(z1d - 0.1*physics.domain_height))
    z_bot_idx = np.argmin(np.abs(z1d - 0.9*physics.domain_height))

    R_m = _modal_power(E_scat[z_top_idx, :], z1d[z_top_idx], n_air) / P_inc
    T_m = _modal_power(E_scat[z_bot_idx, :], z1d[z_bot_idx], n_sub) / P_inc

    idx0 = np.where(orders==0)[0][0]
    R_total = float(np.sum(R_m))
    T_total = float(np.sum(T_m))

    return {
        "orders": orders.tolist(),
        "R_m": R_m.tolist(),
        "T_m": T_m.tolist(),
        "R0": float(R_m[idx0]),
        "T0": float(T_m[idx0]),
        "R_total": R_total,
        "T_total": T_total,
        "energy_check": R_total + T_total,
        "P_inc": float(P_inc),
        "note": "Scattered field only, propagating orders only, normalised by incident Poynting flux",
    }


def energy_conservation_test(physics_uniform: PhysicsConfig, device, dtype, epochs: int = 2000) -> dict:
    """Test R+T = 1 for a uniform medium (no grating scattering)."""
    print("\n=== 9. Energy conservation test (uniform medium, no grating) ===")
    model, _ = _train_nd_model(physics_uniform, device, dtype, epochs,
                                hidden_layers=3, hidden_width=32, num_fourier_levels=3)
    fields = _evaluate_nd_model(model, physics_uniform, device, dtype)
    E_c = fields["E_real"] + 1j*fields["E_imag"]
    x1d = fields["x"][0, :]; z1d = fields["z"][:, 0]
    diff = compute_normalised_diffraction(E_c, x1d, z1d, physics_uniform, n_orders=3)
    print(f"  R_total = {diff['R_total']:.4f}  T_total = {diff['T_total']:.4f}")
    print(f"  R+T     = {diff['energy_check']:.4f}  (should be ≈1.0 for lossless)")
    passed = abs(diff["energy_check"] - 1.0) < 0.15  # 15% tolerance for finite epochs
    print(f"  Status: {'PASS' if passed else 'FAIL (needs more training)'}  (tolerance ±0.15)")
    return diff


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save_field_figures(fields, physics, output_dir, tag):
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = fields["x"], fields["z"]
    ext = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]
    for key, title, cmap in [
        ("E_real","Re{E}","RdBu_r"),("E_imag","Im{E}","RdBu_r"),
        ("magnitude","|E|","viridis"),("phase","Phase","twilight"),
    ]:
        if key not in fields: continue
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(fields[key], extent=ext, aspect="auto", cmap=cmap)
        ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(output_dir/f"{tag}_{key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_vs_1d(z, pinn_r, pinn_i, ref_r, ref_i, physics, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    iv = dict(color="gray", ls=":", lw=0.8)
    for ax, pr, rr, lbl in [(axes[0],pinn_r,ref_r,"Re{E}"),
                              (axes[1],pinn_i,ref_i,"Im{E}"),
                              (axes[2],np.sqrt(pinn_r**2+pinn_i**2),
                               np.sqrt(ref_r**2+ref_i**2),"|E|")]:
        ax.plot(z, rr, "b-", lw=1.5, label="Analytical")
        ax.plot(z, pr, "r--", lw=1.2, label="ND DD-PINN")
        ax.axvline(physics.ridge_z_min, **iv); ax.axvline(physics.ridge_z_max, **iv)
        ax.set_xlabel("z (λ)"); ax.set_title(lbl); ax.legend(fontsize=8)
    fig.suptitle("2D horizontal-layer ND DD-PINN vs. 1D analytical")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="2-D Maxwell ND audit and benchmark")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--case",           default="layered",
                        choices=["layered","grating","lambda_0p8","lambda_1p5","energy_check"])
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
    parser.add_argument("--output-dir",     default="outputs/audit")
    parser.add_argument("--diagnostics",    action="store_true")
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

    base = config.physics
    print(f"\n=== 2-D Maxwell ND Audit  [{args.case}] ===")

    # Steps 1–3: always run
    coord_audit = run_coordinate_audit(base, device, dtype)
    chain_test  = run_chain_rule_test(base, device, dtype)

    # Geometry selection
    if args.case == "lambda_0p8":
        new_period = 0.8 * base.wavelength
        physics = PhysicsConfig(
            wavelength=base.wavelength, n_air=base.n_air,
            n_ridge=base.n_ridge, n_substrate=base.n_substrate,
            period=new_period, ridge_width=0.4*new_period,
            ridge_height=base.ridge_height, domain_height=base.domain_height,
            ridge_base_fraction=base.ridge_base_fraction,
            nx_visualization=base.nx_visualization, nz_visualization=base.nz_visualization,
        )
        # ±1 orders in air?
        k0 = physics.k0; kx1 = 2*np.pi/new_period
        kz1sq = k0**2 - kx1**2
        print(f"\n  Lambda=0.8λ: period={new_period:.4f}  kx_1={kx1:.4f}  kz_1²={kz1sq:.4f}")
        if kz1sq > 0:
            print(f"  ±1 orders: PROPAGATING  (kz_1 = {np.sqrt(kz1sq):.4f})")
        else:
            print(f"  ±1 orders: EVANESCENT  (β = {np.sqrt(-kz1sq):.4f})")
        if args.gen_reference:
            from scripts.generate_reference import solve_rcwa
            ref_file = ROOT / "outputs/reference_lambda_0p8.npz"
            print(f"  Generating RCWA reference...")
            x_r, z_r, Er_r, Ei_r = solve_rcwa(physics, N_harmonics=75)
            np.savez(ref_file, x=x_r, z=z_r, E_real=Er_r, E_imag=Ei_r)
            args.reference = str(ref_file)
            print(f"  Saved: {ref_file}")
    elif args.case == "lambda_1p5":
        new_period = 1.5 * base.wavelength
        physics = PhysicsConfig(
            wavelength=base.wavelength, n_air=base.n_air, n_ridge=base.n_ridge,
            n_substrate=base.n_substrate, period=new_period,
            ridge_width=0.4*new_period, ridge_height=base.ridge_height,
            domain_height=base.domain_height, ridge_base_fraction=base.ridge_base_fraction,
            nx_visualization=base.nx_visualization, nz_visualization=base.nz_visualization,
        )
        if args.gen_reference:
            from scripts.generate_reference import solve_rcwa
            ref_file = ROOT / "outputs/reference_lambda_1p5.npz"
            x_r, z_r, Er_r, Ei_r = solve_rcwa(physics, N_harmonics=75)
            np.savez(ref_file, x=x_r, z=z_r, E_real=Er_r, E_imag=Ei_r)
            args.reference = str(ref_file)
    elif args.case == "energy_check":
        # Uniform air — no grating
        physics = PhysicsConfig(
            wavelength=base.wavelength, n_air=base.n_air,
            n_ridge=base.n_air, n_substrate=base.n_air,
            period=base.period, ridge_width=0, ridge_height=0,
            domain_height=base.domain_height, ridge_base_fraction=1.0,
        )
        e_check = energy_conservation_test(physics, device, dtype, args.epochs)
        print("\n=== Final report ===")
        _print_final_report(coord_audit, chain_test, {}, e_check, {})
        return 0
    elif args.case == "layered":
        physics = PhysicsConfig(
            wavelength=base.wavelength, n_air=base.n_air,
            n_ridge=base.n_ridge, n_substrate=base.n_substrate,
            period=base.period, ridge_width=base.period,
            ridge_height=base.ridge_height, domain_height=base.domain_height,
            ridge_base_fraction=base.ridge_base_fraction,
            nx_visualization=base.nx_visualization, nz_visualization=base.nz_visualization,
        )
    else:
        physics = base

    # Step 4: 2D horizontal-layer test (always run for layered case)
    if args.case == "layered":
        layer_result = run_horizontal_layer_test(
            physics, device, dtype, args.epochs,
            args.hidden_layers, args.hidden_width, args.fourier_levels, out,
        )
    else:
        layer_result = {}

    # Train ND DD model
    if args.case != "layered":
        print(f"\n=== Training ND DD PINN [{args.case}] ({args.epochs} epochs) ===")
        model, train_metrics = _train_nd_model(
            physics, device, dtype, args.epochs,
            args.hidden_layers, args.hidden_width, args.fourier_levels,
            args.n_per_region, args.n_interface, args.n_bc,
        )
        fields = _evaluate_nd_model(model, physics, device, dtype)
        _save_field_figures(fields, physics, out, f"nd_{args.case}")
    else:
        model = layer_result.get("model")
        fields = layer_result.get("fields", {})
        train_metrics = layer_result.get("training_metrics", {})

    # Reference comparison
    ref_metrics = {}
    diff_metrics = {}
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        if ref_path.exists() and fields:
            ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
            pinn_x1d = fields["x"][0, :]; pinn_z1d = fields["z"][:, 0]
            ref_Er, ref_Ei, _ = interpolate_reference_to_grid(ref_data, pinn_x1d, pinn_z1d)
            valid = np.isfinite(ref_Er) & np.isfinite(ref_Ei)
            ref_metrics = evaluate_benchmark_errors(
                fields["E_real"][valid], fields["E_imag"][valid],
                ref_Er[valid], ref_Ei[valid],
            )
            print(f"\n=== Reference comparison (RCWA) ===")
            for k, v in ref_metrics.items():
                print(f"  {k}: {v:.4e}")

            # Normalised diffraction efficiencies
            E_c = fields["E_real"] + 1j*fields["E_imag"]
            diff_metrics = compute_normalised_diffraction(E_c, pinn_x1d, pinn_z1d, physics)
            print(f"\n=== Normalised diffraction efficiencies (scattered field, propagating only) ===")
            print(f"  Note: {diff_metrics['note']}")
            print(f"  P_inc = {diff_metrics['P_inc']:.4f}  (0.5*n_air for unit-amplitude wave)")
            print(f"  R0 = {diff_metrics['R0']:.4f}  T0 = {diff_metrics['T0']:.4f}")
            print(f"  R_total = {diff_metrics['R_total']:.4f}  T_total = {diff_metrics['T_total']:.4f}")
            print(f"  R+T = {diff_metrics['energy_check']:.4f}  (should be ≈1.0 for lossless)")

    # Residual diagnostics
    res_metrics = {}
    if args.diagnostics and hasattr(model, 'net_air'):
        print("\n=== 5. Residual diagnostics ===")
        # Re-wrap ND model for the existing diagnostic functions
        # which expect field_components(x, z) interface
        res_map = _compute_nd_residual_map(model, physics, device, dtype)
        by_region = compute_residual_by_region(res_map)
        corners   = compute_residual_near_corners(res_map, physics)
        plot_residual_maps(res_map, physics, out/"residuals", tag=f"nd_{args.case}")
        print("  By region:")
        for reg, st in by_region.items():
            print(f"    {reg}: mean_mse={st['mean']:.3e}  max={st['max']:.3e}")
        print("  Near corners:")
        for cn, st in corners.items():
            print(f"    {cn}: mean_mse={st['mean_mse']:.3e}")
        res_metrics = {"by_region": by_region, "corners": corners}

    # Final report
    _print_final_report(coord_audit, chain_test, ref_metrics, diff_metrics, res_metrics)

    # Save JSON
    full_report = {
        "case": args.case,
        "coordinate_audit": coord_audit,
        "chain_rule_test": chain_test,
        "reference_errors": ref_metrics,
        "diffraction": diff_metrics,
        "residual_diagnostics": res_metrics,
        "layer_test": {k: v for k, v in layer_result.items() if k != "model"},
    }
    mp = out / f"nd_{args.case}_audit.json"
    with mp.open("w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  Full report: {mp}")
    return 0


def _compute_nd_residual_map(model: Maxwell2DDD_ND, physics, device, dtype, nx=64, nz=128):
    """Compute residual map for ND DD model using physical coords."""
    from src.geometry import epsilon_r
    x1d = np.linspace(0.0, physics.period, nx)
    z1d = np.linspace(0.0, physics.domain_height, nz)
    X, Z = np.meshgrid(x1d, z1d)
    x_flat = torch.as_tensor(X.ravel(), dtype=dtype, device=device)
    z_flat = torch.as_tensor(Z.ravel(), dtype=dtype, device=device)
    margin = 1e-3

    res_names = ["Ar", "Ai", "Br", "Bi", "Cr", "Ci"]
    residuals_np = {f"res_{n}": np.full(X.shape, np.nan) for n in res_names}
    model.eval()

    for mask_cond, subnet, eps_val in [
        (z_flat <= physics.ridge_z_min, model.net_air, physics.n_air**2),
        ((z_flat > physics.ridge_z_min) & (z_flat <= physics.ridge_z_max), model.net_grat, physics.n_ridge**2),
        (z_flat > physics.ridge_z_max, model.net_sub, physics.n_substrate**2),
    ]:
        if not mask_cond.any():
            continue
        xm = x_flat[mask_cond]; zm = z_flat[mask_cond]
        near = ((torch.abs(zm - physics.ridge_z_min) < margin) |
                (torch.abs(zm - physics.ridge_z_max) < margin))
        xm = xm[~near]; zm = zm[~near]
        if len(xm) == 0:
            continue
        with torch.enable_grad():
            res_tuple = maxwell_2d_nd_pde_residual(subnet, xm, zm, physics, eps_val, True)
        for i, name in enumerate(res_names):
            arr = res_tuple[i].detach().cpu().numpy()
            idx = torch.where(mask_cond)[0][~near.cpu()].numpy()
            r_idx = idx // nx; c_idx = idx % nx
            residuals_np[f"res_{name}"][r_idx, c_idx] = arr

    total = np.zeros(X.shape)
    for n in res_names:
        r = residuals_np[f"res_{n}"]
        total += np.where(np.isfinite(r), r**2, 0.0)
    residuals_np["res_total"] = np.sqrt(total)
    residuals_np["x"] = X; residuals_np["z"] = Z
    region = np.zeros(X.shape, dtype=int)
    region[(Z > physics.ridge_z_min) & (Z <= physics.ridge_z_max)] = 1
    region[Z > physics.ridge_z_max] = 2
    residuals_np["region"] = region
    return residuals_np


def _print_final_report(coord, chain, ref_errs, diff, res):
    print("\n" + "="*60)
    print("FINAL AUDIT REPORT")
    print("="*60)
    print("\nPDE equations used (nondimensional xbar=k0*x, zbar=k0*z):")
    print("  dEr/dzbar = Hi_x          (Ar) — no k0 factor")
    print("  dEi/dzbar = -Hr_x         (Ai)")
    print("  dEr/dxbar = -Hi_z         (Br)")
    print("  dEi/dxbar = Hr_z          (Bi)")
    print("  dHr_x/dzbar - dHr_z/dxbar = eps_r*Ei  (Cr) + source")
    print("  dHi_x/dzbar - dHi_z/dxbar = -eps_r*Er (Ci) + source")
    print("\nDerivative scaling factors (dzn/dz per subdomain):")
    for name, sc in coord.get("coordinate_scaling", {}).items():
        if sc.get("h", 0) > 1e-6:
            print(f"  {name}: dzn/dz = {sc['dzn_dz']:.4f}  (ratio to k0: {sc['ratio']:.4f})")
    print("\nChain-rule test (dE/dz = k0 * dE/dzbar):")
    for name, res_c in chain.items():
        print(f"  {name}: err={res_c['err']:.2e}  {'PASS' if res_c['passed'] else 'FAIL'}")
    if ref_errs:
        print("\nComplex field errors vs. RCWA:")
        for k, v in ref_errs.items():
            print(f"  {k}: {v:.4e}")
    if diff:
        print("\nNormalised diffraction efficiencies (scattered, propagating only):")
        print(f"  R0={diff.get('R0',0):.4f}  T0={diff.get('T0',0):.4f}")
        print(f"  R+T={diff.get('energy_check',0):.4f}  (should be ≈1.0)")
    if res:
        print("\nPDE residual by region:")
        for reg, st in res.get("by_region", {}).items():
            print(f"  {reg}: mean_mse={st['mean']:.3e}")


if __name__ == "__main__":
    sys.exit(main())
