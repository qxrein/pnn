"""Staged 2-D Maxwell benchmark configurations.

Progression for validating the 2-D Maxwell PINN:

Case A — Homogeneous medium
    No grating.  εr = n_air² everywhere.
    Analytical: E = exp(-ik0 z), H̃_x = n_air * E.
    Success: complex_L2 < 0.01.

Case B — Horizontal layers (x-independent)
    Full-width 'ridge' = planar slab, identical to 1-D layered benchmark.
    Analytical: TMM solution.  Must match 1-D Maxwell DD result.
    Success: complex_L2 < 0.05 (same threshold as 1-D benchmark).

Case C — Shallow grating
    Reduced ridge height and index contrast.
    Smaller scattering.  Easier for the PINN than the full grating.
    Success: complex_L2 < 0.20.

Case D — Non-Rayleigh grating (period ≠ wavelength)
    Use period = 0.8 * wavelength.  Avoids the Rayleigh anomaly.
    The ±1 diffraction orders are propagating in this configuration.
    Success: complex_L2 < 0.30.

Case E — Full grating (original geometry)
    Success: complex_L2 < 0.30 (intermediate milestone).

For each case, the RCWA reference field is regenerated if `period` changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import PhysicsConfig


@dataclass
class BenchmarkCase:
    """Description of a single benchmark configuration."""

    name: str
    tag: str
    physics: PhysicsConfig
    description: str
    success_complex_l2: float
    reference_path: str | None = None  # path to RCWA NPZ if available


def make_homogeneous(base: PhysicsConfig) -> BenchmarkCase:
    """Case A: uniform air medium, no grating."""
    p = PhysicsConfig(
        wavelength=base.wavelength,
        n_air=base.n_air, n_ridge=base.n_air, n_substrate=base.n_air,
        period=base.period,
        ridge_width=0.0, ridge_height=0.0,
        domain_height=base.domain_height,
        ridge_base_fraction=1.0,
        interface_margin=base.interface_margin,
        nx_visualization=base.nx_visualization,
        nz_visualization=base.nz_visualization,
    )
    return BenchmarkCase(
        name="Homogeneous 2D",
        tag="2d_hom",
        physics=p,
        description="Uniform medium (n_air). Exact solution: exp(-ik0 z). No grating.",
        success_complex_l2=0.01,
    )


def make_horizontal_layers(base: PhysicsConfig) -> BenchmarkCase:
    """Case B: full-width slab, x-independent (matches 1-D layered benchmark)."""
    p = PhysicsConfig(
        wavelength=base.wavelength,
        n_air=base.n_air, n_ridge=base.n_ridge, n_substrate=base.n_substrate,
        period=base.period,
        ridge_width=base.period,   # full period width = planar slab
        ridge_height=base.ridge_height,
        domain_height=base.domain_height,
        ridge_base_fraction=base.ridge_base_fraction,
        interface_margin=base.interface_margin,
        nx_visualization=base.nx_visualization,
        nz_visualization=base.nz_visualization,
    )
    return BenchmarkCase(
        name="Horizontal layers (1D equiv)",
        tag="2d_lay",
        physics=p,
        description="Full-width slab = planar layer. Must match 1-D Maxwell DD benchmark.",
        success_complex_l2=0.05,
    )


def make_shallow_grating(base: PhysicsConfig) -> BenchmarkCase:
    """Case C: shallow grating with reduced height and contrast."""
    p = PhysicsConfig(
        wavelength=base.wavelength,
        n_air=base.n_air,
        n_ridge=base.n_air + 0.1 * (base.n_ridge - base.n_air),  # 10% of contrast
        n_substrate=base.n_substrate,
        period=base.period,
        ridge_width=base.ridge_width,
        ridge_height=0.05 * base.wavelength,   # very shallow ridge
        domain_height=base.domain_height,
        ridge_base_fraction=base.ridge_base_fraction,
        interface_margin=base.interface_margin / 2.0,
        nx_visualization=base.nx_visualization,
        nz_visualization=base.nz_visualization,
    )
    return BenchmarkCase(
        name="Shallow grating",
        tag="2d_shallow",
        physics=p,
        description=(
            f"Shallow ridge h=0.05λ, low contrast "
            f"n_ridge={p.n_ridge:.3f}. Weak scattering."
        ),
        success_complex_l2=0.20,
    )


def make_non_rayleigh_grating(base: PhysicsConfig) -> BenchmarkCase:
    """Case D: period = 0.8*wavelength, avoids Rayleigh anomaly.

    At period=0.8λ the ±1 diffraction orders are propagating in air:
        kx_±1 = ±2π/period = ±2π/(0.8λ) = ±2.5π/λ
        kz_±1 = sqrt((k0)² - kx²) = sqrt((2π/λ)² - (2.5π/λ)²)
              = (π/λ) sqrt(4 - 6.25) < 0  → NOT propagating in air (still evanescent)

    Wait: period=0.8λ:  kx_1 = 2π/period = 2π/(0.8λ) = 2.5*(2π/λ) = 2.5*k0
    kz² = k0² - kx_1² = k0² - 6.25*k0² = -5.25*k0² < 0  → evanescent!

    Actually for ±1 to propagate: kx_1 < k0 → 2π/period < 2π/λ → period > λ.
    For period = 1.5*λ the ±1 orders are propagating.
    Use period = 1.5*λ as the non-Rayleigh (multi-order) test.
    """
    new_period = 1.5 * base.wavelength
    p = PhysicsConfig(
        wavelength=base.wavelength,
        n_air=base.n_air, n_ridge=base.n_ridge, n_substrate=base.n_substrate,
        period=new_period,
        ridge_width=0.4 * new_period,     # same fill factor
        ridge_height=base.ridge_height,
        domain_height=base.domain_height,
        ridge_base_fraction=base.ridge_base_fraction,
        interface_margin=base.interface_margin,
        nx_visualization=base.nx_visualization,
        nz_visualization=base.nz_visualization,
    )
    # Verify ±1 orders propagate
    k0 = 2.0 * math.pi / base.wavelength
    kx1 = 2.0 * math.pi / new_period
    kz1_sq = k0**2 - kx1**2
    assert kz1_sq > 0, f"±1 order still evanescent at period={new_period:.3f}λ, kz²={kz1_sq:.4f}"

    return BenchmarkCase(
        name=f"Non-Rayleigh grating (period={new_period:.2f}λ)",
        tag="2d_nr",
        physics=p,
        description=(
            f"Period={new_period:.3f}λ = 1.5λ. "
            f"±1 diffraction orders propagating (kz_1 = {math.sqrt(kz1_sq):.4f}). "
            "No Rayleigh anomaly."
        ),
        success_complex_l2=0.30,
    )


def make_full_grating(base: PhysicsConfig) -> BenchmarkCase:
    """Case E: original full grating (period = wavelength, Rayleigh geometry)."""
    return BenchmarkCase(
        name="Full grating (original)",
        tag="2d_grating",
        physics=base,
        description=(
            f"Period=λ. ±1 orders evanescent. "
            f"n_ridge={base.n_ridge}, ridge_height={base.ridge_height}λ."
        ),
        success_complex_l2=0.30,
        reference_path="outputs/reference_grating.npz",
    )


def print_diffraction_order_analysis(physics: PhysicsConfig) -> dict:
    """Print which diffraction orders are propagating for given geometry."""
    k0 = physics.k0
    n_air = physics.n_air
    n_sub = physics.n_substrate
    period = physics.period
    wavelength = physics.wavelength

    print(f"\n  Diffraction order analysis  (period={period:.4f}λ,  k0={k0:.4f})")
    print(f"  {'Order':>6}  {'kx':>10}  {'kz_air':>14}  {'kz_sub':>14}  {'air':>12}  {'sub':>12}")
    orders_info = {}
    for m in range(-4, 5):
        kx = m * 2.0 * math.pi / period
        kz2_air = (k0 * n_air)**2 - kx**2
        kz2_sub = (k0 * n_sub)**2 - kx**2
        status_air = "propagating" if kz2_air > 0 else f"evanescent (β={math.sqrt(abs(kz2_air)):.3f})"
        status_sub = "propagating" if kz2_sub > 0 else f"evanescent"
        kz_air_str = f"{math.sqrt(max(0,kz2_air)):.4f}" if kz2_air > 0 else f"i·{math.sqrt(abs(kz2_air)):.4f}"
        kz_sub_str = f"{math.sqrt(max(0,kz2_sub)):.4f}" if kz2_sub > 0 else f"i·{math.sqrt(abs(kz2_sub)):.4f}"
        print(f"  {m:>6}  {kx:>10.4f}  {kz_air_str:>14}  {kz_sub_str:>14}  {status_air:>12}  {status_sub:>12}")
        orders_info[m] = {
            "kx": kx,
            "kz2_air": kz2_air,
            "kz2_sub": kz2_sub,
            "prop_air": kz2_air > 0,
            "prop_sub": kz2_sub > 0,
        }
    n_prop_air = sum(1 for v in orders_info.values() if v["prop_air"])
    print(f"\n  Propagating orders in air: {n_prop_air}")
    print(f"  Rayleigh anomaly at period=λ: ±1 orders have kz=0")
    return orders_info
