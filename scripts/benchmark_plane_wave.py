#!/usr/bin/env python3
"""Benchmark 1: Homogeneous plane-wave PINN.

Removes the grating (uniform medium, constant εr).
Trains a PINN to solve ∂²E/∂z² + k0² n² E = 0 with:
    E(z=0) = 1 + 0i   (incident wave)
    Robin outgoing BC at z=domain_height

Analytical solution: E(z) = exp(-i k0 n z).

This benchmark MUST pass before the full grating problem is considered valid.
Success criterion: relative L2 error < 1%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks import PlaneWaveBenchmark, evaluate_benchmark_errors
from src.config import (
    LossWeights,
    ModelConfig,
    PathsConfig,
    PhysicsConfig,
    PINNConfig,
    SamplingConfig,
    TrainingConfig,
)
from src.model import FieldMLP
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Homogeneous physics helpers
# ---------------------------------------------------------------------------


def helmholtz_residual_uniform(
    model: FieldMLP,
    x: torch.Tensor,
    z: torch.Tensor,
    n: float,
    k0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nondimensional Helmholtz residual for uniform medium.

    ∂²E/∂x̃² + ∂²E/∂z̃² + n² E = 0  (x̃ = k0 x, z̃ = k0 z)
    """
    from src.derivatives import field_laplacian, prepare_coords

    k0_val = k0
    x_t = (x * k0_val).detach().clone().requires_grad_(True)
    z_t = (z * k0_val).detach().clone().requires_grad_(True)
    e_real, e_imag = model.field_components_nd(x_t, z_t)
    _, _, _, _, lap_real, lap_imag = field_laplacian(e_real, e_imag, x_t, z_t)
    eps = n**2
    return lap_real + eps * e_real, lap_imag + eps * e_imag


def top_bc_loss(model, x, z, k0):
    """E(z=0) = 1 + 0i."""
    e = model(x, z)
    target = torch.zeros_like(e)
    target[:, 0] = 1.0   # E_real = 1 at z=0
    # E_imag = 0 at z=0
    return torch.mean((e - target) ** 2)


def robin_bc_loss(model, x, z, k_sub):
    """Robin outgoing BC: ∂E/∂z + i k_sub E = 0."""
    from src.derivatives import first_derivative
    xr = x.detach().clone().requires_grad_(True)
    zr = z.detach().clone().requires_grad_(True)
    e_real, e_imag = model.field_components(xr, zr)
    from src.derivatives import first_derivative
    dEr_dz = first_derivative(e_real, zr)
    dEi_dz = first_derivative(e_imag, zr)
    res_r = dEr_dz - k_sub * e_imag
    res_i = dEi_dz + k_sub * e_real
    return torch.mean(res_r**2 + res_i**2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_plane_wave(
    bm: PlaneWaveBenchmark,
    epochs: int = 5000,
    n_interior: int = 2048,
    n_bc: int = 256,
    lr: float = 5e-4,
    hidden_layers: int = 4,
    hidden_width: int = 64,
    device_str: str = "cpu",
    w_pde: float = 1.0,
    w_top: float = 100.0,
    w_bot: float = 50.0,
    seed: int = 42,
    use_fourier: bool = False,
    activation: str = "tanh",
) -> tuple[FieldMLP, list[dict]]:
    set_seed(seed)
    device = resolve_device(device_str)
    dtype = resolve_training_dtype("float64", device)

    # Build a minimal PhysicsConfig that matches the benchmark geometry
    physics = PhysicsConfig(
        wavelength=2.0 * np.pi / bm.k0,
        n_air=bm.n,
        n_ridge=bm.n,
        n_substrate=bm.n,
        period=bm.period,
        ridge_width=0.0,
        ridge_height=0.0,
        domain_height=bm.domain_height,
        ridge_base_fraction=1.0,  # no grating
    )
    model_cfg = ModelConfig(
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        activation=activation,
        fourier_features=use_fourier,
        num_fourier_features=32,
    )
    model = FieldMLP(model_cfg, physics).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _t(arr):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    # Interior points
    rng = np.random.default_rng(seed)
    z_int = _t(rng.uniform(0.0, bm.domain_height, n_interior))
    x_int = _t(rng.uniform(0.0, bm.period, n_interior))

    # Top BC (z=0)
    z_top = _t(np.zeros(n_bc))
    x_top = _t(rng.uniform(0.0, bm.period, n_bc))

    # Bottom BC
    z_bot = _t(np.full(n_bc, bm.domain_height))
    x_bot = _t(rng.uniform(0.0, bm.period, n_bc))

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        res_r, res_i = helmholtz_residual_uniform(model, x_int, z_int, bm.n, bm.k0)
        pde = torch.mean(res_r**2 + res_i**2)
        top = top_bc_loss(model, x_top, z_top, bm.k0)
        bot = robin_bc_loss(model, x_bot, z_bot, bm.k0 * bm.n)
        loss = w_pde * pde + w_top * top + w_bot * bot
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            history.append({
                "epoch": epoch,
                "pde": float(pde), "top": float(top), "bot": float(bot),
                "total": float(loss),
            })

    # LBFGS refinement
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        r_r, r_i = helmholtz_residual_uniform(model, x_int, z_int, bm.n, bm.k0)
        pde_ = torch.mean(r_r**2 + r_i**2)
        top_ = top_bc_loss(model, x_top, z_top, bm.k0)
        bot_ = robin_bc_loss(model, x_bot, z_bot, bm.k0 * bm.n)
        l = w_pde * pde_ + w_top * top_ + w_bot * bot_
        l.backward()
        return l
    for _ in range(200):
        lbfgs.step(closure)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation and plotting
# ---------------------------------------------------------------------------


def evaluate_plane_wave(
    model: FieldMLP,
    bm: PlaneWaveBenchmark,
    output_dir: Path,
    device_str: str = "cpu",
) -> dict:
    device = resolve_device(device_str)
    dtype = resolve_training_dtype("float64", device)

    Nz = 256
    z_np = np.linspace(0.0, bm.domain_height, Nz)
    x_np = np.full(Nz, bm.period / 2.0)  # arbitrary x (solution is z-only)

    z_t = torch.as_tensor(z_np, device=device, dtype=dtype)
    x_t = torch.as_tensor(x_np, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        out = model(x_t, z_t)
    pinn_real = detach_numpy(out[:, 0])
    pinn_imag = detach_numpy(out[:, 1])
    pinn_mag  = np.sqrt(pinn_real**2 + pinn_imag**2)
    pinn_phase = np.arctan2(pinn_imag, pinn_real)

    ref_real, ref_imag = bm.analytical_field_np(x_np, z_np)
    ref_mag   = np.sqrt(ref_real**2 + ref_imag**2)
    ref_phase = np.arctan2(ref_imag, ref_real)

    errors = evaluate_benchmark_errors(pinn_real, pinn_imag, ref_real, ref_imag)

    # PDE residual
    from src.derivatives import field_laplacian
    k0 = bm.k0
    x_r = torch.as_tensor(x_np, device=device, dtype=dtype)
    z_r = torch.as_tensor(z_np, device=device, dtype=dtype)
    x_nd = (x_r * k0).detach().clone().requires_grad_(True)
    z_nd = (z_r * k0).detach().clone().requires_grad_(True)
    model.train()
    e_re, e_im = model.field_components_nd(x_nd, z_nd)
    _, _, _, _, lap_r, lap_i = field_laplacian(e_re, e_im, x_nd, z_nd)
    res_r = lap_r + (bm.n**2) * e_re
    res_i = lap_i + (bm.n**2) * e_im
    pde_mse = float(torch.mean(res_r**2 + res_i**2).detach())

    # Plotting
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(z_np, ref_real, "b-", label="Analytical Re{E}")
    axes[0, 0].plot(z_np, pinn_real, "r--", label="PINN Re{E}")
    axes[0, 0].set_xlabel("z"); axes[0, 0].set_ylabel("Re{E}")
    axes[0, 0].set_title("Real field"); axes[0, 0].legend()

    axes[0, 1].plot(z_np, ref_imag, "b-", label="Analytical Im{E}")
    axes[0, 1].plot(z_np, pinn_imag, "r--", label="PINN Im{E}")
    axes[0, 1].set_xlabel("z"); axes[0, 1].set_ylabel("Im{E}")
    axes[0, 1].set_title("Imaginary field"); axes[0, 1].legend()

    axes[1, 0].plot(z_np, ref_mag, "b-", label="|E| analytical")
    axes[1, 0].plot(z_np, pinn_mag, "r--", label="|E| PINN")
    axes[1, 0].set_xlabel("z"); axes[1, 0].set_ylabel("|E|")
    axes[1, 0].set_title("Field magnitude"); axes[1, 0].legend()

    axes[1, 1].plot(z_np, ref_phase, "b-", label="phase analytical")
    axes[1, 1].plot(z_np, pinn_phase, "r--", label="phase PINN")
    axes[1, 1].set_xlabel("z"); axes[1, 1].set_ylabel("phase (rad)")
    axes[1, 1].set_title("Field phase"); axes[1, 1].legend()

    fig.suptitle(
        f"Plane-wave benchmark (n={bm.n})\n"
        f"L2_real={errors['relative_l2_real']:.3e}  "
        f"L2_imag={errors['relative_l2_imag']:.3e}  "
        f"L2_mag={errors['relative_l2_magnitude']:.3e}  "
        f"PDE_MSE={pde_mse:.3e}"
    )
    fig.savefig(output_dir / "bm1_plane_wave.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "bm1_plane_wave.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {output_dir / 'bm1_plane_wave.png'}")

    return {**errors, "pde_mse": pde_mse}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plane-wave PINN benchmark")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-medium", type=float, default=1.0, help="Refractive index")
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | cuda | mps  (default: cpu for float64)")
    parser.add_argument("--fourier", action="store_true", help="Use Fourier features")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "sin"])
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    bm = PlaneWaveBenchmark(
        n=args.n_medium,
        k0=2.0 * np.pi,
        domain_height=2.0,
        period=1.0,
    )

    print(f"\n=== Plane-wave benchmark  n={bm.n}  k0={bm.k0:.4f} ===")
    print(f"  Convention: E_inc(z) = exp(-i k0 z)  =>  Re=cos(k0 z), Im=-sin(k0 z)")
    print(f"  x̃ = k0 x  ∈ [0, {bm.k0 * bm.period:.2f}]")
    print(f"  z̃ = k0 z  ∈ [0, {bm.k0 * bm.domain_height:.2f}]")
    print(f"  Network: {args.hidden_layers}×{args.hidden_width}  activation={args.activation}")
    print(f"  Fourier features: {args.fourier}")
    print()

    model, history = train_plane_wave(
        bm,
        epochs=args.epochs,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        device_str=args.device,
        use_fourier=args.fourier,
        activation=args.activation,
    )

    print("  Training history:")
    for row in history:
        print(f"    epoch={row['epoch']:5d}  pde={row['pde']:.3e}  "
              f"top={row['top']:.3e}  bot={row['bot']:.3e}  total={row['total']:.3e}")

    errors = evaluate_plane_wave(model, bm, output_dir, device_str=args.device)

    print(f"\n  Results:")
    for k, v in errors.items():
        print(f"    {k}: {v:.4e}")

    passed = errors["relative_l2_magnitude"] < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"\n  Benchmark status: {status}  "
          f"(relative_l2_magnitude = {errors['relative_l2_magnitude']:.3e}, threshold < 0.01)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
