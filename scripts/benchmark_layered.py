#!/usr/bin/env python3
"""Benchmark 2: Homogeneous layered medium (1-D, no grating).

Air | dielectric slab | substrate — no lateral periodicity.
Analytical solution via Transfer Matrix Method (TMM).

This benchmark tests that the PINN can:
1. Capture reflection and transmission at planar interfaces.
2. Represent a standing-wave pattern in the slab.
3. Reproduce correct phase in the substrate.

Success criterion: relative L2 error on |E| < 5%.
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

from src.benchmarks import LayeredMediumBenchmark, evaluate_benchmark_errors
from src.config import ModelConfig, PhysicsConfig
from src.derivatives import field_laplacian
from src.model import FieldMLP
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# Layered-medium PDE residual (εr depends only on z)
# ---------------------------------------------------------------------------


def _eps_tensor(z: torch.Tensor, bm: LayeredMediumBenchmark) -> torch.Tensor:
    eps = torch.full_like(z, bm.n_air**2)
    eps = torch.where((z > bm.z_slab_top) & (z <= bm.z_slab_bot),
                      torch.full_like(z, bm.n_slab**2), eps)
    eps = torch.where(z > bm.z_slab_bot,
                      torch.full_like(z, bm.n_sub**2), eps)
    return eps


def helmholtz_residual_layered(model, x, z, bm):
    from src.derivatives import field_laplacian
    k0 = bm.k0
    x_t = (x * k0).detach().clone().requires_grad_(True)
    z_t = (z * k0).detach().clone().requires_grad_(True)
    e_real, e_imag = model.field_components_nd(x_t, z_t)
    _, _, _, _, lap_real, lap_imag = field_laplacian(e_real, e_imag, x_t, z_t)
    eps = _eps_tensor(z, bm)
    return lap_real + eps * e_real, lap_imag + eps * e_imag


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_layered(
    bm: LayeredMediumBenchmark,
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
    activation: str = "tanh",
    use_fourier: bool = True,
) -> tuple[FieldMLP, list[dict]]:
    set_seed(seed)
    device = resolve_device(device_str)
    dtype = resolve_training_dtype("float64", device)

    physics = PhysicsConfig(
        wavelength=2.0 * np.pi / bm.k0,
        n_air=bm.n_air,
        n_ridge=bm.n_slab,
        n_substrate=bm.n_sub,
        period=bm.period,
        ridge_width=0.0,
        ridge_height=0.0,
        domain_height=bm.domain_height,
        ridge_base_fraction=bm.z_slab_top / bm.domain_height,
    )
    model_cfg = ModelConfig(
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        activation=activation,
        fourier_features=use_fourier,
        num_fourier_features=16,  # 4 levels covers k2=k0*n_slab frequencies
    )
    model = FieldMLP(model_cfg, physics).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def _t(arr):
        return torch.as_tensor(arr, device=device, dtype=dtype)

    rng = np.random.default_rng(seed)
    z_int = _t(rng.uniform(0.0, bm.domain_height, n_interior))
    x_int = _t(np.zeros(n_interior))   # x-independent problem

    z_top_ = _t(np.zeros(n_bc))
    x_top_ = _t(np.zeros(n_bc))

    z_bot_ = _t(np.full(n_bc, bm.domain_height))
    x_bot_ = _t(np.zeros(n_bc))

    def _top_loss():
        e = model(x_top_, z_top_)
        # E_total at z=0 = 1 + r  (incident + reflected)
        r = bm.reflection_coefficient()
        t_re = float(1.0 + r.real)
        t_im = float(r.imag)
        target_r = torch.full_like(z_top_, t_re)
        target_i = torch.full_like(z_top_, t_im)
        target = torch.stack([target_r, target_i], dim=-1)
        return torch.mean((e - target) ** 2)

    def _bot_loss():
        from src.derivatives import first_derivative
        k_sub = bm.k0 * bm.n_sub
        xr = x_bot_.detach().clone().requires_grad_(True)
        zr = z_bot_.detach().clone().requires_grad_(True)
        e_re, e_im = model.field_components(xr, zr)
        dEr_dz = first_derivative(e_re, zr)
        dEi_dz = first_derivative(e_im, zr)
        res_r = dEr_dz - k_sub * e_im
        res_i = dEi_dz + k_sub * e_re
        return torch.mean(res_r**2 + res_i**2)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        res_r, res_i = helmholtz_residual_layered(model, x_int, z_int, bm)
        pde = torch.mean(res_r**2 + res_i**2)
        top = _top_loss()
        bot = _bot_loss()
        loss = w_pde * pde + w_top * top + w_bot * bot
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % 500 == 0 or epoch == 1:
            history.append({"epoch": epoch, "pde": float(pde),
                             "top": float(top), "bot": float(bot), "total": float(loss)})

    # LBFGS
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20,
                                history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        r_r, r_i = helmholtz_residual_layered(model, x_int, z_int, bm)
        l = w_pde * torch.mean(r_r**2 + r_i**2) + w_top * _top_loss() + w_bot * _bot_loss()
        l.backward()
        return l
    for _ in range(200):
        lbfgs.step(closure)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_layered(model, bm, output_dir, device_str="cpu"):
    device = resolve_device(device_str)
    dtype = resolve_training_dtype("float64", device)
    Nz = 512
    z_np = np.linspace(0.0, bm.domain_height, Nz)
    x_np = np.zeros(Nz)
    z_t = torch.as_tensor(z_np, device=device, dtype=dtype)
    x_t = torch.as_tensor(x_np, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        out = model(x_t, z_t)
    pinn_real = detach_numpy(out[:, 0])
    pinn_imag = detach_numpy(out[:, 1])
    ref_real, ref_imag = bm.analytical_field_np(x_np, z_np)

    errors = evaluate_benchmark_errors(pinn_real, pinn_imag, ref_real, ref_imag)
    r = bm.reflection_coefficient()
    t_coef = bm.transmission_coefficient()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, (pr, rr, lbl) in zip(axes, [
        (pinn_real, ref_real, "Re{E}"),
        (pinn_imag, ref_imag, "Im{E}"),
        (np.sqrt(pinn_real**2+pinn_imag**2), np.sqrt(ref_real**2+ref_imag**2), "|E|"),
    ]):
        ax.plot(z_np, rr, "b-", label=f"Analytical {lbl}")
        ax.plot(z_np, pr, "r--", label=f"PINN {lbl}")
        ax.axvline(bm.z_slab_top, color="gray", ls=":", lw=0.8)
        ax.axvline(bm.z_slab_bot, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("z"); ax.set_ylabel(lbl); ax.legend(fontsize=8)

    fig.suptitle(
        f"Layered benchmark  r={r:.3f}  t={t_coef:.3f}\n"
        f"L2_real={errors['relative_l2_real']:.3e}  "
        f"L2_imag={errors['relative_l2_imag']:.3e}  "
        f"L2_mag={errors['relative_l2_magnitude']:.3e}"
    )
    fig.savefig(output_dir / "bm2_layered.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "bm2_layered.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {output_dir / 'bm2_layered.png'}")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Layered-medium PINN benchmark")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "sin"])
    parser.add_argument("--output-dir", type=str, default="outputs/benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    bm = LayeredMediumBenchmark(k0=2.0 * np.pi)
    r = bm.reflection_coefficient()
    t_coef = bm.transmission_coefficient()

    print(f"\n=== Layered-medium benchmark ===")
    print(f"  n_air={bm.n_air}  n_slab={bm.n_slab}  n_sub={bm.n_sub}")
    print(f"  slab: z=[{bm.z_slab_top}, {bm.z_slab_bot}]  (thickness={bm.z_slab_bot-bm.z_slab_top:.3f}λ)")
    print(f"  Analytical r={r:.4f}  |r|={abs(r):.4f}  |t|={abs(t_coef):.4f}")
    print(f"  Energy conservation: |r|²+|t|²*(n_sub/n_air) = "
          f"{abs(r)**2 + abs(t_coef)**2 * bm.n_sub / bm.n_air:.6f}  (should be 1.0)")
    print()

    model, history = train_layered(
        bm, epochs=args.epochs,
        hidden_layers=args.hidden_layers, hidden_width=args.hidden_width,
        device_str=args.device, activation=args.activation,
        use_fourier=True,
    )
    print("  Training history:")
    for row in history:
        print(f"    epoch={row['epoch']:5d}  pde={row['pde']:.3e}  "
              f"top={row['top']:.3e}  bot={row['bot']:.3e}")

    errors = evaluate_layered(model, bm, output_dir, device_str=args.device)
    print(f"\n  Results:")
    for k, v in errors.items():
        print(f"    {k}: {v:.4e}")

    passed = errors["relative_l2_magnitude"] < 0.05
    print(f"\n  Benchmark status: {'PASS' if passed else 'FAIL'}  "
          f"(threshold < 0.05, got {errors['relative_l2_magnitude']:.3e})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
