#!/usr/bin/env python3
"""Train and evaluate the 2-D Maxwell PINN with staged benchmark validation.

Staged benchmark progression (A → E, must pass in order):
    A  homogeneous   -- uniform medium, plane-wave solution
    B  layered       -- full-width slab, must match 1-D benchmark
    C  shallow       -- reduced ridge height and contrast
    D  non_rayleigh  -- period=1.5λ, ±1 orders propagating
    E  grating       -- full original grating (default)

Cases A and B use the single-network Maxwell PINN (no domain decomposition
needed since there are no lateral grating interfaces).
Cases C, D, E use the domain-decomposition Maxwell PINN.

Usage::

    .venv/bin/python scripts/train_maxwell_dd.py --case homogeneous --epochs 3000
    .venv/bin/python scripts/train_maxwell_dd.py --case layered --epochs 5000
    .venv/bin/python scripts/train_maxwell_dd.py --case grating --epochs 8000 \\
        --reference outputs/reference_grating.npz --diagnostics --boundary-audit
    .venv/bin/python scripts/train_maxwell_dd.py --case non_rayleigh \\
        --epochs 8000 --gen-reference
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
from src.config import load_config
from src.geometry import epsilon_r_grid
from src.maxwell_benchmarks_2d import (
    BenchmarkCase,
    make_full_grating,
    make_homogeneous,
    make_horizontal_layers,
    make_non_rayleigh_grating,
    make_shallow_grating,
    print_diffraction_order_analysis,
)
from src.maxwell_2d import (
    Maxwell2DMLP,
    maxwell_2d_total_loss,
    sample_collocation_points,
)
from src.maxwell_2d_dd import (
    Maxwell2DDD,
    maxwell_2d_dd_total_loss,
    sample_dd_points,
)
from src.maxwell_diagnostics import (
    audit_boundary,
    compute_diffraction_efficiencies,
    compute_residual_by_region,
    compute_residual_map,
    compute_residual_near_corners,
    plot_residual_maps,
    residual_based_refinement,
)
from src.reference_data import (
    interpolate_reference_to_grid,
    load_reference_npz,
    normalize_reference_orientation,
)
from src.utils import detach_numpy, resolve_device, resolve_training_dtype, set_seed


# ---------------------------------------------------------------------------
# PDE implementation check
# ---------------------------------------------------------------------------


def check_homogeneous_pde(physics, device, dtype) -> bool:
    """Verify PDE residual = 0 for the exact plane-wave in air."""
    from src.maxwell_2d_dd import maxwell_2d_dd_pde_residual
    from src.geometry import epsilon_r
    k0 = physics.k0

    class ExactPW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def field_components(self, x, z):
            n = physics.n_air
            Er  =  torch.cos(k0*z) + self.dummy*0 + x*0
            Ei  = -torch.sin(k0*z) + self.dummy*0 + x*0
            Hrx = n*Er; Hix = n*Ei
            Hrz = torch.zeros_like(Er) + self.dummy*0 + x*0
            Hiz = torch.zeros_like(Er) + self.dummy*0 + x*0
            return Er, Ei, Hrx, Hix, Hrz, Hiz

    net = ExactPW()
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.uniform(0.1, physics.period-0.1, 64), dtype=dtype)
    z = torch.as_tensor(rng.uniform(0.1, min(physics.ridge_z_min-0.1, physics.domain_height*0.8), 64), dtype=dtype)
    def eps_fn(xp, zp): return epsilon_r(xp, zp, physics)
    with torch.enable_grad():
        res = maxwell_2d_dd_pde_residual(net, x, z, physics, eps_fn, scattered=True)
    mse = float(sum(torch.mean(r**2) for r in res).detach()/len(res))
    passed = mse < 1e-8
    print(f"  [PDE check] exact plane-wave residual: {mse:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Single-network trainer (Cases A and B)
# ---------------------------------------------------------------------------


def train_single(
    case: BenchmarkCase, config, output_dir: Path, device_str: str,
    hidden_layers: int, hidden_width: int, num_fourier_levels: int,
    scattered: bool, n_interior: int, n_bc: int, epochs: int,
) -> tuple[Maxwell2DMLP, list, Path, torch.device, torch.dtype]:
    """Train the single-network Maxwell PINN (no domain decomposition).

    Used for homogeneous (Case A) and horizontal-layer (Case B) benchmarks
    where there are no lateral interfaces and the DD architecture is unnecessary.
    """
    physics = case.physics
    set_seed(config.training.seed)
    device = resolve_device(device_str)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    model = Maxwell2DMLP(physics, hidden_layers, hidden_width, num_fourier_levels)
    model = model.to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    w_pde = config.loss_weights.pde
    w_top = config.loss_weights.top
    w_bot = config.loss_weights.bottom
    w_per = config.loss_weights.periodic

    print(f"  [Single-network PINN]  Parameters: {n_params:,}")
    print(f"  Loss weights: pde={w_pde}  top={w_top}  bot={w_bot}  per={w_per}")

    def _resample(seed_off=0):
        return sample_collocation_points(
            physics, n_interior, n_bc, n_bc, n_bc,
            device, dtype, seed=config.training.seed + seed_off,
        )

    pts = _resample(0)
    opt   = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: list[dict] = []
    best_total = float("inf")
    best_state = None
    best_epoch = 0
    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        if ep % 500 == 0:
            pts = _resample(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = maxwell_2d_total_loss(model, pts, physics, w_pde, w_top, w_bot, w_per, scattered)
        losses["total"].backward()
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        opt.step(); sched.step()

        if ep % config.training.validation_interval == 0 or ep == 1:
            tv = float(losses["total"].detach())
            row = {
                "epoch": ep,
                "pde": float(losses["pde"].detach()),
                "top": float(losses["top"].detach()),
                "bottom": float(losses["bottom"].detach()),
                "total": tv,
            }
            history.append(row)
            if tv < best_total:
                best_total = tv; best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ep % (config.training.validation_interval * 5) == 0:
                print(f"  ep={ep:6d}  pde={row['pde']:.2e}  "
                      f"top={row['top']:.2e}  bot={row['bottom']:.2e}"
                      + ("  *best*" if ep == best_epoch else ""))

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ckpt_dir / f"{case.tag}_best.pt"
    torch.save({
        "best_epoch": best_epoch, "best_loss": best_total,
        "model_state_dict": model.state_dict(), "config": config,
        "scattered": scattered, "num_fourier_levels": num_fourier_levels,
        "hidden_layers": hidden_layers, "hidden_width": hidden_width,
        "case_tag": case.tag, "single_network": True,
    }, ckpt_path)
    print(f"  Best: total={best_total:.4e}  epoch={best_epoch}")
    pd.DataFrame(history).to_csv(
        Path(config.paths.history_file).parent / f"{case.tag}_history.csv", index=False
    )
    return model, history, ckpt_path, device, dtype


# ---------------------------------------------------------------------------
# Domain-decomposition trainer (Cases C, D, E)
# ---------------------------------------------------------------------------


def train_dd(
    case: BenchmarkCase, config, output_dir: Path, device_str: str,
    hidden_layers: int, hidden_width: int, num_fourier_levels: int,
    scattered: bool, n_per_region: int, n_interface: int, n_bc: int,
    epochs: int, use_adaptive: bool = False,
) -> tuple[Maxwell2DDD, list, Path, torch.device, torch.dtype]:
    """Train the domain-decomposition Maxwell PINN."""
    physics = case.physics
    set_seed(config.training.seed)
    device = resolve_device(device_str)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    model = Maxwell2DDD(physics, hidden_layers, hidden_width, num_fourier_levels)
    model = model.to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    w_pde = config.loss_weights.pde
    w_E   = config.loss_weights.periodic
    w_H   = config.loss_weights.periodic
    w_top = config.loss_weights.top
    w_bot = config.loss_weights.bottom

    print(f"  [DD PINN]  Parameters: {n_params:,}  (3 × {n_params//3:,})")
    print(f"  Formulation: {'scattered-field' if scattered else 'total-field'}")
    print(f"  Loss weights: pde={w_pde}  E_int={w_E}  H_int={w_H}  top={w_top}  bot={w_bot}")
    print(f"  Points/region: {n_per_region}  interface: {n_interface}  bc: {n_bc}")

    def _resample(seed_off=0):
        return sample_dd_points(
            physics, n_per_region, n_interface, n_bc, n_bc,
            device, dtype, seed=config.training.seed + seed_off,
        )

    pts = _resample(0)
    opt   = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: list[dict] = []
    best_total = float("inf")
    best_state = None
    best_epoch = 0
    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        if ep % 500 == 0:
            pts = _resample(ep)
            if use_adaptive and ep % 2000 == 0:
                new_pts = residual_based_refinement(
                    model, physics, device, dtype, n_new=256, scattered=scattered, seed=ep
                )
                xn, zn = new_pts["x_new"], new_pts["z_new"]
                for key, mask in [
                    ("air",  zn <= physics.ridge_z_min),
                    ("grat", (zn > physics.ridge_z_min) & (zn <= physics.ridge_z_max)),
                    ("sub",  zn > physics.ridge_z_max),
                ]:
                    if mask.any():
                        pts[f"x_{key}"] = torch.cat([pts[f"x_{key}"], xn[mask]])
                        pts[f"z_{key}"] = torch.cat([pts[f"z_{key}"], zn[mask]])

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = maxwell_2d_dd_total_loss(
            model, pts, physics, w_pde, w_E, w_H, w_top, w_bot, scattered
        )
        losses["total"].backward()
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        opt.step(); sched.step()

        if ep % config.training.validation_interval == 0 or ep == 1:
            tv = float(losses["total"].detach())
            row = {
                "epoch":    ep,
                "pde_air":  float(losses["pde_air"].detach()),
                "pde_grat": float(losses["pde_grat"].detach()),
                "pde_sub":  float(losses["pde_sub"].detach()),
                "E_int":    float((losses["E_int1"] + losses["E_int2"]).detach()),
                "H_int":    float((losses["H_int1"] + losses["H_int2"]).detach()),
                "top":      float(losses["top"].detach()),
                "bottom":   float(losses["bottom"].detach()),
                "total":    tv,
            }
            history.append(row)
            if tv < best_total:
                best_total = tv; best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ep % (config.training.validation_interval * 5) == 0:
                print(f"  ep={ep:6d}  "
                      f"pde_air={row['pde_air']:.2e}  pde_grat={row['pde_grat']:.2e}  pde_sub={row['pde_sub']:.2e}  "
                      f"E_int={row['E_int']:.2e}  H_int={row['H_int']:.2e}  "
                      f"top={row['top']:.2e}  bot={row['bottom']:.2e}"
                      + ("  *best*" if ep == best_epoch else ""))

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = ckpt_dir / f"{case.tag}_best.pt"
    torch.save({
        "best_epoch": best_epoch, "best_loss": best_total,
        "model_state_dict": model.state_dict(), "config": config,
        "scattered": scattered, "num_fourier_levels": num_fourier_levels,
        "hidden_layers": hidden_layers, "hidden_width": hidden_width,
        "case_tag": case.tag, "single_network": False,
    }, ckpt_path)
    print(f"  Best: total={best_total:.4e}  epoch={best_epoch}")
    pd.DataFrame(history).to_csv(
        Path(config.paths.history_file).parent / f"{case.tag}_history.csv", index=False
    )
    return model, history, ckpt_path, device, dtype


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_single(model: Maxwell2DMLP, case: BenchmarkCase, device, dtype, scattered: bool) -> dict:
    """Evaluate single-network model on visualization grid."""
    physics = case.physics
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1); z_flat = z_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        out = model.forward(x_flat, z_flat)
    shape = x_grid.shape
    Er_net = detach_numpy(out[:, 0].reshape(shape))
    Ei_net = detach_numpy(out[:, 1].reshape(shape))

    if scattered:
        z_np = detach_numpy(z_grid); k0 = physics.k0
        Er = Er_net + np.cos(k0*z_np)
        Ei = Ei_net - np.sin(k0*z_np)
    else:
        Er, Ei = Er_net, Ei_net

    mag = np.sqrt(Er**2 + Ei**2)
    return {
        "x": detach_numpy(x_grid), "z": detach_numpy(z_grid),
        "eps_r": detach_numpy(eps_grid),
        "E_real": Er, "E_imag": Ei,
        "E_scat_real": Er_net, "E_scat_imag": Ei_net,
        "magnitude": mag, "phase": np.arctan2(Ei, Er),
        "intensity": mag**2,
    }


def evaluate_dd(model: Maxwell2DDD, case: BenchmarkCase, device, dtype, scattered: bool) -> dict:
    """Evaluate DD model on visualization grid."""
    physics = case.physics
    x_grid, z_grid, eps_grid = epsilon_r_grid(physics, device, dtype)
    x_flat = x_grid.reshape(-1); z_flat = z_grid.reshape(-1)

    model.eval()
    with torch.no_grad():
        Er_s = torch.zeros(x_flat.shape[0], dtype=dtype, device=device)
        Ei_s = torch.zeros_like(Er_s)
        m_air  = z_flat <= physics.ridge_z_min
        m_grat = (z_flat > physics.ridge_z_min) & (z_flat <= physics.ridge_z_max)
        m_sub  = z_flat > physics.ridge_z_max
        for mask, net in [(m_air, model.net_air), (m_grat, model.net_grat), (m_sub, model.net_sub)]:
            if mask.any():
                out = net.forward(x_flat[mask], z_flat[mask])
                Er_s[mask] = out[:, 0]; Ei_s[mask] = out[:, 1]

    shape = x_grid.shape
    Er_net = detach_numpy(Er_s.reshape(shape))
    Ei_net = detach_numpy(Ei_s.reshape(shape))

    if scattered:
        z_np = detach_numpy(z_grid); k0 = physics.k0
        Er = Er_net + np.cos(k0*z_np)
        Ei = Ei_net - np.sin(k0*z_np)
    else:
        Er, Ei = Er_net, Ei_net

    mag = np.sqrt(Er**2 + Ei**2)
    return {
        "x": detach_numpy(x_grid), "z": detach_numpy(z_grid),
        "eps_r": detach_numpy(eps_grid),
        "E_real": Er, "E_imag": Ei,
        "E_scat_real": Er_net, "E_scat_imag": Ei_net,
        "magnitude": mag, "phase": np.arctan2(Ei, Er),
        "intensity": mag**2,
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def compare_to_analytical(fields: dict, case: BenchmarkCase) -> dict:
    """Compare x-averaged field to 1-D analytical solution."""
    physics = case.physics
    z_np = fields["z"][:, 0]
    pinn_Er = fields["E_real"].mean(axis=1)
    pinn_Ei = fields["E_imag"].mean(axis=1)

    if case.tag == "2d_hom":
        k0 = physics.k0; n = physics.n_air
        ref_Er = np.cos(k0*n*z_np)
        ref_Ei = -np.sin(k0*n*z_np)
    elif case.tag == "2d_lay":
        bm = LayeredMediumBenchmark(
            n_air=physics.n_air, n_slab=physics.n_ridge, n_sub=physics.n_substrate,
            k0=physics.k0, z_slab_top=physics.ridge_z_min, z_slab_bot=physics.ridge_z_max,
            domain_height=physics.domain_height,
        )
        ref_Er, ref_Ei = bm.analytical_field_np(np.zeros_like(z_np), z_np)
    else:
        return {}

    errs = evaluate_benchmark_errors(pinn_Er, pinn_Ei, ref_Er, ref_Ei)
    # x-variation check (should be ~0 for layered/homogeneous)
    errs["x_variation_std"] = float(np.std(fields["E_real"], axis=1).mean())
    return errs


def save_figures(fields: dict, case: BenchmarkCase, output_dir: Path, history: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, z = fields["x"], fields["z"]
    ext = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]

    def _s(fig, name):
        fig.savefig(output_dir / f"{case.tag}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for key, title, cmap in [
        ("eps_r",     "εr(x,z)",   "viridis"),
        ("E_real",    "Re{E_y}",   "RdBu_r"),
        ("E_imag",    "Im{E_y}",   "RdBu_r"),
        ("magnitude", "|E_y|",     "viridis"),
        ("phase",     "Phase",     "twilight"),
    ]:
        if key not in fields: continue
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(fields[key], extent=ext, aspect="auto", cmap=cmap)
        ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        _s(fig, key)

    if history:
        df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in ("pde", "pde_air", "pde_grat", "pde_sub", "top", "bottom"):
            if col in df.columns and not df[col].isna().all():
                ax.semilogy(df["epoch"], df[col].clip(lower=1e-12), label=col)
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.set_title("Training losses")
        _s(fig, "history")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="2-D Maxwell PINN staged benchmarks")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--reference",      default=None)
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--epochs",         type=int,   default=5000)
    parser.add_argument("--n-per-region",   type=int,   default=1024)
    parser.add_argument("--n-interface",    type=int,   default=256)
    parser.add_argument("--n-bc",           type=int,   default=256)
    parser.add_argument("--hidden-layers",  type=int,   default=4)
    parser.add_argument("--hidden-width",   type=int,   default=64)
    parser.add_argument("--fourier-levels", type=int,   default=6)
    parser.add_argument("--scattered",      action="store_true", default=True)
    parser.add_argument("--total-field",    action="store_true")
    parser.add_argument("--case",           default="grating",
                        choices=["grating", "homogeneous", "layered", "shallow", "non_rayleigh"])
    parser.add_argument("--diagnostics",    action="store_true")
    parser.add_argument("--boundary-audit", action="store_true")
    parser.add_argument("--adaptive",       action="store_true")
    parser.add_argument("--gen-reference",  action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    config.training.epochs = args.epochs
    scattered = not args.total_field

    base = config.physics
    case_map = {
        "homogeneous": make_homogeneous(base),
        "layered":     make_horizontal_layers(base),
        "shallow":     make_shallow_grating(base),
        "non_rayleigh":make_non_rayleigh_grating(base),
        "grating":     make_full_grating(base),
    }
    case = case_map[args.case]
    config.physics = case.physics

    out = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "staged"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype  = resolve_training_dtype(config.training.dtype, device)

    print(f"\n=== 2-D Maxwell PINN  [{case.name}]  {'scattered' if scattered else 'total-field'} ===")
    print(f"  {case.description}")
    print(f"  Epochs={args.epochs}  arch={args.hidden_layers}×{args.hidden_width}  fourier={args.fourier_levels}")

    # Diffraction order analysis
    print_diffraction_order_analysis(case.physics)

    # Optional reference generation (for non-standard period)
    ref_path = None
    if args.gen_reference:
        from scripts.generate_reference import solve_rcwa
        ref_file = ROOT / f"outputs/reference_{case.tag}.npz"
        print(f"\n  Generating RCWA reference → {ref_file}")
        x_r, z_r, Er_r, Ei_r, _amps_r = solve_rcwa(case.physics, N_harmonics=75)
        np.savez(ref_file, x=x_r, z=z_r, E_real=Er_r, E_imag=Ei_r)
        ref_path = ref_file
        print(f"  Saved: {ref_path}")
    elif args.reference:
        ref_path = Path(args.reference)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
    elif case.reference_path:
        ref_path = ROOT / case.reference_path

    # PDE check
    if not check_homogeneous_pde(case.physics, device, dtype):
        print("  ERROR: PDE check failed. Aborting.")
        return 1

    # Choose trainer: single-network only for truly degenerate (zero grating) cases
    use_single = args.case == "homogeneous"

    if use_single:
        model, history, ckpt_path, device, dtype = train_single(
            case, config, out, args.device,
            args.hidden_layers, args.hidden_width, args.fourier_levels,
            scattered, args.n_per_region * 3, args.n_bc, args.epochs,
        )
        fields = evaluate_single(model, case, device, dtype, scattered)
    else:
        model, history, ckpt_path, device, dtype = train_dd(
            case, config, out, args.device,
            args.hidden_layers, args.hidden_width, args.fourier_levels,
            scattered, args.n_per_region, args.n_interface, args.n_bc, args.epochs,
            use_adaptive=args.adaptive,
        )
        fields = evaluate_dd(model, case, device, dtype, scattered)

    np.savez_compressed(out / f"{case.tag}_results.npz", **fields)
    save_figures(fields, case, out, history)

    # ---------- Metrics ----------
    metrics: dict = {"case": args.case, "scattered": scattered}
    print(f"\n=== Metric summary  [{case.name}] ===")

    # Analytical comparison (A, B)
    if args.case in ("homogeneous", "layered"):
        errs = compare_to_analytical(fields, case)
        metrics["analytical_errors"] = errs
        print(f"\n  vs. analytical solution:")
        for k, v in errs.items():
            print(f"    {k}: {v:.4e}")
        cl2 = errs.get("relative_complex_l2", 1.0)
        status = "PASS" if cl2 < case.success_complex_l2 else "FAIL"
        print(f"  Status: {status}  (complex_L2={cl2:.3e}, threshold < {case.success_complex_l2})")
        metrics["status"] = status

    # Reference comparison (C, D, E)
    if ref_path and Path(ref_path).exists():
        print(f"\n  Reference comparison (RCWA):")
        ref_data = normalize_reference_orientation(load_reference_npz(ref_path))
        pinn_x1d = fields["x"][0, :]; pinn_z1d = fields["z"][:, 0]
        ref_Er, ref_Ei, _ = interpolate_reference_to_grid(ref_data, pinn_x1d, pinn_z1d)
        valid = np.isfinite(ref_Er) & np.isfinite(ref_Ei)
        errs = evaluate_benchmark_errors(
            fields["E_real"][valid], fields["E_imag"][valid],
            ref_Er[valid], ref_Ei[valid],
        )
        for k, v in errs.items():
            print(f"    {k}: {v:.4e}")
        metrics["rcwa_errors"] = errs
        cl2 = errs["relative_complex_l2"]
        status = "PASS" if cl2 < case.success_complex_l2 else "FAIL"
        print(f"  Status: {status}  (complex_L2={cl2:.3e}, threshold < {case.success_complex_l2})")
        metrics["status"] = status

        # Diffraction efficiencies
        E_c = fields["E_real"] + 1j * fields["E_imag"]
        diff = compute_diffraction_efficiencies(E_c, pinn_x1d, pinn_z1d, case.physics, n_orders=3)
        ref_Ec = ref_Er + 1j * ref_Ei
        diff_rcwa = compute_diffraction_efficiencies(ref_Ec, pinn_x1d, pinn_z1d, case.physics, n_orders=3)
        print(f"\n  Diffraction efficiencies:")
        print(f"    {'':12}  {'PINN':>10}  {'RCWA':>10}  {'|Δ|':>10}")
        for key in ("R0", "T0", "R_total", "T_total"):
            pv = diff.get(key, float("nan"))
            rv = diff_rcwa.get(key, float("nan"))
            print(f"    {key:12}  {pv:10.4f}  {rv:10.4f}  {abs(pv-rv):10.4f}")
        print(f"    Energy (PINN): R+T = {diff['energy_check']:.4f}")
        print(f"    Energy (RCWA): R+T = {diff_rcwa['energy_check']:.4f}")
        metrics["diffraction_pinn"] = {k: float(v) for k, v in diff.items()
                                        if isinstance(v, (int, float, np.floating))}
        metrics["diffraction_rcwa"] = {k: float(v) for k, v in diff_rcwa.items()
                                        if isinstance(v, (int, float, np.floating))}

    # Residual diagnostics
    if args.diagnostics and not use_single:
        print("\n  Computing residual diagnostics...")
        res_map = compute_residual_map(model, case.physics, device, dtype,
                                       nx=64, nz=128, scattered=scattered)
        by_region = compute_residual_by_region(res_map)
        corners   = compute_residual_near_corners(res_map, case.physics)
        plot_residual_maps(res_map, case.physics, out / "residuals", tag=case.tag)
        print(f"  By region:")
        for reg, st in by_region.items():
            print(f"    {reg}: mean_mse={st['mean']:.3e}  max={st['max']:.3e}  p95={st['p95']:.3e}")
        print(f"  Near corners:")
        for cn, st in corners.items():
            print(f"    {cn}: mean_mse={st['mean_mse']:.3e}  max={st['max_res']:.3e}")
        metrics["residual_by_region"] = by_region
        metrics["residual_corners"]   = corners

    # Boundary audit
    if args.boundary_audit and not use_single:
        print("\n  Boundary audit (Robin vs. modal DtN):")
        audit = audit_boundary(model, case.physics, device, dtype, scattered=scattered)
        for k, v in audit.items():
            print(f"    {k}: {v}")
        metrics["boundary_audit"] = audit

    mp = out / f"{case.tag}_metrics.json"
    with mp.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Metrics: {mp}")
    print(f"  Figures: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
