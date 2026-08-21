"""Evaluation of trained PINN on regular grids and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.geometry import epsilon_r_grid
from src.losses import compute_losses
from src.model import FieldMLP
from src.physics import helmholtz_residual
from src.sampling import sample_points
from src.utils import detach_numpy, resolve_device, resolve_training_dtype

if TYPE_CHECKING:
    from src.config import PINNConfig


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[FieldMLP, dict[str, Any]]:
    """Load model from checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    config: PINNConfig = ckpt["config"]
    dtype = resolve_training_dtype(config.training.dtype, device)
    model = FieldMLP(config.model, config.physics).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {"epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss")}
    return model, {"config": config, **meta}


def evaluate_field_on_grid(model: FieldMLP, config: PINNConfig, device: torch.device) -> dict[str, np.ndarray]:
    """Evaluate complex field and derived quantities on visualization grid."""
    dtype = resolve_training_dtype(config.training.dtype, device)
    x_grid, z_grid, eps_grid = epsilon_r_grid(config.physics, device, dtype)

    x_flat = x_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    with torch.no_grad():
        out = model(x_flat, z_flat)
    e_real = out[:, 0].reshape(x_grid.shape)
    e_imag = out[:, 1].reshape(x_grid.shape)

    e_real_np = detach_numpy(e_real)
    e_imag_np = detach_numpy(e_imag)
    magnitude = np.sqrt(e_real_np**2 + e_imag_np**2)
    intensity = magnitude**2
    intensity_norm = intensity / (intensity.max() + 1e-30)
    phase = np.arctan2(e_imag_np, e_real_np)

    return {
        "x": detach_numpy(x_grid),
        "z": detach_numpy(z_grid),
        "eps_r": detach_numpy(eps_grid),
        "E_real": e_real_np,
        "E_imag": e_imag_np,
        "magnitude": magnitude,
        "intensity": intensity,
        "intensity_normalized": intensity_norm,
        "phase": phase,
    }


def evaluate_metrics(model: FieldMLP, config: PINNConfig, device: torch.device) -> dict[str, float]:
    """Compute validation metrics: PDE residual and periodic BC error."""
    dtype = resolve_training_dtype(config.training.dtype, device)
    samples = sample_points(
        config.physics,
        config.sampling,
        device=device,
        dtype=dtype,
        seed=config.training.seed + 999,
    )

    with torch.enable_grad():
        loss_dict = compute_losses(model, samples, config.physics, config.loss_weights)
        res_real, res_imag = helmholtz_residual(
            model, samples.interior_x, samples.interior_z, config.physics
        )
        pde_mse = float(torch.mean(res_real**2 + res_imag**2).detach().cpu())

    metrics = {
        "pde_mse": pde_mse,
        "periodic_mse": float(loss_dict["periodic"].detach().cpu()),
        "top_bc_mse": float(loss_dict["top"].detach().cpu()),
        "bottom_bc_mse": float(loss_dict["bottom"].detach().cpu()),
        "total_loss": float(loss_dict["total"].detach().cpu()),
    }
    return metrics


def evaluate_pinn(
    config: PINNConfig,
    checkpoint_path: str | Path,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full evaluation pipeline: grid fields, metrics, NPZ/JSON export.

    Parameters
    ----------
    config :
        Loaded PINN configuration.
    checkpoint_path :
        Path to the trained model checkpoint (``.pt`` file).
    reference_path :
        Optional path to a reference-field NPZ file for independent
        validation.  When ``None`` or the file does not exist, reference
        comparison is skipped and ``"reference_validation": None`` is
        included in the returned dictionary.

    Returns
    -------
    dict
        Keys: ``results_file``, ``metrics_file``, ``metrics``,
        ``checkpoint_epoch``, ``reference_validation``.
    """
    device = resolve_device(config.training.device)
    model, meta = load_checkpoint(checkpoint_path, device)

    fields = evaluate_field_on_grid(model, config, device)
    metrics = evaluate_metrics(model, config, device)

    results_path = Path(config.paths.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(results_path, **fields)

    metrics_path = Path(config.paths.metrics_file)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Optional reference comparison                                        #
    # ------------------------------------------------------------------ #
    reference_validation: dict[str, Any] | None = None
    if reference_path is not None:
        from src.reference_data import run_comparison  # local import to keep deps optional

        # Extract 1-D coordinate axes from the evaluation grid
        x_grid = fields["x"]   # shape (Nz, Nx) from epsilon_r_grid
        z_grid = fields["z"]
        pinn_x_1d = x_grid[0, :] if x_grid.ndim == 2 else x_grid
        pinn_z_1d = z_grid[:, 0] if z_grid.ndim == 2 else z_grid

        figure_dir = Path(config.paths.figure_dir)
        reference_validation = run_comparison(
            reference_path=reference_path,
            pinn_x=pinn_x_1d,
            pinn_z=pinn_z_1d,
            pinn_E_real=fields["E_real"],
            pinn_E_imag=fields["E_imag"],
            output_dir=figure_dir,
            save_figure=True,
        )
    else:
        reference_validation = None

    return {
        "results_file": str(results_path),
        "metrics_file": str(metrics_path),
        "metrics": metrics,
        "checkpoint_epoch": meta.get("epoch"),
        "reference_validation": reference_validation,
    }
