"""Publication-quality figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    }
)


def _extent(x: np.ndarray, z: np.ndarray) -> list[float]:
    return [float(x.min()), float(x.max()), float(z.max()), float(z.min())]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_geometry(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Plot permittivity map."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data["eps_r"], extent=_extent(data["x"], data["z"]), aspect="auto", cmap="viridis")
    ax.set_xlabel("x (λ)")
    ax.set_ylabel("z (λ)")
    ax.set_title("Relative permittivity εr(x, z)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("εr")
    path = output_dir / "01_geometry_epsr.png"
    _save(fig, path)
    return path


def plot_scalar_field(
    data: dict[str, np.ndarray],
    field_key: str,
    title: str,
    cbar_label: str,
    cmap: str,
    filename: str,
    output_dir: Path,
    log_scale: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    field = data[field_key]
    if log_scale:
        plot_data = np.log10(field + 1e-30)
        cbar_label = f"log10({cbar_label})"
    else:
        plot_data = field
    im = ax.imshow(plot_data, extent=_extent(data["x"], data["z"]), aspect="auto", cmap=cmap)
    ax.set_xlabel("x (λ)")
    ax.set_ylabel("z (λ)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label(cbar_label)
    path = output_dir / filename
    _save(fig, path)
    return path


def plot_pde_residual_from_arrays(
    x: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(residual, extent=_extent(x, z), aspect="auto", cmap="magma")
    ax.set_xlabel("x (λ)")
    ax.set_ylabel("z (λ)")
    ax.set_title("PDE residual magnitude")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("|r|")
    path = output_dir / "07_pde_residual.png"
    _save(fig, path)
    return path


def plot_training_history(history_file: Path, output_dir: Path) -> Path:
    """Plot training loss curves."""
    df = pd.read_csv(history_file)
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ("total", "pde", "periodic", "top", "bottom"):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    if "val_total" in df.columns:
        ax.plot(df["epoch"], df["val_total"], "--", label="val_total", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history")
    ax.set_yscale("log")
    ax.legend()
    path = output_dir / "08_training_loss.png"
    _save(fig, path)
    return path


def compute_pde_residual_grid(model, config, device) -> np.ndarray:
    """Evaluate PDE residual magnitude on visualization grid."""
    import torch

    from src.geometry import epsilon_r_grid
    from src.physics import helmholtz_residual
    from src.utils import detach_numpy, resolve_training_dtype

    dtype = resolve_training_dtype(config.training.dtype, device)
    x_grid, z_grid, _ = epsilon_r_grid(config.physics, device, dtype)
    x_flat = x_grid.reshape(-1).clone().requires_grad_(True)
    z_flat = z_grid.reshape(-1).clone().requires_grad_(True)
    res_real, res_imag = helmholtz_residual(model, x_flat, z_flat, config.physics)
    residual = torch.sqrt(res_real**2 + res_imag**2).reshape(x_grid.shape)
    return detach_numpy(residual)


def plot_reference_magnitude_comparison(
    x: np.ndarray,
    z: np.ndarray,
    ref_magnitude: np.ndarray,
    pinn_magnitude: np.ndarray,
    abs_error: np.ndarray,
    valid_mask: np.ndarray | None = None,
    output_dir: Path | None = None,
    filename: str = "reference_comparison",
) -> plt.Figure:
    """Three-panel side-by-side comparison: reference |E|, PINN |E|, absolute error.

    Parameters
    ----------
    x :
        1-D horizontal coordinate array, shape ``(Nx,)``.
    z :
        1-D vertical coordinate array, shape ``(Nz,)``.
    ref_magnitude :
        Reference field magnitude, shape ``(Nz, Nx)``.
    pinn_magnitude :
        PINN field magnitude, shape ``(Nz, Nx)``.
    abs_error :
        Absolute magnitude error ``|PINN_mag − ref_mag|``, shape ``(Nz, Nx)``.
    valid_mask :
        Boolean mask of valid (finite) points, shape ``(Nz, Nx)``.  Invalid
        points are rendered as white using ``np.nan``.
    output_dir :
        Directory for output files.  Figures are saved only when this is not
        ``None``.
    filename :
        Base filename without extension.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (already closed if ``output_dir`` is provided).
    """
    extent = [float(x.min()), float(x.max()), float(z.max()), float(z.min())]

    # Apply mask: set invalid points to NaN for display
    def _masked(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(float).copy()
        if valid_mask is not None:
            out[~valid_mask] = np.nan
        return out

    ref_plot = _masked(ref_magnitude)
    pinn_plot = _masked(pinn_magnitude)
    err_plot = _masked(abs_error)

    # Shared colour limits for the first two panels
    vmin = float(np.nanmin(np.concatenate([ref_plot.ravel(), pinn_plot.ravel()])))
    vmax = float(np.nanmax(np.concatenate([ref_plot.ravel(), pinn_plot.ravel()])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Panel (a) – Reference
    im0 = axes[0].imshow(
        ref_plot, extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("(a) Reference |E|")
    axes[0].set_xlabel("x (λ)")
    axes[0].set_ylabel("z (λ)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046)
    cbar0.set_label("|E|")

    # Panel (b) – PINN
    im1 = axes[1].imshow(
        pinn_plot, extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("(b) PINN |E|")
    axes[1].set_xlabel("x (λ)")
    axes[1].set_ylabel("z (λ)")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046)
    cbar1.set_label("|E|")

    # Panel (c) – Absolute error
    im2 = axes[2].imshow(err_plot, extent=extent, aspect="auto", cmap="magma")
    axes[2].set_title("(c) ||PINN| − |Reference||")
    axes[2].set_xlabel("x (λ)")
    axes[2].set_ylabel("z (λ)")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046)
    cbar2.set_label("Absolute error")

    fig.suptitle("PINN vs. Reference — Field Magnitude Comparison", fontsize=13)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / f"{filename}.png"
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def generate_all_figures(
    results_npz: Path,
    history_file: Path | None,
    output_dir: Path,
    model=None,
    config=None,
    device=None,
) -> list[Path]:
    """Generate full figure set from evaluation results."""
    data = dict(np.load(results_npz))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(plot_geometry(data, output_dir))
    paths.append(
        plot_scalar_field(data, "E_real", "Real field Re{E}", "Re{E}", "RdBu_r", "02_field_real.png", output_dir)
    )
    paths.append(
        plot_scalar_field(data, "E_imag", "Imaginary field Im{E}", "Im{E}", "RdBu_r", "03_field_imag.png", output_dir)
    )
    paths.append(
        plot_scalar_field(data, "magnitude", "Field magnitude |E|", "|E|", "viridis", "04_field_magnitude.png", output_dir)
    )
    paths.append(
        plot_scalar_field(
            data,
            "intensity_normalized",
            "Normalized intensity |E|²",
            "|E|² (normalized)",
            "magma",
            "05_intensity_log.png",
            output_dir,
            log_scale=True,
        )
    )
    paths.append(
        plot_scalar_field(data, "phase", "Field phase arg(E)", "phase (rad)", "twilight", "06_field_phase.png", output_dir)
    )

    if model is not None and config is not None and device is not None:
        residual = compute_pde_residual_grid(model, config, device)
        paths.append(plot_pde_residual_from_arrays(data["x"], data["z"], residual, output_dir))

    if history_file is not None and history_file.exists():
        paths.append(plot_training_history(history_file, output_dir))

    return paths
