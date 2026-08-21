"""Reference electromagnetic solver data import and comparison.

This module supports independent validation of PINN solutions against field
data exported from an external electromagnetic solver (e.g., RCWA, FDTD, FEM).

No synthetic or fake reference data are generated here.  All comparison
functions require real data files produced by an independent solver.

Expected NPZ format
-------------------
The reference file must contain exactly these arrays::

    x       : shape (Nx,)     – 1-D horizontal coordinate array
    z       : shape (Nz,)     – 1-D vertical coordinate array
    E_real  : shape (Nz, Nx)  – real part of the complex field
    E_imag  : shape (Nz, Nx)  – imaginary part of the complex field

Field indexing convention::

    E(z_index, x_index) = E_real[z_index, x_index]
                        + 1j * E_imag[z_index, x_index]

The coordinate arrays must be finite, 1-D, and strictly monotonic (either
increasing or decreasing).  Decreasing coordinates are normalised to
increasing order before interpolation, with the corresponding field axes
reversed consistently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Public skip message — kept as a module constant so other modules can reuse it
# ---------------------------------------------------------------------------
_SKIP_MESSAGE = (
    "No reference file found. Skipping reference validation.\n"
    "Provide an NPZ file containing x, z, E_real, and E_imag."
)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class ReferenceField:
    """Structured container for a loaded reference electromagnetic field.

    Attributes
    ----------
    x : np.ndarray
        1-D horizontal coordinate array, shape ``(Nx,)``.
    z : np.ndarray
        1-D vertical coordinate array, shape ``(Nz,)``.
    E_real : np.ndarray
        Real part of the field, shape ``(Nz, Nx)``.
    E_imag : np.ndarray
        Imaginary part of the field, shape ``(Nz, Nx)``.
    E_complex : np.ndarray
        Complex field, shape ``(Nz, Nx)``.
    """

    __slots__ = ("x", "z", "E_real", "E_imag", "E_complex")

    def __init__(
        self,
        x: np.ndarray,
        z: np.ndarray,
        E_real: np.ndarray,
        E_imag: np.ndarray,
    ) -> None:
        self.x = x
        self.z = z
        self.E_real = E_real
        self.E_imag = E_imag
        self.E_complex: np.ndarray = E_real.astype(complex) + 1j * E_imag.astype(complex)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_reference_npz(path: str | Path) -> ReferenceField:
    """Load reference field data from an NPZ file.

    Parameters
    ----------
    path :
        Path to the reference NPZ file.

    Returns
    -------
    ReferenceField
        Validated reference field data container.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    KeyError
        When required array keys are missing.
    ValueError
        When array shapes, dimensions, or values fail validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    data = np.load(path)
    required_keys = ("x", "z", "E_real", "E_imag")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Reference NPZ missing required keys: {missing}")

    x = np.asarray(data["x"], dtype=np.float64)
    z = np.asarray(data["z"], dtype=np.float64)
    E_real = np.asarray(data["E_real"], dtype=np.float64)
    E_imag = np.asarray(data["E_imag"], dtype=np.float64)

    # Dimension checks
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if z.ndim != 1:
        raise ValueError(f"z must be 1-D, got shape {z.shape}")
    if E_real.ndim != 2:
        raise ValueError(f"E_real must be 2-D, got shape {E_real.shape}")
    if E_imag.ndim != 2:
        raise ValueError(f"E_imag must be 2-D, got shape {E_imag.shape}")

    # Shape consistency
    expected_shape = (len(z), len(x))
    if E_real.shape != expected_shape:
        raise ValueError(
            f"E_real shape {E_real.shape} does not match expected (Nz, Nx) = {expected_shape}"
        )
    if E_imag.shape != expected_shape:
        raise ValueError(
            f"E_imag shape {E_imag.shape} does not match expected (Nz, Nx) = {expected_shape}"
        )

    # Finiteness checks
    for name, arr in (("x", x), ("z", z), ("E_real", E_real), ("E_imag", E_imag)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Array '{name}' contains non-finite values (NaN or Inf).")

    # Duplicate coordinate checks
    if len(np.unique(x)) != len(x):
        raise ValueError("Coordinate array 'x' contains duplicate values.")
    if len(np.unique(z)) != len(z):
        raise ValueError("Coordinate array 'z' contains duplicate values.")

    return ReferenceField(x=x, z=z, E_real=E_real, E_imag=E_imag)


# ---------------------------------------------------------------------------
# Orientation normalisation
# ---------------------------------------------------------------------------


def normalize_reference_orientation(reference: ReferenceField) -> ReferenceField:
    """Ensure coordinate arrays are strictly increasing.

    If ``x`` is decreasing, columns of ``E_real`` and ``E_imag`` are reversed
    so that the field data remain consistent with the reordered coordinates.
    If ``z`` is decreasing, rows are reversed analogously.

    Parameters
    ----------
    reference :
        A loaded :class:`ReferenceField` (may have decreasing coordinates).

    Returns
    -------
    ReferenceField
        A new :class:`ReferenceField` with strictly increasing ``x`` and ``z``.
    """
    x = reference.x.copy()
    z = reference.z.copy()
    E_real = reference.E_real.copy()
    E_imag = reference.E_imag.copy()

    if len(x) > 1 and x[-1] < x[0]:
        # Decreasing x → reverse column order
        x = x[::-1].copy()
        E_real = E_real[:, ::-1].copy()
        E_imag = E_imag[:, ::-1].copy()

    if len(z) > 1 and z[-1] < z[0]:
        # Decreasing z → reverse row order
        z = z[::-1].copy()
        E_real = E_real[::-1, :].copy()
        E_imag = E_imag[::-1, :].copy()

    return ReferenceField(x=x, z=z, E_real=E_real, E_imag=E_imag)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate_reference_to_grid(
    reference: ReferenceField,
    x_target: np.ndarray,
    z_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate reference field onto a target grid.

    Real and imaginary parts are interpolated independently using
    ``scipy.interpolate.RegularGridInterpolator``.  Points outside the
    reference domain are returned as ``NaN`` rather than silently extrapolated.

    Parameters
    ----------
    reference :
        A :class:`ReferenceField` with strictly increasing coordinates (call
        :func:`normalize_reference_orientation` first if unsure).
    x_target :
        1-D target x coordinates, shape ``(Nx_target,)``.
    z_target :
        1-D target z coordinates, shape ``(Nz_target,)``.

    Returns
    -------
    E_real_interp : np.ndarray
        Interpolated real field, shape ``(Nz_target, Nx_target)``.
    E_imag_interp : np.ndarray
        Interpolated imaginary field, shape ``(Nz_target, Nx_target)``.
    E_complex_interp : np.ndarray
        Complex combination, shape ``(Nz_target, Nx_target)``.
    """
    x_target = np.asarray(x_target, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    if x_target.ndim != 1:
        raise ValueError(f"x_target must be 1-D, got shape {x_target.shape}")
    if z_target.ndim != 1:
        raise ValueError(f"z_target must be 1-D, got shape {z_target.shape}")

    # Build (z, x) interpolators — RegularGridInterpolator axes are (z, x)
    interp_real = RegularGridInterpolator(
        (reference.z, reference.x),
        reference.E_real,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_imag = RegularGridInterpolator(
        (reference.z, reference.x),
        reference.E_imag,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Build meshgrid with (Nz_target, Nx_target) layout
    X_target, Z_target = np.meshgrid(x_target, z_target)
    query_points = np.stack([Z_target.ravel(), X_target.ravel()], axis=-1)

    E_real_interp = interp_real(query_points).reshape(len(z_target), len(x_target))
    E_imag_interp = interp_imag(query_points).reshape(len(z_target), len(x_target))
    E_complex_interp = E_real_interp.astype(complex) + 1j * E_imag_interp.astype(complex)

    return E_real_interp, E_imag_interp, E_complex_interp


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def relative_l2_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Relative L2 error ``||predicted − reference||_2 / (||reference||_2 + eps)``.

    Parameters
    ----------
    predicted :
        Predicted field array (any shape).
    reference :
        Reference field array (same shape as ``predicted``).
    mask :
        Optional boolean array; only ``True`` positions are included.
        Invalid values (NaN, Inf) are always excluded.
    eps :
        Small constant in denominator to avoid division by zero.

    Returns
    -------
    float
        Relative L2 error.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    valid = np.isfinite(predicted) & np.isfinite(reference)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)

    if not np.any(valid):
        return float("nan")

    diff_norm = np.linalg.norm((predicted - reference)[valid])
    ref_norm = np.linalg.norm(reference[valid])
    return float(diff_norm / (ref_norm + eps))


def maximum_absolute_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Maximum pointwise absolute error.

    Parameters
    ----------
    predicted :
        Predicted field array.
    reference :
        Reference field array (same shape).
    mask :
        Optional boolean mask; invalid values are always excluded.

    Returns
    -------
    float
        Maximum absolute error over valid points.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    valid = np.isfinite(predicted) & np.isfinite(reference)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)

    if not np.any(valid):
        return float("nan")

    return float(np.max(np.abs((predicted - reference)[valid])))


def compute_reference_metrics(
    pinn_E_real: np.ndarray,
    pinn_E_imag: np.ndarray,
    reference_E_real: np.ndarray,
    reference_E_imag: np.ndarray,
) -> dict[str, Any]:
    """Compute comparison metrics between PINN and reference fields.

    Parameters
    ----------
    pinn_E_real, pinn_E_imag :
        Real and imaginary PINN field arrays, both shape ``(Nz, Nx)``.
    reference_E_real, reference_E_imag :
        Interpolated reference field arrays, same shape.

    Returns
    -------
    dict
        JSON-serialisable dictionary with keys:

        * ``relative_l2_error_real`` – relative L2 error of real field
        * ``relative_l2_error_imag`` – relative L2 error of imaginary field
        * ``relative_l2_error_magnitude`` – relative L2 error of field magnitude
        * ``maximum_absolute_error`` – max pointwise |PINN_mag − ref_mag|
        * ``num_valid_points`` – number of points used in comparison
    """
    pinn_complex = pinn_E_real.astype(complex) + 1j * pinn_E_imag.astype(complex)
    ref_complex = reference_E_real.astype(complex) + 1j * reference_E_imag.astype(complex)
    pinn_magnitude = np.abs(pinn_complex)
    ref_magnitude = np.abs(ref_complex)

    # Common finite mask: covers all four arrays + both magnitudes
    valid = (
        np.isfinite(pinn_E_real)
        & np.isfinite(pinn_E_imag)
        & np.isfinite(reference_E_real)
        & np.isfinite(reference_E_imag)
    )

    l2_real = relative_l2_error(pinn_E_real, reference_E_real, mask=valid)
    l2_imag = relative_l2_error(pinn_E_imag, reference_E_imag, mask=valid)
    l2_mag = relative_l2_error(pinn_magnitude, ref_magnitude, mask=valid)
    max_abs = maximum_absolute_error(pinn_magnitude, ref_magnitude, mask=valid)
    num_valid = int(np.sum(valid))

    return {
        "relative_l2_error_real": float(l2_real),
        "relative_l2_error_imag": float(l2_imag),
        "relative_l2_error_magnitude": float(l2_mag),
        "maximum_absolute_error": float(max_abs),
        "num_valid_points": num_valid,
    }


# ---------------------------------------------------------------------------
# Comparison figure  (thin wrapper around visualization)
# ---------------------------------------------------------------------------


def _plot_comparison_figure(
    x: np.ndarray,
    z: np.ndarray,
    ref_magnitude: np.ndarray,
    pinn_magnitude: np.ndarray,
    abs_error: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
    filename: str = "reference_comparison",
) -> plt.Figure:
    """Internal helper; prefer calling :func:`run_comparison` directly."""
    from src.visualization import plot_reference_magnitude_comparison

    return plot_reference_magnitude_comparison(
        x=x,
        z=z,
        ref_magnitude=ref_magnitude,
        pinn_magnitude=pinn_magnitude,
        abs_error=abs_error,
        valid_mask=valid_mask,
        output_dir=output_dir,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def run_comparison(
    reference_path: str | Path,
    pinn_x: np.ndarray,
    pinn_z: np.ndarray,
    pinn_E_real: np.ndarray,
    pinn_E_imag: np.ndarray,
    output_dir: str | Path,
    save_figure: bool = True,
) -> dict[str, Any] | None:
    """Run the full reference-validation pipeline.

    Steps:

    1. Check whether the reference file exists; skip gracefully if not.
    2. Load and validate the reference NPZ.
    3. Normalise coordinate orientation to strictly increasing.
    4. Interpolate the reference onto the PINN evaluation grid.
    5. Compute a finite common mask.
    6. Compute comparison metrics.
    7. Save ``reference_metrics.json``.
    8. Optionally save the comparison figure
       (``reference_comparison.png`` and ``reference_comparison.pdf``).

    Parameters
    ----------
    reference_path :
        Path to the reference NPZ file.
    pinn_x :
        1-D x-coordinate array used by the PINN evaluation grid,
        shape ``(Nx,)``.
    pinn_z :
        1-D z-coordinate array used by the PINN evaluation grid,
        shape ``(Nz,)``.
    pinn_E_real :
        PINN real field on the evaluation grid, shape ``(Nz, Nx)``.
    pinn_E_imag :
        PINN imaginary field on the evaluation grid, shape ``(Nz, Nx)``.
    output_dir :
        Directory for saving ``reference_metrics.json`` and figures.
    save_figure :
        Whether to generate and save the comparison figure.

    Returns
    -------
    dict or None
        Metrics dictionary when comparison succeeds, ``None`` when the
        reference file is absent.
    """
    reference_path = Path(reference_path)
    output_dir = Path(output_dir)

    if not reference_path.exists():
        print(_SKIP_MESSAGE)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalise
    ref = load_reference_npz(reference_path)
    ref = normalize_reference_orientation(ref)

    # pinn_x / pinn_z may be 2-D grids (Nz, Nx); extract 1-D axes
    pinn_x_1d = np.asarray(pinn_x)
    pinn_z_1d = np.asarray(pinn_z)
    if pinn_x_1d.ndim == 2:
        pinn_x_1d = pinn_x_1d[0, :]
    if pinn_z_1d.ndim == 2:
        pinn_z_1d = pinn_z_1d[:, 0]

    # Interpolate onto PINN grid
    ref_E_real, ref_E_imag, _ = interpolate_reference_to_grid(ref, pinn_x_1d, pinn_z_1d)

    # Compute metrics
    metrics = compute_reference_metrics(pinn_E_real, pinn_E_imag, ref_E_real, ref_E_imag)

    # Serialise JSON (all values must be Python scalars)
    metrics_path = output_dir / "reference_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if save_figure:
        pinn_complex = pinn_E_real.astype(complex) + 1j * pinn_E_imag.astype(complex)
        ref_complex = ref_E_real.astype(complex) + 1j * ref_E_imag.astype(complex)
        pinn_magnitude = np.abs(pinn_complex)
        ref_magnitude = np.abs(ref_complex)
        abs_error = np.abs(pinn_magnitude - ref_magnitude)

        # Valid mask: points where both grids are finite
        valid_mask = np.isfinite(pinn_magnitude) & np.isfinite(ref_magnitude)

        _plot_comparison_figure(
            x=pinn_x_1d,
            z=pinn_z_1d,
            ref_magnitude=ref_magnitude,
            pinn_magnitude=pinn_magnitude,
            abs_error=abs_error,
            valid_mask=valid_mask,
            output_dir=output_dir,
        )

    return metrics
