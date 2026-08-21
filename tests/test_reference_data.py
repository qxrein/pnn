"""Tests for src/reference_data.py — reference-validation pipeline.

All tests use analytic data or controlled synthetic arrays.
No production reference data are generated or assumed to exist.

Analytic test field
-------------------
For interpolation accuracy tests we use the bilinear field::

    E_real(z, x) = 2*x + 3*z
    E_imag(z, x) = x - z

Linear interpolation is exact for bilinear functions, so equality assertions
are appropriate (within floating-point round-off).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.reference_data import (
    ReferenceField,
    _SKIP_MESSAGE,
    compute_reference_metrics,
    interpolate_reference_to_grid,
    load_reference_npz,
    maximum_absolute_error,
    normalize_reference_orientation,
    relative_l2_error,
    run_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analytic_reference(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic (E_real, E_imag) on a meshgrid for given 1-D axes.

    E_real(z, x) = 2x + 3z
    E_imag(z, x) = x − z
    """
    X, Z = np.meshgrid(x, z)
    E_real = 2.0 * X + 3.0 * Z
    E_imag = X - Z
    return E_real, E_imag


def _write_npz(path: Path, x, z, E_real, E_imag) -> None:
    np.savez(path, x=x, z=z, E_real=E_real, E_imag=E_imag)


def _valid_reference(
    nx: int = 20, nz: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    z = np.linspace(0.0, 2.0, nz)
    E_real, E_imag = _make_analytic_reference(x, z)
    return x, z, E_real, E_imag


# ---------------------------------------------------------------------------
# 1. Successful NPZ loading
# ---------------------------------------------------------------------------


def test_load_reference_npz_success(tmp_path: Path) -> None:
    x, z, E_real, E_imag = _valid_reference()
    npz_path = tmp_path / "ref.npz"
    _write_npz(npz_path, x, z, E_real, E_imag)

    ref = load_reference_npz(npz_path)
    assert isinstance(ref, ReferenceField)
    assert ref.x.shape == (len(x),)
    assert ref.z.shape == (len(z),)
    assert ref.E_real.shape == (len(z), len(x))
    assert ref.E_imag.shape == (len(z), len(x))
    assert ref.E_complex.dtype == complex


# ---------------------------------------------------------------------------
# 2. Missing required key
# ---------------------------------------------------------------------------


def test_load_reference_npz_missing_key(tmp_path: Path) -> None:
    x, z, E_real, _ = _valid_reference()
    npz_path = tmp_path / "ref_missing.npz"
    # Deliberately omit E_imag
    np.savez(npz_path, x=x, z=z, E_real=E_real)

    with pytest.raises(KeyError, match="E_imag"):
        load_reference_npz(npz_path)


# ---------------------------------------------------------------------------
# 3. Incorrect coordinate dimensions (2-D x)
# ---------------------------------------------------------------------------


def test_load_reference_npz_bad_x_dim(tmp_path: Path) -> None:
    x, z, E_real, E_imag = _valid_reference()
    npz_path = tmp_path / "ref_bad_x.npz"
    X2d = np.tile(x, (3, 1))  # 2-D — not allowed
    np.savez(npz_path, x=X2d, z=z, E_real=E_real, E_imag=E_imag)

    with pytest.raises(ValueError, match="x must be 1-D"):
        load_reference_npz(npz_path)


# ---------------------------------------------------------------------------
# 4. Incorrect field shape
# ---------------------------------------------------------------------------


def test_load_reference_npz_bad_field_shape(tmp_path: Path) -> None:
    x, z, E_real, E_imag = _valid_reference()
    npz_path = tmp_path / "ref_bad_shape.npz"
    # Transpose E_real so it has shape (Nx, Nz) instead of (Nz, Nx)
    np.savez(npz_path, x=x, z=z, E_real=E_real.T, E_imag=E_imag)

    with pytest.raises(ValueError, match="E_real shape"):
        load_reference_npz(npz_path)


# ---------------------------------------------------------------------------
# 5. Non-finite values in E_real
# ---------------------------------------------------------------------------


def test_load_reference_npz_nonfinite(tmp_path: Path) -> None:
    x, z, E_real, E_imag = _valid_reference()
    E_real[0, 0] = np.nan
    npz_path = tmp_path / "ref_nan.npz"
    _write_npz(npz_path, x, z, E_real, E_imag)

    with pytest.raises(ValueError, match="non-finite"):
        load_reference_npz(npz_path)


# ---------------------------------------------------------------------------
# 6. Decreasing x normalisation
# ---------------------------------------------------------------------------


def test_normalize_decreasing_x() -> None:
    x_dec = np.array([1.0, 0.75, 0.5, 0.25, 0.0])  # decreasing
    z = np.linspace(0.0, 1.0, 4)
    E_real, E_imag = _make_analytic_reference(x_dec[::-1], z)
    # Build a reference as if loaded with decreasing x
    ref_dec = ReferenceField(x=x_dec, z=z, E_real=E_real[:, ::-1], E_imag=E_imag[:, ::-1])
    ref_norm = normalize_reference_orientation(ref_dec)

    assert ref_norm.x[0] < ref_norm.x[-1], "x should be increasing after normalisation"
    np.testing.assert_array_almost_equal(ref_norm.x, x_dec[::-1])


# ---------------------------------------------------------------------------
# 7. Decreasing z normalisation
# ---------------------------------------------------------------------------


def test_normalize_decreasing_z() -> None:
    x = np.linspace(0.0, 1.0, 5)
    z_dec = np.array([2.0, 1.5, 1.0, 0.5, 0.0])  # decreasing
    # Build reference with decreasing z; build field accordingly
    E_real, E_imag = _make_analytic_reference(x, z_dec[::-1])
    ref_dec = ReferenceField(x=x, z=z_dec, E_real=E_real[::-1, :], E_imag=E_imag[::-1, :])
    ref_norm = normalize_reference_orientation(ref_dec)

    assert ref_norm.z[0] < ref_norm.z[-1], "z should be increasing after normalisation"
    np.testing.assert_array_almost_equal(ref_norm.z, z_dec[::-1])


# ---------------------------------------------------------------------------
# 8. Interpolation onto a different target grid (shape check)
# ---------------------------------------------------------------------------


def test_interpolation_output_shapes() -> None:
    x, z, E_real, E_imag = _valid_reference(nx=20, nz=15)
    ref = ReferenceField(x=x, z=z, E_real=E_real, E_imag=E_imag)

    x_target = np.linspace(0.1, 0.9, 12)
    z_target = np.linspace(0.2, 1.8, 18)
    E_r, E_i, E_c = interpolate_reference_to_grid(ref, x_target, z_target)

    assert E_r.shape == (len(z_target), len(x_target))
    assert E_i.shape == (len(z_target), len(x_target))
    assert E_c.shape == (len(z_target), len(x_target))
    assert E_c.dtype == complex


# ---------------------------------------------------------------------------
# 9. Known analytic interpolation case (linear ⟹ exact reproduction)
# ---------------------------------------------------------------------------


def test_interpolation_analytic_accuracy() -> None:
    """Linear interpolation must reproduce a bilinear field exactly."""
    x_src = np.linspace(0.0, 1.0, 30)
    z_src = np.linspace(0.0, 2.0, 30)
    E_real_src, E_imag_src = _make_analytic_reference(x_src, z_src)
    ref = ReferenceField(x=x_src, z=z_src, E_real=E_real_src, E_imag=E_imag_src)

    # Dense target grid strictly inside the source domain
    x_tgt = np.linspace(0.05, 0.95, 50)
    z_tgt = np.linspace(0.1, 1.9, 60)
    E_r_interp, E_i_interp, _ = interpolate_reference_to_grid(ref, x_tgt, z_tgt)

    E_r_expected, E_i_expected = _make_analytic_reference(x_tgt, z_tgt)
    np.testing.assert_allclose(E_r_interp, E_r_expected, atol=1e-10)
    np.testing.assert_allclose(E_i_interp, E_i_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Relative L2 error is exactly zero for identical arrays
# ---------------------------------------------------------------------------


def test_relative_l2_error_identical() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert relative_l2_error(arr, arr) == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# 11. Relative L2 error for known arrays
# ---------------------------------------------------------------------------


def test_relative_l2_error_known() -> None:
    # predicted = reference * 2  →  ||pred - ref|| / ||ref|| = 1.0  (eps negligible)
    ref = np.array([1.0, 2.0, 3.0])
    pred = 2.0 * ref
    eps = 1e-12
    expected = np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps)
    assert relative_l2_error(pred, ref, eps=eps) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# 12. Maximum absolute error for known arrays
# ---------------------------------------------------------------------------


def test_maximum_absolute_error_known() -> None:
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.5, 1.0, 1.5, 3.0])
    # differences: 0.5, 0.0, 0.5, 0.0  →  max = 0.5
    assert maximum_absolute_error(pred, ref) == pytest.approx(0.5, abs=1e-14)


# ---------------------------------------------------------------------------
# 13. NaN masking in error functions
# ---------------------------------------------------------------------------


def test_relative_l2_nan_masking() -> None:
    ref = np.array([1.0, 2.0, np.nan])
    pred = np.array([1.0, 2.0, 999.0])
    # NaN in ref → should not contribute; perfect agreement on valid points
    assert relative_l2_error(pred, ref) == pytest.approx(0.0, abs=1e-14)


def test_maximum_absolute_error_nan_masking() -> None:
    ref = np.array([1.0, np.nan, 3.0])
    pred = np.array([1.5, 999.0, 3.0])
    # Only positions 0 and 2 are valid → max |diff| = 0.5
    assert maximum_absolute_error(pred, ref) == pytest.approx(0.5, abs=1e-14)


# ---------------------------------------------------------------------------
# 14. JSON serialisation of metrics
# ---------------------------------------------------------------------------


def test_compute_reference_metrics_json_serialisable(tmp_path: Path) -> None:
    x, z, E_real, E_imag = _valid_reference()
    metrics = compute_reference_metrics(E_real, E_imag, E_real, E_imag)

    # Must be serialisable without raising
    json_str = json.dumps(metrics)
    loaded = json.loads(json_str)

    assert set(loaded.keys()) == {
        "relative_l2_error_real",
        "relative_l2_error_imag",
        "relative_l2_error_magnitude",
        "maximum_absolute_error",
        "num_valid_points",
    }
    # Identical arrays → L2 errors are zero
    assert loaded["relative_l2_error_real"] == pytest.approx(0.0, abs=1e-14)
    assert loaded["relative_l2_error_imag"] == pytest.approx(0.0, abs=1e-14)
    assert loaded["relative_l2_error_magnitude"] == pytest.approx(0.0, abs=1e-14)
    assert loaded["maximum_absolute_error"] == pytest.approx(0.0, abs=1e-14)
    assert isinstance(loaded["num_valid_points"], int)
    assert loaded["num_valid_points"] > 0


# ---------------------------------------------------------------------------
# 15. Missing reference file returns None without raising inside run_comparison
# ---------------------------------------------------------------------------


def test_run_comparison_missing_file_returns_none(tmp_path: Path, capsys) -> None:
    x, z, E_real, E_imag = _valid_reference()
    missing = tmp_path / "does_not_exist.npz"

    result = run_comparison(
        reference_path=missing,
        pinn_x=x,
        pinn_z=z,
        pinn_E_real=E_real,
        pinn_E_imag=E_imag,
        output_dir=tmp_path / "out",
        save_figure=False,
    )

    assert result is None
    captured = capsys.readouterr()
    assert "No reference file found" in captured.out


# ---------------------------------------------------------------------------
# 16. Comparison figure and JSON are created for a valid reference file
# ---------------------------------------------------------------------------


def test_run_comparison_creates_outputs(tmp_path: Path) -> None:
    x_src = np.linspace(0.0, 1.0, 20)
    z_src = np.linspace(0.0, 2.0, 15)
    E_real_src, E_imag_src = _make_analytic_reference(x_src, z_src)
    npz_path = tmp_path / "reference.npz"
    _write_npz(npz_path, x_src, z_src, E_real_src, E_imag_src)

    # PINN grid (slightly different resolution)
    x_pinn = np.linspace(0.0, 1.0, 16)
    z_pinn = np.linspace(0.0, 2.0, 12)
    E_real_pinn, E_imag_pinn = _make_analytic_reference(x_pinn, z_pinn)

    out_dir = tmp_path / "outputs"
    metrics = run_comparison(
        reference_path=npz_path,
        pinn_x=x_pinn,
        pinn_z=z_pinn,
        pinn_E_real=E_real_pinn,
        pinn_E_imag=E_imag_pinn,
        output_dir=out_dir,
        save_figure=True,
    )

    assert metrics is not None
    assert (out_dir / "reference_metrics.json").exists()
    assert (out_dir / "reference_comparison.png").exists()
    assert (out_dir / "reference_comparison.pdf").exists()

    # Verify JSON content
    with (out_dir / "reference_metrics.json").open() as fh:
        loaded = json.load(fh)
    assert "relative_l2_error_magnitude" in loaded
    assert isinstance(loaded["num_valid_points"], int)
    # Same analytic field → errors should be small (linear interp on linear field)
    assert loaded["relative_l2_error_magnitude"] < 1e-6


# ---------------------------------------------------------------------------
# 17. FileNotFoundError is raised when load_reference_npz is called directly
#     with a missing path
# ---------------------------------------------------------------------------


def test_load_reference_npz_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_reference_npz("/nonexistent/path/to/ref.npz")


# ---------------------------------------------------------------------------
# 18. compute_reference_metrics returns expected key set and correct types
# ---------------------------------------------------------------------------


def test_compute_reference_metrics_key_set() -> None:
    x, z, E_real, E_imag = _valid_reference(nx=8, nz=6)
    E_pred_r = E_real + 0.1  # small uniform offset
    E_pred_i = E_imag - 0.05

    metrics = compute_reference_metrics(E_pred_r, E_pred_i, E_real, E_imag)
    assert set(metrics.keys()) == {
        "relative_l2_error_real",
        "relative_l2_error_imag",
        "relative_l2_error_magnitude",
        "maximum_absolute_error",
        "num_valid_points",
    }
    assert isinstance(metrics["relative_l2_error_real"], float)
    assert isinstance(metrics["num_valid_points"], int)
    assert metrics["relative_l2_error_real"] > 0.0
