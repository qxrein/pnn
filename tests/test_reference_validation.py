"""Canonical-reference metadata and stale-reference guards."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from scripts.train_lbg import make_lambda_0p8
from src.config import PhysicsConfig
from src.reference_validation import validate_reference


def _write_reference(path: Path, p: PhysicsConfig) -> None:
    np.savez(path, field_representation="total", wavelength=p.wavelength, period=p.period,
             ridge_width=p.ridge_width, ridge_height=p.ridge_height, n_ridge=p.n_ridge,
             n_substrate=p.n_substrate,
             geometry_convention="dielectric ridge on substrate; substrate outside ridge footprint",
             c_refl=np.zeros(3, complex), c_trans=np.zeros(3, complex),
             R_m=np.zeros(3), T_m=np.zeros(3), R_total=0.1, T_total=0.9,
             z_top_monitor=.08*p.domain_height, z_bot_monitor=.92*p.domain_height)


def test_reference_metadata_must_match_pinn_geometry(tmp_path):
    p = PhysicsConfig(period=.8, ridge_width=.32)
    path = tmp_path / "geometry_consistent.npz"; _write_reference(path, p)
    result = validate_reference(path, p)
    assert result["field_representation"] == "total"
    changed = PhysicsConfig(period=.8, ridge_width=.31)
    with pytest.raises(ValueError, match="ridge_width"):
        validate_reference(path, changed)


def test_old_lambda_reference_is_explicitly_rejected(tmp_path):
    p = PhysicsConfig(period=.8, ridge_width=.32)
    path = tmp_path / "reference_lambda_0p8.npz"; _write_reference(path, p)
    with pytest.raises(ValueError, match="stale"):
        validate_reference(path, p)
