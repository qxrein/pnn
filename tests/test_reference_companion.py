"""Tests for the solver-side RCWA companion export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import PhysicsConfig, load_config
from src.reference_companion import (
    COMPANION_FILENAME,
    build_companion_arrays,
    sha256_file,
    write_canonical_snapshot_for_tests,
    write_companion,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def physics_small():
    cfg = load_config(ROOT / "configs/default.yaml")
    p = cfg.physics
    return PhysicsConfig(
        wavelength=p.wavelength,
        n_air=p.n_air,
        n_ridge=p.n_ridge,
        n_substrate=p.n_substrate,
        period=0.8 * p.wavelength,
        ridge_width=0.4 * 0.8 * p.wavelength,
        ridge_height=p.ridge_height,
        domain_height=p.domain_height,
        ridge_base_fraction=p.ridge_base_fraction,
        nx_visualization=32,
        nz_visualization=64,
    )


@pytest.fixture
def canonical_and_companion(tmp_path, physics_small):
    can = write_canonical_snapshot_for_tests(
        tmp_path / "canonical.npz", physics_small, n_harmonics=7, nfine=256,
    )
    sha_before = sha256_file(can)
    mtime_before = can.stat().st_mtime_ns
    size_before = can.stat().st_size
    out = tmp_path / "reference_companion"
    manifest = write_companion(can, out, nfine=256)
    return {
        "canonical": can,
        "manifest": manifest,
        "sha_before": sha_before,
        "mtime_before": mtime_before,
        "size_before": size_before,
        "companion": Path(manifest["companion_path"]),
        "physics": physics_small,
    }


def test_canonical_hash_agreement(canonical_and_companion):
    info = canonical_and_companion
    assert info["manifest"]["canonical_sha256"] == info["sha_before"]
    with np.load(info["companion"], allow_pickle=False) as data:
        assert str(data["canonical_npz_sha256"].item()) == info["sha_before"]


def test_grid_and_coordinate_agreement(canonical_and_companion):
    info = canonical_and_companion
    with np.load(info["canonical"], allow_pickle=False) as can, np.load(
        info["companion"], allow_pickle=False
    ) as cmp:
        np.testing.assert_array_equal(cmp["x"], can["x"])
        np.testing.assert_array_equal(cmp["z"], can["z"])


def test_region_mask_agreement(canonical_and_companion):
    info = canonical_and_companion
    physics = info["physics"]
    with np.load(info["companion"], allow_pickle=False) as cmp:
        x, z = cmp["x"], cmp["z"]
        X, Z = np.meshgrid(x, z)
        expected = np.zeros(Z.shape, dtype=np.int8)
        expected[Z >= physics.ridge_base_z] = 2
        in_ridge = (
            (X >= physics.ridge_x_min)
            & (X <= physics.ridge_x_max)
            & (Z >= physics.ridge_z_min)
            & (Z <= physics.ridge_z_max)
        )
        expected[in_ridge] = 1
        np.testing.assert_array_equal(cmp["region_id"], expected)
        assert np.all(cmp["epsilon_r"][expected == 0] == physics.eps_air)
        assert np.all(cmp["epsilon_r"][expected == 1] == physics.eps_ridge)
        assert np.all(cmp["epsilon_r"][expected == 2] == physics.eps_substrate)


def test_boundary_coefficient_agreement(canonical_and_companion):
    info = canonical_and_companion
    with np.load(info["canonical"], allow_pickle=False) as can, np.load(
        info["companion"], allow_pickle=False
    ) as cmp:
        np.testing.assert_allclose(cmp["c_refl"], can["c_refl"], atol=1e-14)
        np.testing.assert_allclose(cmp["c_trans"], can["c_trans"], atol=1e-14)
        np.testing.assert_allclose(cmp["kx"], can["kx"], atol=1e-14)
        np.testing.assert_allclose(cmp["kz_air"], can["kz_air"], atol=1e-14)
        np.testing.assert_allclose(cmp["kz_substrate"], can["kz_sub"], atol=1e-14)


def test_field_reconstruction_away_from_interfaces(canonical_and_companion):
    info = canonical_and_companion
    assert info["manifest"]["max_reconstruction_error"] < 1e-10
    with np.load(info["canonical"], allow_pickle=False) as can, np.load(
        info["companion"], allow_pickle=False
    ) as cmp:
        ey_can = can["E_real"] + 1j * can["E_imag"]
        np.testing.assert_allclose(cmp["Ey_total"], ey_can, atol=1e-12)


def test_one_sided_interface_array_shapes(canonical_and_companion):
    info = canonical_and_companion
    required = {
        "ridge_left_edge",
        "ridge_right_edge",
        "ridge_top",
        "ridge_bottom",
        "air_substrate_left",
        "air_substrate_right",
        "upper_triple_junction_left",
        "upper_triple_junction_right",
    }
    with np.load(info["companion"], allow_pickle=False) as cmp:
        names = cmp["interface_name"].astype(str)
        assert required <= set(names)
        n = names.shape[0]
        for key in (
            "interface_x", "interface_z",
            "Ey_minus", "Hx_minus", "Hz_minus",
            "Ey_plus", "Hx_plus", "Hz_plus",
        ):
            assert cmp[key].shape[0] == n
        assert cmp["interface_normal"].shape == (n, 2)
        for name in required:
            count = int(np.count_nonzero(names == name))
            assert count >= 1
            if name.startswith("upper_triple"):
                assert count == 1


def test_no_accidental_canonical_overwrite(canonical_and_companion, tmp_path, physics_small):
    info = canonical_and_companion
    can = info["canonical"]
    assert sha256_file(can) == info["sha_before"]
    assert can.stat().st_size == info["size_before"]
    assert can.stat().st_mtime_ns == info["mtime_before"]

    other = write_canonical_snapshot_for_tests(
        tmp_path / "other_canonical.npz", physics_small, n_harmonics=5, nfine=128,
    )
    dest_dir = tmp_path / "reference_companion"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_companion(other, dest_dir, nfine=128, filename=COMPANION_FILENAME)


def test_validation_gates_pass(canonical_and_companion):
    gates = canonical_and_companion["manifest"]["gates"]
    assert gates["A_ey_reconstruction"]
    assert gates["B_boundary_coefficients"]
    assert gates["C_homogeneous_residual"]
    assert gates["D_vertical_Ey_Hz"]
    assert gates["E_horizontal_Ey_Hx"]


def test_build_does_not_open_canonical_for_write(tmp_path, physics_small, monkeypatch):
    can = write_canonical_snapshot_for_tests(
        tmp_path / "canonical.npz", physics_small, n_harmonics=5, nfine=128,
    )

    def _blocked_savez(path, *args, **kwargs):
        if Path(path).resolve() == can.resolve():
            raise AssertionError("canonical NPZ was opened for write")
        return np.savez(path, *args, **kwargs)

    monkeypatch.setattr(np, "savez", _blocked_savez)
    build_companion_arrays(can, nfine=128)
