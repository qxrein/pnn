"""Tests for diagnostic and reformulation additions.

Covers:
- Analytical plane-wave field values
- Coordinate nondimensionalisation
- Fourier-feature encoding shape
- Scattered-field source term
- Boundary target generation
- Propagation-sign consistency
- Nondimensional PDE residual
- Model field_components_nd interface
- Layered-medium TMM energy conservation
- Benchmark error metric for identical arrays
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.benchmarks import (
    LayeredMediumBenchmark,
    PlaneWaveBenchmark,
    evaluate_benchmark_errors,
    plane_wave_field,
)
from src.boundary_conditions import top_target
from src.config import ModelConfig, PhysicsConfig
from src.model import FieldMLP, FourierFeatureEncoding
from src.physics import incident_field, scattered_source_term


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_physics() -> PhysicsConfig:
    return PhysicsConfig()


@pytest.fixture
def small_model(default_physics) -> FieldMLP:
    cfg = ModelConfig(hidden_layers=2, hidden_width=16, activation="tanh")
    return FieldMLP(cfg, default_physics)


# ---------------------------------------------------------------------------
# 1. Analytical plane-wave field values
# ---------------------------------------------------------------------------


def test_plane_wave_at_z0():
    """E_inc at z=0 must be 1 + 0i."""
    z = torch.tensor([0.0])
    k0 = 2.0 * math.pi
    e_r, e_i = plane_wave_field(z, n=1.0, k0=k0)
    assert float(e_r) == pytest.approx(1.0, abs=1e-12)
    assert float(e_i) == pytest.approx(0.0, abs=1e-12)


def test_plane_wave_one_wavelength():
    """E_inc at z = lambda = 2pi/k0 must also be 1 + 0i."""
    k0 = 2.0 * math.pi
    lam = 2.0 * math.pi / k0  # = 1.0
    z = torch.tensor([lam])
    e_r, e_i = plane_wave_field(z, n=1.0, k0=k0)
    assert float(e_r) == pytest.approx(1.0, abs=1e-6)
    assert float(e_i) == pytest.approx(0.0, abs=1e-6)


def test_plane_wave_quarter_wavelength():
    """At z = λ/4: Re ≈ 0, Im ≈ -1."""
    k0 = 2.0 * math.pi
    z = torch.tensor([0.25])  # λ/4 = 0.25 when λ=1
    e_r, e_i = plane_wave_field(z, n=1.0, k0=k0)
    assert float(e_r) == pytest.approx(0.0, abs=1e-6)
    assert float(e_i) == pytest.approx(-1.0, abs=1e-6)


def test_plane_wave_magnitude_is_one():
    """For n=1 plane wave, |E| = 1 everywhere."""
    k0 = 2.0 * math.pi
    z = torch.linspace(0, 3.0, 100)
    e_r, e_i = plane_wave_field(z, n=1.0, k0=k0)
    mag = torch.sqrt(e_r**2 + e_i**2)
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Coordinate nondimensionalisation in model
# ---------------------------------------------------------------------------


def test_model_k0_stored(default_physics, small_model):
    """Model stores k0 from physics config."""
    assert small_model._k0 == pytest.approx(default_physics.k0, rel=1e-10)


def test_field_components_nd_output_shape(small_model, default_physics):
    """field_components_nd returns two tensors of shape (N,)."""
    N = 20
    k0 = default_physics.k0
    x = torch.linspace(0.0, 1.0, N) * k0
    z = torch.linspace(0.0, 2.0, N) * k0
    x = x.requires_grad_(True)
    z = z.requires_grad_(True)
    e_r, e_i = small_model.field_components_nd(x, z)
    assert e_r.shape == (N,)
    assert e_i.shape == (N,)


def test_forward_vs_field_components_nd_consistency(small_model, default_physics):
    """forward(x,z) and field_components_nd(k0*x, k0*z) give same values."""
    N = 10
    k0 = default_physics.k0
    x_phys = torch.linspace(0.05, 0.95, N)
    z_phys = torch.linspace(0.1, 1.9, N)
    out_fwd = small_model.forward(x_phys, z_phys)
    x_nd = x_phys * k0
    z_nd = z_phys * k0
    e_r, e_i = small_model.field_components_nd(x_nd, z_nd)
    torch.testing.assert_close(out_fwd[:, 0], e_r, atol=1e-6, rtol=0)
    torch.testing.assert_close(out_fwd[:, 1], e_i, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# 3. Fourier feature encoding shape
# ---------------------------------------------------------------------------


def test_fourier_feature_encoding_shape():
    """FourierFeatureEncoding output shape is (N, 4*num_levels)."""
    enc = FourierFeatureEncoding(num_levels=4)
    N = 32
    x = torch.randn(N)
    z = torch.randn(N)
    out = enc(x, z)
    assert out.shape == (N, 16)  # 4 * 4 = 16


def test_model_with_fourier_features(default_physics):
    """Model with Fourier features runs forward pass without error."""
    cfg = ModelConfig(
        hidden_layers=2,
        hidden_width=16,
        activation="tanh",
        fourier_features=True,
        num_fourier_features=16,  # => 4 levels => 16-dim input
    )
    model = FieldMLP(cfg, default_physics)
    x = torch.linspace(0.0, 1.0, 8)
    z = torch.linspace(0.0, 2.0, 8)
    out = model(x, z)
    assert out.shape == (8, 2)


# ---------------------------------------------------------------------------
# 4. Scattered-field source term
# ---------------------------------------------------------------------------


def test_scattered_source_in_air_is_zero(default_physics):
    """Source term is zero in uniform air (εr = 1 → δεr = 0)."""
    # Air region: x in middle, z well above ridge
    x = torch.full((10,), 0.5)
    z = torch.linspace(0.1, 1.1, 10)  # all above ridge_z_min=1.2
    src_r, src_i = scattered_source_term(x, z, default_physics)
    assert torch.allclose(src_r, torch.zeros_like(src_r), atol=1e-10)
    assert torch.allclose(src_i, torch.zeros_like(src_i), atol=1e-10)


def test_scattered_source_nonzero_in_ridge(default_physics):
    """Source term is nonzero inside the dielectric ridge (εr > 1)."""
    p = default_physics
    x = torch.full((5,), p.period / 2.0)
    z = torch.full((5,), p.ridge_z_min + p.ridge_height / 2.0)
    src_r, src_i = scattered_source_term(x, z, p)
    # δεr = n_ridge² - 1 > 0
    assert torch.any(torch.abs(src_r) > 1e-6) or torch.any(torch.abs(src_i) > 1e-6)


# ---------------------------------------------------------------------------
# 5. Boundary target generation
# ---------------------------------------------------------------------------


def test_top_target_at_z0(default_physics):
    """top_target at z=0 returns (1, 0)."""
    z = torch.zeros(10)
    t_r, t_i = top_target(z, default_physics)
    assert torch.allclose(t_r, torch.ones(10), atol=1e-10)
    assert torch.allclose(t_i, torch.zeros(10), atol=1e-10)


def test_top_target_matches_incident_field(default_physics):
    """top_target == incident_field for all z values."""
    z = torch.linspace(0, 2.0, 50)
    tr, ti = top_target(z, default_physics)
    ir, ii = incident_field(z, default_physics)
    torch.testing.assert_close(tr, ir)
    torch.testing.assert_close(ti, ii)


# ---------------------------------------------------------------------------
# 6. Propagation-sign consistency
# ---------------------------------------------------------------------------


def test_incident_field_sign_convention(default_physics):
    """Verify E_inc = exp(-i k0 z): at z=λ/4, Re≈0 and Im≈-1."""
    z = torch.tensor([0.25])  # z = λ/4 when λ=1
    k0 = default_physics.k0  # 2π
    e_r, e_i = incident_field(z, default_physics)
    assert float(e_r) == pytest.approx(math.cos(k0 * 0.25), abs=1e-6)
    assert float(e_i) == pytest.approx(-math.sin(k0 * 0.25), abs=1e-6)


def test_incident_field_is_unit_amplitude():
    """Incident field has unit amplitude for all z."""
    physics = PhysicsConfig()
    z = torch.linspace(0, 2.0, 100)
    e_r, e_i = incident_field(z, physics)
    mag = torch.sqrt(e_r**2 + e_i**2)
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Nondimensional PDE residual
# ---------------------------------------------------------------------------


def test_pde_residual_for_exact_solution(default_physics):
    """PDE residual is near zero for the exact plane-wave solution (air, εr=1).

    Tests the nondimensional form: ∂²E/∂z̃² + ε E = 0  (∂²/∂x̃² = 0 for plane wave).
    The model adds x_tilde * 0 to keep x in the graph; autograd must not error.
    """
    from src.physics import helmholtz_residual

    k0 = default_physics.k0

    class ExactPlaneWave(torch.nn.Module):
        """E(x̃, z̃) = exp(-i z̃) — trivially x-independent."""

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def field_components_nd(self, x_t, z_t):
            # Multiply by (1 + 0*x_t) so x_t is in the computation graph
            ones_x = 1.0 + x_t * 0.0 + self.dummy * 0.0
            return torch.cos(z_t) * ones_x, -torch.sin(z_t) * ones_x

        def field_components(self, x, z):
            phase = k0 * z
            return torch.cos(phase), -torch.sin(phase)

        def forward(self, x, z):
            e_r, e_i = self.field_components(x, z)
            return torch.stack([e_r, e_i], dim=-1)

    model = ExactPlaneWave()
    # Points in air region only (εr = 1) — avoid ridge/substrate for clean test
    x = torch.linspace(0.1, 0.9, 20)
    z = torch.linspace(0.05, 1.15, 20)   # all above ridge_z_min=1.2
    res_r, res_i = helmholtz_residual(model, x, z, default_physics)
    assert torch.max(torch.abs(res_r)).item() < 1e-4
    assert torch.max(torch.abs(res_i)).item() < 1e-4


# ---------------------------------------------------------------------------
# 8. Layered-medium TMM energy conservation
# ---------------------------------------------------------------------------


def test_layered_tmm_energy_conservation():
    """Energy conservation: |r|² + |t|²*(n_sub/n_air) = 1."""
    bm = LayeredMediumBenchmark(k0=2.0 * math.pi)
    r = bm.reflection_coefficient()
    t = bm.transmission_coefficient()
    energy = abs(r)**2 + abs(t)**2 * (bm.n_sub / bm.n_air)
    assert energy == pytest.approx(1.0, abs=1e-6)


def test_layered_analytical_field_continuity():
    """Analytical field should be continuous across slab interfaces."""
    bm = LayeredMediumBenchmark(k0=2.0 * math.pi)
    dz = 1e-6
    z_at_top = np.array([bm.z_slab_top - dz, bm.z_slab_top + dz])
    z_at_bot = np.array([bm.z_slab_bot - dz, bm.z_slab_bot + dz])
    x = np.zeros(2)

    er_top, ei_top = bm.analytical_field_np(x, z_at_top)
    assert abs(er_top[0] - er_top[1]) < 0.01  # continuous Re at top interface
    assert abs(ei_top[0] - ei_top[1]) < 0.01  # continuous Im at top interface

    er_bot, ei_bot = bm.analytical_field_np(x, z_at_bot)
    assert abs(er_bot[0] - er_bot[1]) < 0.01
    assert abs(ei_bot[0] - ei_bot[1]) < 0.01


# ---------------------------------------------------------------------------
# 9. Benchmark error metric
# ---------------------------------------------------------------------------


def test_evaluate_benchmark_errors_identical():
    """Error for identical arrays should be zero."""
    arr = np.ones((32, 32))
    errors = evaluate_benchmark_errors(arr, arr, arr, arr)
    assert errors["relative_l2_real"] == pytest.approx(0.0, abs=1e-12)
    assert errors["relative_l2_magnitude"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_benchmark_errors_known():
    """Error for scaled arrays should be > 0."""
    ref = np.ones(10)
    pred = 2.0 * ref
    errors = evaluate_benchmark_errors(pred, ref * 0, ref, ref * 0)
    assert errors["relative_l2_real"] > 0.5
