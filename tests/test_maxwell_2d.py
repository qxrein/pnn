"""Tests for src/maxwell_2d.py — 2-D Maxwell PINN for the grating.

Tests:
1.  Maxwell2DMLP output shape (N, 6).
2.  PDE residual is zero for an exact plane-wave solution in air.
3.  Top BC loss is zero when field matches the incident wave.
4.  Bottom BC loss is zero when H = n_sub * E.
5.  Periodic BC loss is zero when left == right.
6.  Sample_collocation_points returns correct shapes.
7.  Interior points exclude interface margins.
8.  Total loss returns all expected keys.
9.  Incident plane-wave satisfies all six Maxwell equations analytically.
10. Sign convention: H̃_x = +n_air*E for forward plane wave.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.config import PhysicsConfig
from src.maxwell_2d import (
    Maxwell2DMLP,
    incident_E_H,
    maxwell_2d_bottom_bc_loss,
    maxwell_2d_pde_residual,
    maxwell_2d_periodic_bc_loss,
    maxwell_2d_top_bc_loss,
    maxwell_2d_total_loss,
    sample_collocation_points,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def physics() -> PhysicsConfig:
    return PhysicsConfig()


@pytest.fixture
def dtype():
    return torch.float64


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_model(physics, dtype) -> Maxwell2DMLP:
    return Maxwell2DMLP(physics, hidden_layers=2, hidden_width=16,
                        num_fourier_levels=2).to(dtype=dtype)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------


def test_model_output_shape(physics, small_model, dtype, device):
    N = 32
    x = torch.linspace(0.0, physics.period, N, dtype=dtype, device=device)
    z = torch.linspace(0.0, physics.domain_height, N, dtype=dtype, device=device)
    out = small_model.forward(x, z)
    assert out.shape == (N, 6)


def test_field_components_count(physics, small_model, dtype, device):
    N = 10
    x = torch.ones(N, dtype=dtype) * 0.5
    z = torch.linspace(0.1, 1.9, N, dtype=dtype)
    comps = small_model.field_components(x, z)
    assert len(comps) == 6
    for c in comps:
        assert c.shape == (N,)


# ---------------------------------------------------------------------------
# 2. PDE residual is zero for exact plane-wave in air (εr=1)
# ---------------------------------------------------------------------------


def test_pde_residual_exact_plane_wave(physics, dtype, device):
    """For a z-propagating plane wave in air (εr=1), all 6 residuals = 0.

    The analytic solution:
        Er = cos(k0*z), Ei = -sin(k0*z)
        Hr_x = n_air * Er = cos(k0*z),  Hi_x = n_air * Ei = -sin(k0*z)
        Hr_z = 0,  Hi_z = 0   (no x-variation)
    """
    k0    = physics.k0
    n_air = physics.n_air

    class ExactPlaneWave(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, z):
            Er  =  torch.cos(k0 * z) + self.dummy * 0 + x * 0
            Ei  = -torch.sin(k0 * z) + self.dummy * 0 + x * 0
            Hrx =  n_air * torch.cos(k0 * z) + self.dummy * 0 + x * 0
            Hix = -n_air * torch.sin(k0 * z) + self.dummy * 0 + x * 0
            Hrz = torch.zeros_like(z) + self.dummy * 0 + x * 0
            Hiz = torch.zeros_like(z) + self.dummy * 0 + x * 0
            return torch.stack([Er, Ei, Hrx, Hix, Hrz, Hiz], dim=-1)

        def field_components(self, x, z):
            out = self.forward(x, z)
            return tuple(out[:, i] for i in range(6))

    # Use an air-only physics config (no grating) to avoid εr > 1 in interior
    air_physics = PhysicsConfig(
        wavelength=1.0, n_air=1.0, n_ridge=1.0, n_substrate=1.0,
        period=1.0, ridge_width=0.0, ridge_height=0.0,
        domain_height=2.0, ridge_base_fraction=1.0,
    )
    model = ExactPlaneWave()
    # Points well inside air (z < ridge_z_min = domain_height since ridge_base_fraction=1)
    N = 20
    x = torch.linspace(0.1, 0.9, N, dtype=dtype)
    z = torch.linspace(0.1, 1.9, N, dtype=dtype)
    residuals = maxwell_2d_pde_residual(model, x, z, air_physics)
    for i, res in enumerate(residuals):
        mse = float(torch.mean(res**2).detach())
        assert mse < 1e-8, f"Residual {i} MSE = {mse:.3e} for exact plane wave"


# ---------------------------------------------------------------------------
# 3. Top BC loss is zero when field = incident wave
# ---------------------------------------------------------------------------


def test_top_bc_zero_for_incident_field(physics, dtype, device):
    """Top BC loss should be zero if E=(1,0) and H=(n_air,0) at z=0."""
    k0    = physics.k0
    n_air = physics.n_air

    class IncidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, z):
            ones = torch.ones_like(z) + self.dummy * 0 + x * 0
            zeros = torch.zeros_like(z) + self.dummy * 0 + x * 0
            n = ones * n_air
            return torch.stack([ones, zeros, n, zeros, zeros, zeros], dim=-1)

        def field_components(self, x, z):
            out = self.forward(x, z)
            return tuple(out[:, i] for i in range(6))

    model = IncidentModel()
    N = 32
    x = torch.linspace(0.0, physics.period, N, dtype=dtype)
    z = torch.zeros(N, dtype=dtype)
    loss = maxwell_2d_top_bc_loss(model, x, z, physics)
    assert float(loss.detach()) < 1e-20


# ---------------------------------------------------------------------------
# 4. Bottom BC loss is zero when H = n_sub * E
# ---------------------------------------------------------------------------


def test_bottom_bc_zero_for_outgoing(physics, dtype, device):
    n_sub = physics.n_substrate

    class OutgoingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, z):
            # Arbitrary E, with H = n_sub * E
            k0 = physics.k0
            Er = torch.cos(k0 * z) + self.dummy * 0 + x * 0
            Ei = -torch.sin(k0 * z) + self.dummy * 0 + x * 0
            Hrx = n_sub * Er
            Hix = n_sub * Ei
            zeros = torch.zeros_like(z) + self.dummy * 0 + x * 0
            return torch.stack([Er, Ei, Hrx, Hix, zeros, zeros], dim=-1)

        def field_components(self, x, z):
            out = self.forward(x, z)
            return tuple(out[:, i] for i in range(6))

    model = OutgoingModel()
    N = 32
    x = torch.linspace(0.0, physics.period, N, dtype=dtype)
    z = torch.full((N,), physics.domain_height, dtype=dtype)
    loss = maxwell_2d_bottom_bc_loss(model, x, z, physics)
    assert float(loss.detach()) < 1e-20


# ---------------------------------------------------------------------------
# 5. Periodic BC loss is zero when left == right
# ---------------------------------------------------------------------------


def test_periodic_bc_zero_for_constant_field(physics, small_model, dtype, device):
    """A spatially uniform field satisfies periodic BC exactly."""
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, z):
            out = torch.ones(x.shape[0], 6, dtype=x.dtype, device=x.device)
            return out + self.dummy * 0

    model = ConstantModel()
    N = 32
    z = torch.linspace(0.0, physics.domain_height, N, dtype=dtype)
    x_left  = torch.zeros(N, dtype=dtype)
    x_right = torch.full((N,), physics.period, dtype=dtype)
    loss = maxwell_2d_periodic_bc_loss(model, x_left, z, x_right)
    assert float(loss.detach()) < 1e-20


# ---------------------------------------------------------------------------
# 6. Sample_collocation_points shapes
# ---------------------------------------------------------------------------


def test_sample_collocation_shapes(physics, dtype, device):
    n_int, n_top, n_bot, n_per = 256, 64, 64, 64
    pts = sample_collocation_points(physics, n_int, n_top, n_bot, n_per,
                                    device, dtype, seed=0)
    assert pts["x_int"].shape[0] <= n_int
    assert pts["x_top"].shape[0] == n_top
    assert pts["x_bot"].shape[0] == n_bot
    assert pts["x_left"].shape[0] == n_per
    assert pts["x_right"].shape[0] == n_per
    assert pts["z_per"].shape[0] == n_per


# ---------------------------------------------------------------------------
# 7. Interior points exclude interface margins
# ---------------------------------------------------------------------------


def test_interior_points_exclude_interfaces(physics, dtype, device):
    margin = 0.02
    pts = sample_collocation_points(physics, 512, 64, 64, 64,
                                    device, dtype, seed=1, margin=margin)
    from src.geometry import interface_mask
    near = interface_mask(pts["x_int"], pts["z_int"], physics, margin)
    assert not near.any(), "Interior points should not be near interfaces"


# ---------------------------------------------------------------------------
# 8. Total loss returns all expected keys
# ---------------------------------------------------------------------------


def test_total_loss_keys(physics, small_model, dtype, device):
    pts = sample_collocation_points(physics, 64, 16, 16, 16, device, dtype, seed=2)
    losses = maxwell_2d_total_loss(small_model, pts, physics)
    for key in ("pde", "top", "bottom", "periodic", "total"):
        assert key in losses
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])


# ---------------------------------------------------------------------------
# 9. Incident plane-wave satisfies all six Maxwell equations analytically
# ---------------------------------------------------------------------------


def test_incident_plane_wave_maxwell_equations(physics):
    """Verify each Maxwell equation for E=exp(-ik0 z), H̃_x=n*E, H̃_z=0."""
    k0    = physics.k0
    n     = physics.n_air
    dz    = 1e-7
    z_pts = np.array([0.3, 0.7, 1.1])

    def E(z): return np.exp(-1j * k0 * z)
    def H(z): return n * E(z)  # H̃_x = +n * Ey

    Ep  = E(z_pts + dz); Em = E(z_pts - dz)
    Hp  = H(z_pts + dz); Hm = H(z_pts - dz)
    dE_dz = (Ep - Em) / (2 * dz)
    dH_dz = (Hp - Hm) / (2 * dz)

    # Eq (Ar): dEr/dz = k0*Hi_x = k0*n*(-sin(k0*z)) -> k0*Im(H) = -k0*n*sin(k0*z)
    np.testing.assert_allclose(np.real(dE_dz), k0 * np.imag(H(z_pts)), atol=1e-5)
    # Eq (Ai): dEi/dz = -k0*Hr_x = -k0*n*cos(k0*z)
    np.testing.assert_allclose(np.imag(dE_dz), -k0 * np.real(H(z_pts)), atol=1e-5)
    # Eq (Cr): dHr_x/dz - 0 = k0*eps_r*Ei
    np.testing.assert_allclose(np.real(dH_dz), k0 * n**2 * np.imag(E(z_pts)), atol=1e-5)
    # Eq (Ci): dHi_x/dz - 0 = -k0*eps_r*Er
    np.testing.assert_allclose(np.imag(dH_dz), -k0 * n**2 * np.real(E(z_pts)), atol=1e-5)


# ---------------------------------------------------------------------------
# 10. Sign convention: H̃_x = +n * E for forward plane wave
# ---------------------------------------------------------------------------


def test_sign_convention_H_tilde(physics):
    """H̃_x = +n * E_y for a downward-propagating plane wave."""
    k0 = physics.k0; n = physics.n_air
    z  = np.array([0.0, 0.25, 0.5, 1.0])
    E  = np.exp(-1j * k0 * z)
    # dE/dz = -ik0*E,  H̃_x = (i/k0)*dE/dz = (i/k0)*(-ik0*E) = -i^2*E = E ... wait
    # No: from the validated equation (Ar): dEr/dz = k0*Hi_x  => Hi_x = dEr_dz / k0
    dz = 1e-8
    Er_p = np.real(np.exp(-1j*k0*(z+dz))); Er_m = np.real(np.exp(-1j*k0*(z-dz)))
    dEr_dz = (Er_p - Er_m)/(2*dz)
    Hi_x_from_eq = dEr_dz / k0
    # Also: H̃_x = n*Ey, so Hi_x = n*Ei = n*(-sin(k0*z))
    Hi_x_from_H = n * (-np.sin(k0*z))
    np.testing.assert_allclose(Hi_x_from_eq, Hi_x_from_H, atol=1e-5)
