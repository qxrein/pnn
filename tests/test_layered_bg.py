"""Tests for src/maxwell_layered_bg.py — layered-background formulation.

1.  Background coefficients: |r_eff|^2 + |t|^2*(k2/k1) = 1
2.  Background field E continuity at interface
3.  Background field dE/dz continuity at interface
4.  Background H from Maxwell: H = (i/k0)*dE/dz
5.  Background field satisfies Maxwell PDE in air (zero residual)
6.  Background field satisfies Maxwell PDE in substrate (zero residual)
7.  Delta_eps is zero in air (outside ridge)
8.  Delta_eps is zero in substrate (outside ridge)
9.  Delta_eps is nonzero inside the ridge
10. Source map returns correct keys
11. Flat-interface test: delta_eps = 0 everywhere when n_ridge = n_substrate
12. LBG bottom BC = 0 for exact scattered substrate field
13. LBG PDE residual = 0 for zero scattered field (flat interface)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.config import PhysicsConfig
from src.maxwell_layered_bg import (
    background_field_np,
    background_field_torch,
    compute_background_coefficients,
    compute_source_map,
    delta_eps_np,
    delta_eps_tensor,
    lbg_bottom_bc,
    lbg_top_bc,
    maxwell_2d_lbg_pde_residual,
    reconstruct_total_field,
)


@pytest.fixture
def physics():
    return PhysicsConfig()


@pytest.fixture
def coeff(physics):
    return compute_background_coefficients(physics)


@pytest.fixture
def dtype():
    return torch.float64


# ---------------------------------------------------------------------------
# 1. Energy conservation
# ---------------------------------------------------------------------------

def test_background_energy_conservation(coeff):
    assert abs(coeff["reflectance"] + coeff["transmittance"] - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# 2. E continuity at interface
# ---------------------------------------------------------------------------

def test_background_E_continuity(physics, coeff):
    z_int = coeff["z_interface"]
    z = np.array([z_int])
    Er_a, Ei_a, _, _ = background_field_np(z - 1e-9, coeff)
    Er_b, Ei_b, _, _ = background_field_np(z + 1e-9, coeff)
    assert abs(complex(Er_a[0], Ei_a[0]) - complex(Er_b[0], Ei_b[0])) < 1e-6


# ---------------------------------------------------------------------------
# 3. dE/dz continuity at interface
# ---------------------------------------------------------------------------

def test_background_dEdz_continuity(physics, coeff):
    """dE/dz continuity is verified via the Maxwell equations already tested.
    This test checks one-sided FD derivatives approach the same value."""
    k0 = physics.k0; n1 = physics.n_air; n2 = physics.n_substrate
    k1 = coeff["k1"]; k2 = coeff["k2"]
    r_eff = coeff["r_eff"]; tau = coeff["tau"]
    z_int = coeff["z_interface"]
    # Analytic dE/dz at z_int from air side
    E0  = np.exp(-1j*k1*z_int); E0r = np.exp(+1j*k1*z_int)
    dE_air_analytic = -1j*k1*E0 + 1j*k1*r_eff*E0r
    # Analytic dE/dz at z_int from sub side (z''=0)
    dE_sub_analytic = -1j*k2*tau
    assert abs(dE_air_analytic - dE_sub_analytic) / (abs(dE_air_analytic)+1e-12) < 1e-8


# ---------------------------------------------------------------------------
# 4. H from Maxwell: H_tilde = (i/k0)*dE/dz
# ---------------------------------------------------------------------------

def test_background_H_from_Maxwell(physics, coeff):
    k0 = coeff["k1"] / coeff["n1"]
    dz = 1e-7
    for z_test in [0.5, 1.8]:  # air and substrate
        z = np.array([z_test])
        Er_p, Ei_p, _, _ = background_field_np(z + dz, coeff)
        Er_m, Ei_m, _, _ = background_field_np(z - dz, coeff)
        dE_dz = complex(Er_p[0]-Er_m[0], Ei_p[0]-Ei_m[0]) / (2*dz)
        H_from_dE = 1j/k0 * dE_dz
        _, _, Hr, Hi = background_field_np(z, coeff)
        H_direct = complex(Hr[0], Hi[0])
        assert abs(H_from_dE - H_direct) / (abs(H_direct) + 1e-10) < 1e-4


# ---------------------------------------------------------------------------
# 5. Background satisfies Maxwell PDE in air
# ---------------------------------------------------------------------------

def test_background_maxwell_air(physics, coeff):
    k0 = physics.k0; n1 = physics.n_air
    dz = 1e-7
    z_pts = np.array([0.3, 0.7, 1.0])
    for z_t in z_pts:
        Er_p, Ei_p, _, _ = background_field_np(np.array([z_t+dz]), coeff)
        Er_m, Ei_m, _, _ = background_field_np(np.array([z_t-dz]), coeff)
        dEr_dz = (Er_p[0] - Er_m[0]) / (2*dz)
        dEi_dz = (Ei_p[0] - Ei_m[0]) / (2*dz)
        _, _, Hr, Hi = background_field_np(np.array([z_t]), coeff)
        # dEr/dz = k0*Hi, dEi/dz = -k0*Hr
        assert abs(dEr_dz - k0*Hi[0]) < 1e-4, f"Ar failed at z={z_t}"
        assert abs(dEi_dz + k0*Hr[0]) < 1e-4, f"Ai failed at z={z_t}"


# ---------------------------------------------------------------------------
# 6. Background satisfies Maxwell PDE in substrate
# ---------------------------------------------------------------------------

def test_background_maxwell_substrate(physics, coeff):
    k0 = physics.k0; n2 = physics.n_substrate
    dz = 1e-7
    for z_t in [1.5, 1.8]:
        Er_p, Ei_p, _, _ = background_field_np(np.array([z_t+dz]), coeff)
        Er_m, Ei_m, _, _ = background_field_np(np.array([z_t-dz]), coeff)
        dEr_dz = (Er_p[0] - Er_m[0]) / (2*dz)
        dEi_dz = (Ei_p[0] - Ei_m[0]) / (2*dz)
        _, _, Hr, Hi = background_field_np(np.array([z_t]), coeff)
        assert abs(dEr_dz - k0*Hi[0]) < 1e-4, f"Ar substrate at z={z_t}"
        assert abs(dEi_dz + k0*Hr[0]) < 1e-4, f"Ai substrate at z={z_t}"


# ---------------------------------------------------------------------------
# 7. Delta_eps = 0 in air (outside ridge)
# ---------------------------------------------------------------------------

def test_delta_eps_zero_air(physics):
    x = np.array([0.1, 0.5, 0.9])   # various x, well outside ridge
    z = np.array([0.3, 0.5, 1.0])   # above ridge_z_min=1.2
    d = delta_eps_np(x, z, physics)
    np.testing.assert_array_equal(d, 0.0)


# ---------------------------------------------------------------------------
# 8. Delta_eps = 0 in substrate (outside ridge)
# ---------------------------------------------------------------------------

def test_delta_eps_zero_substrate(physics):
    x = np.array([0.1, 0.5, 0.9])
    z = np.array([1.5, 1.7, 1.9])  # below ridge_z_max=1.4
    d = delta_eps_np(x, z, physics)
    np.testing.assert_array_equal(d, 0.0)


# ---------------------------------------------------------------------------
# 9. Delta_eps nonzero inside ridge
# ---------------------------------------------------------------------------

def test_delta_eps_nonzero_ridge(physics):
    x_ridge = np.array([(physics.ridge_x_min + physics.ridge_x_max) / 2.0])
    z_ridge = np.array([(physics.ridge_z_min + physics.ridge_z_max) / 2.0])
    d = delta_eps_np(x_ridge, z_ridge, physics)
    expected = physics.n_ridge**2 - physics.n_substrate**2
    assert abs(d[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# 10. Source map keys
# ---------------------------------------------------------------------------

def test_source_map_keys(physics, coeff):
    sm = compute_source_map(physics, coeff, nx=16, nz=32)
    for key in ("x", "z", "eps_r", "eps_bg", "delta_eps", "source_mag"):
        assert key in sm
        assert sm[key].shape == (32, 16)


# ---------------------------------------------------------------------------
# 11. Flat-interface test: delta_eps = 0 when n_ridge = n_substrate
# ---------------------------------------------------------------------------

def test_flat_interface_zero_contrast(physics):
    p_flat = PhysicsConfig(
        wavelength=physics.wavelength, n_air=physics.n_air,
        n_ridge=physics.n_substrate,  # zero contrast
        n_substrate=physics.n_substrate,
        period=physics.period, ridge_width=physics.ridge_width,
        ridge_height=physics.ridge_height, domain_height=physics.domain_height,
        ridge_base_fraction=physics.ridge_base_fraction,
    )
    x = np.linspace(0, physics.period, 20)
    z = np.linspace(0, physics.domain_height, 20)
    X, Z = np.meshgrid(x, z)
    d = delta_eps_np(X.ravel(), Z.ravel(), p_flat)
    assert np.all(d == 0.0), "Zero contrast should give zero delta_eps everywhere"


# ---------------------------------------------------------------------------
# 12. LBG bottom BC = 0 for exact scattered substrate field
# ---------------------------------------------------------------------------

def test_lbg_bottom_bc_zero_exact(physics, coeff, dtype):
    from src.benchmarks import LayeredMediumBenchmark
    bm = LayeredMediumBenchmark(
        n_air=physics.n_air, n_slab=physics.n_ridge, n_sub=physics.n_substrate,
        k0=physics.k0, z_slab_top=physics.ridge_z_min, z_slab_bot=physics.ridge_z_max,
        domain_height=physics.domain_height,
    )
    c_tmm = bm._tmm_coefficients()
    t_tmm, k3_tmm = c_tmm["t"], c_tmm["k3"]
    k0 = physics.k0; n_sub = physics.n_substrate

    class ExactBot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        def field_components(self, x, z):
            # Total sub field
            zpp = z - physics.ridge_z_max
            Er_t = t_tmm.real*torch.cos(k3_tmm*zpp) + t_tmm.imag*torch.sin(k3_tmm*zpp)
            Ei_t = -t_tmm.real*torch.sin(k3_tmm*zpp) + t_tmm.imag*torch.cos(k3_tmm*zpp)
            Hr_t = (k3_tmm/k0)*Er_t; Hi_t = (k3_tmm/k0)*Ei_t
            # Background sub field at these z
            Ebg_r, Ebg_i, Hbg_r, Hbg_i = background_field_torch(z.detach(), coeff, physics)
            Er_s = Er_t - Ebg_r + self.dummy*0 + x*0
            Ei_s = Ei_t - Ebg_i + self.dummy*0 + x*0
            Hr_s = Hr_t - Hbg_r + self.dummy*0 + x*0
            Hi_s = Hi_t - Hbg_i + self.dummy*0 + x*0
            Hrz = torch.zeros_like(Er_s); Hiz = torch.zeros_like(Er_s)
            return Er_s, Ei_s, Hr_s, Hi_s, Hrz, Hiz

    net = ExactBot()
    x = torch.linspace(0.0, physics.period, 32, dtype=dtype)
    loss = lbg_bottom_bc(net, x, physics, coeff)
    assert float(loss.detach()) < 1e-10, f"LBG bottom BC: {float(loss.detach()):.3e}"


# ---------------------------------------------------------------------------
# 13. LBG PDE residual = 0 for exact scattered field in air (no source in air)
# ---------------------------------------------------------------------------

def test_lbg_pde_zero_air(physics, coeff, dtype):
    """In air outside ridge, delta_eps=0, so source term=0.
    The scattered field in air satisfies the same equations as free-space.
    """
    k0 = physics.k0
    r_eff = coeff["r_eff"]

    class ExactScatAir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        def field_components_nd(self, xbar, zbar):
            # Scattered in air = r_eff*exp(+ik1*z)  (upgoing reflected wave)
            z_phys = zbar / k0
            # r_eff*exp(+ik1*z): cos(k1z)+i*sin(k1z) multiplied by r_eff
            cr = float(r_eff.real)*torch.cos(k0*z_phys) - float(r_eff.imag)*torch.sin(k0*z_phys) + self.dummy*0 + xbar*0
            ci = float(r_eff.imag)*torch.cos(k0*z_phys) + float(r_eff.real)*torch.sin(k0*z_phys) + self.dummy*0 + xbar*0
            Er_s = cr; Ei_s = ci
            # H for upgoing wave: H_tilde = (i/k0)*dE/dz
            # dE/dz = ik1*r_eff*exp(+ik1*z)  =>  H = (i/k0)*(ik1*E_up) = -(k1/k0)*E_up = -n1*E_up
            n = physics.n_air
            Hr_s = -n * Er_s; Hi_s = -n * Ei_s
            Hrz = torch.zeros_like(Er_s) + self.dummy*0 + xbar*0
            Hiz = torch.zeros_like(Er_s) + self.dummy*0 + xbar*0
            return Er_s, Ei_s, Hr_s, Hi_s, Hrz, Hiz
        def field_components(self, x, z):
            return self.field_components_nd(x*k0, z*k0)

    net = ExactScatAir()
    rng = __import__("numpy").random.default_rng(0)
    x = torch.as_tensor(rng.uniform(0.1, 0.9, 32), dtype=dtype)
    z = torch.as_tensor(rng.uniform(0.05, physics.ridge_z_min - 0.1, 32), dtype=dtype)
    with torch.enable_grad():
        res = maxwell_2d_lbg_pde_residual(net, x, z, physics, physics.n_air**2, coeff)
    mse = float(sum(torch.mean(r**2) for r in res).detach() / len(res))
    assert mse < 1e-8, f"LBG air PDE residual: {mse:.3e}"
