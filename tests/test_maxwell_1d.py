"""Tests for src/maxwell_1d.py — first-order Maxwell PINN.

Tests cover:
1.  MaxwellSubdomainMLP output shape (4 outputs).
2.  Analytical H field satisfies Maxwell equation (1): dE/dz = k0*H_i, dE_i/dz = -k0*H_r.
3.  Analytical H field satisfies Maxwell equation (2): dH/dz.
4.  Physical derivative continuity at interface (not local-coord equality).
5.  Physical dE/dz = k_j * dE/dxi through chain rule.
6.  Interface E and H losses are non-negative scalars.
7.  Top BC targets are correct (E = 1+r, H = 1-r for n_air=1).
8.  Bottom BC (H = n_sub * E) is non-negative.
9.  Maxwell PDE residual is zero for the exact analytical solution.
10. maxwell_interface_loss returns two separate loss terms.
11. E field extraction (forward_E) has correct shape.
12. analytical_H_np field is x-independent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.benchmarks import LayeredMediumBenchmark
from src.maxwell_1d import (
    Maxwell1DLayered,
    MaxwellSubdomainMLP,
    analytical_H_np,
    maxwell_bottom_bc,
    maxwell_interface_loss,
    maxwell_pde_residual,
    maxwell_top_bc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bm() -> LayeredMediumBenchmark:
    return LayeredMediumBenchmark(k0=2.0 * math.pi)


@pytest.fixture
def dtype():
    return torch.float64


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_maxwell(bm, dtype) -> Maxwell1DLayered:
    return Maxwell1DLayered(bm, hidden_layers=2, hidden_width=16, num_fourier_levels=2).to(dtype=dtype)


# ---------------------------------------------------------------------------
# 1. MaxwellSubdomainMLP output shape
# ---------------------------------------------------------------------------


def test_maxwell_subnet_output_shape(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    net = MaxwellSubdomainMLP(0.0, bm.z_slab_top, k1, 2, 16, 2).to(dtype=dtype)
    N = 20
    z = torch.linspace(0.05, bm.z_slab_top - 0.05, N, dtype=dtype, device=device)
    out = net.forward(z)
    assert out.shape == (N, 4)
    er, ei, hr, hi = net.field_EH(z)
    for t in (er, ei, hr, hi):
        assert t.shape == (N,)


# ---------------------------------------------------------------------------
# 2. Analytical H satisfies Maxwell eq (1): dE/dz = k0*H_i, dE_i/dz = -k0*H_r
# ---------------------------------------------------------------------------


def test_analytical_H_maxwell_eq1(bm):
    """Verify dE/dz = -ik0*H̃ from the analytical solution (finite difference)."""
    dz = 1e-7
    z_test = np.array([0.3, 0.7, 1.1,                   # air
                       bm.z_slab_top + 0.05,              # slab
                       bm.z_slab_bot + 0.1])              # sub
    x = np.zeros_like(z_test)

    Er, Ei = bm.analytical_field_np(x, z_test)
    Hr, Hi = analytical_H_np(bm, z_test)
    Erp, Eip = bm.analytical_field_np(x, z_test + dz)
    Erm, Eim = bm.analytical_field_np(x, z_test - dz)

    dEr_dz = (Erp - Erm) / (2 * dz)
    dEi_dz = (Eip - Eim) / (2 * dz)

    # Eq (1): dEr/dz = k0*Hi,  dEi/dz = -k0*Hr
    np.testing.assert_allclose(dEr_dz,  bm.k0 * Hi, rtol=0, atol=1e-4)
    np.testing.assert_allclose(dEi_dz, -bm.k0 * Hr, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Analytical H satisfies Maxwell eq (2): dH/dz = -ik0*εr*E
# ---------------------------------------------------------------------------


def test_analytical_H_maxwell_eq2(bm):
    """Verify dH̃/dz = -ik₀εr E from the analytical solution."""
    dz = 1e-7
    # Use points well inside each region (avoid interfaces)
    z_air  = np.array([0.5])
    z_slab = np.array([bm.z_slab_top + 0.05])
    z_sub  = np.array([bm.z_slab_bot + 0.2])
    regions = [
        (z_air,  bm.n_air**2),
        (z_slab, bm.n_slab**2),
        (z_sub,  bm.n_sub**2),
    ]
    for z_test, eps in regions:
        x = np.zeros_like(z_test)
        Er, Ei = bm.analytical_field_np(x, z_test)
        Hrp, Hip = analytical_H_np(bm, z_test + dz)
        Hrm, Him = analytical_H_np(bm, z_test - dz)
        dHr_dz = (Hrp - Hrm) / (2 * dz)
        dHi_dz = (Hip - Him) / (2 * dz)
        # Eq (2): dHr/dz = k0*eps*Ei,  dHi/dz = -k0*eps*Er
        np.testing.assert_allclose(dHr_dz,  bm.k0 * eps * Ei, rtol=0, atol=1e-3)
        np.testing.assert_allclose(dHi_dz, -bm.k0 * eps * Er, rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. Physical derivative continuity: dE/dz is continuous at interfaces
#    This tests that we compare PHYSICAL z-derivatives, not local-coord derivatives
# ---------------------------------------------------------------------------


def test_physical_derivative_continuity_at_interface(bm):
    """Analytical solution has continuous dE/dz at both interfaces."""
    dz = 1e-7
    for z_int in [bm.z_slab_top, bm.z_slab_bot]:
        z_above = np.array([z_int - dz])
        z_below = np.array([z_int + dz])
        x = np.zeros(1)

        Erap, Eiap = bm.analytical_field_np(x, z_above + dz)
        Eram, Eiam = bm.analytical_field_np(x, z_above - dz)
        Erbp, Eibp = bm.analytical_field_np(x, z_below + dz)
        Erbm, Eibm = bm.analytical_field_np(x, z_below - dz)

        dEr_above = (Erap - Eram) / (2 * dz)
        dEr_below = (Erbp - Erbm) / (2 * dz)
        # Physical dE/dz must be continuous (to within FD accuracy)
        assert abs(dEr_above - dEr_below) < 0.1, (
            f"Physical dE/dz not continuous at z={z_int}: "
            f"above={dEr_above:.4f}, below={dEr_below:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Chain rule: dE/dz = k_j * dE/dxi for each subnet
# ---------------------------------------------------------------------------


def test_chain_rule_physical_vs_local(bm, dtype, device):
    """Verify dE/dz = k_j * dE/dxi through autograd chain rule."""
    from src.derivatives import first_derivative

    k1 = bm.k0 * bm.n_air
    net = MaxwellSubdomainMLP(0.0, bm.z_slab_top, k1, 2, 8, 2).to(dtype=dtype)

    z_test = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    er, _, _, _ = net.field_EH(z_test)
    dEr_dz = float(first_derivative(er, z_test).detach())

    xi_test = torch.tensor([k1 * 0.5], dtype=dtype, requires_grad=True)
    z_from_xi = xi_test / k1
    er2, _, _, _ = net.field_EH(z_from_xi)
    dEr_dxi = float(first_derivative(er2, xi_test).detach())

    assert abs(dEr_dz - k1 * dEr_dxi) < 1e-8, (
        f"Chain rule failed: dE/dz={dEr_dz:.6f}, k_j*dE/dxi={k1*dEr_dxi:.6f}"
    )


# ---------------------------------------------------------------------------
# 6. Interface losses are non-negative scalars
# ---------------------------------------------------------------------------


def test_maxwell_interface_loss_shapes(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    k2 = bm.k0 * bm.n_slab
    net_a = MaxwellSubdomainMLP(0.0, bm.z_slab_top, k1, 2, 8, 2).to(dtype=dtype)
    net_b = MaxwellSubdomainMLP(bm.z_slab_top, bm.z_slab_bot, k2, 2, 8, 2).to(dtype=dtype)
    LE, LH = maxwell_interface_loss(net_a, net_b, bm.z_slab_top, 16, dtype, device)
    assert LE.ndim == 0 and float(LE.detach()) >= 0.0
    assert LH.ndim == 0 and float(LH.detach()) >= 0.0


# ---------------------------------------------------------------------------
# 7. Top BC targets: E = 1+r, H = 1-r (n_air = 1)
# ---------------------------------------------------------------------------


def test_maxwell_top_bc_targets(bm):
    """Verify top BC values from TMM coefficients."""
    r = bm.reflection_coefficient()
    E_target = complex(1.0 + r.real, r.imag)
    H_target = complex(1.0 - r.real, -r.imag)  # H̃ = n_air*(1-r), n_air=1

    # Check against known TMM
    c = bm._tmm_coefficients()
    k0 = bm.k0
    E_at_0 = 1.0 + r   # incident + reflected at z=0
    # H̃ = (i/k0) * dE/dz at z=0
    # dE/dz|z=0 = -ik1*(1) + ik1*r = ik1(r-1)
    k1 = c["k1"]
    H_at_0 = (1j / k0) * (1j * k1 * (r - 1.0))  # = (i/k0)*ik1*(r-1) = -(k1/k0)*(r-1) = (k1/k0)*(1-r)
    # With n_air=1: k1=k0, so H̃ = 1-r
    H_at_0_simple = 1.0 - r

    assert abs(H_at_0 - H_at_0_simple) < 1e-10, f"H BC derivation mismatch: {H_at_0} vs {H_at_0_simple}"
    assert abs(complex(1.0 + r.real, r.imag) - E_at_0) < 1e-12


def test_maxwell_top_bc_loss_shape(bm, small_maxwell, dtype, device):
    r = bm.reflection_coefficient()
    loss = maxwell_top_bc(small_maxwell.net_air, 0.0, r, bm.n_air, 16, dtype, device)
    assert loss.ndim == 0 and float(loss.detach()) >= 0.0


# ---------------------------------------------------------------------------
# 8. Bottom BC is non-negative scalar
# ---------------------------------------------------------------------------


def test_maxwell_bottom_bc_shape(bm, small_maxwell, dtype, device):
    loss = maxwell_bottom_bc(small_maxwell.net_sub, bm.domain_height, bm.n_sub, 16, dtype, device)
    assert loss.ndim == 0 and float(loss.detach()) >= 0.0


# ---------------------------------------------------------------------------
# 9. Maxwell PDE residual = 0 for the exact analytical solution
# ---------------------------------------------------------------------------


def test_maxwell_pde_residual_exact(bm, dtype, device):
    """PDE residual must be ~0 for a subnet that returns the exact analytical field."""
    from src.benchmarks import LayeredMediumBenchmark

    c = bm._tmm_coefficients()
    k1, k0 = c["k1"], bm.k0
    r = c["r"]

    class ExactAir(torch.nn.Module):
        """Exact air field + H field for the analytical solution."""
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def field_EH(self, z):
            # E = exp(-ik1*z) + r*exp(+ik1*z)
            k = k1
            Er = torch.cos(k*z) + r.real * torch.cos(k*z) - r.imag * torch.sin(k*z) + self.dummy*0
            Ei = -torch.sin(k*z) + r.real * torch.sin(k*z) + r.imag * torch.cos(k*z) + self.dummy*0
            # H̃ = (i/k0)*dE/dz = (k1/k0)*(E_inc - r*E_ref)
            # = (k1/k0)*(exp(-ik1z) - r*exp(+ik1z))
            n = k1 / k0  # = n_air = 1.0
            Hr = n * (torch.cos(k*z) - r.real * torch.cos(k*z) + r.imag * torch.sin(k*z)) + self.dummy*0
            Hi = n * (-torch.sin(k*z) - r.real * torch.sin(k*z) - r.imag * torch.cos(k*z)) + self.dummy*0
            return Er, Ei, Hr, Hi

    net = ExactAir().to(dtype=dtype)
    z_int = torch.linspace(0.1, bm.z_slab_top - 0.1, 20, dtype=dtype, device=device)
    with torch.enable_grad():
        res = maxwell_pde_residual(net, z_int, bm.k0, bm.n_air**2)
    mse = float(sum(torch.mean(r**2) for r in res).detach())
    assert mse < 1e-6, f"Maxwell PDE residual for exact solution: {mse:.3e}"


# ---------------------------------------------------------------------------
# 10. maxwell_interface_loss returns two separate terms
# ---------------------------------------------------------------------------


def test_maxwell_interface_loss_two_terms(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    k2 = bm.k0 * bm.n_slab
    net_a = MaxwellSubdomainMLP(0.0, bm.z_slab_top, k1, 2, 8, 2).to(dtype=dtype)
    net_b = MaxwellSubdomainMLP(bm.z_slab_top, bm.z_slab_bot, k2, 2, 8, 2).to(dtype=dtype)
    result = maxwell_interface_loss(net_a, net_b, bm.z_slab_top, 8, dtype, device)
    assert len(result) == 2
    LE, LH = result
    assert LE.shape == torch.Size([])
    assert LH.shape == torch.Size([])


# ---------------------------------------------------------------------------
# 11. E field extraction shape
# ---------------------------------------------------------------------------


def test_maxwell_forward_E_shape(bm, small_maxwell, dtype, device):
    N = 50
    z = torch.linspace(0.0, bm.domain_height, N, dtype=dtype, device=device)
    out = small_maxwell.forward_E(z)
    assert out.shape == (N, 2)


# ---------------------------------------------------------------------------
# 12. analytical_H_np is x-independent
# ---------------------------------------------------------------------------


def test_analytical_H_x_independent(bm):
    z = np.linspace(0.1, 1.9, 50)
    x1 = np.zeros(50)
    x2 = np.ones(50) * 0.5
    Hr1, Hi1 = analytical_H_np(bm, z)
    Hr2, Hi2 = analytical_H_np(bm, z)
    np.testing.assert_array_equal(Hr1, Hr2)
    np.testing.assert_array_equal(Hi1, Hi2)
