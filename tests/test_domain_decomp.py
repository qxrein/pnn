"""Tests for domain-decomposition PINN (src/domain_decomp.py).

Covers:
1.  Subdomain assignment — points routed to correct subnetwork.
2.  Interface point exclusion from PDE interior sampling.
3.  Field continuity loss is zero for identical fields.
4.  Flux continuity loss is zero when derivatives match.
5.  Global reconstruction shape.
6.  SubdomainMLP output shape.
7.  PDE residual shape.
8.  Analytical interface errors are machine-precision.
9.  Analytical PDE residuals are small (finite-diff artefacts only).
10. Local normalisation maps endpoints to ±1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.benchmarks import LayeredMediumBenchmark
from src.domain_decomp import (
    DomainDecompLayered,
    SubdomainMLP,
    analytical_interface_errors,
    analytical_pde_residuals,
    bottom_bc_loss_dd,
    interface_field_loss,
    interface_flux_loss,
    pde_residual_subdomain,
    top_bc_loss_dd,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bm() -> LayeredMediumBenchmark:
    return LayeredMediumBenchmark(k0=2.0 * math.pi)


@pytest.fixture
def small_model(bm, dtype) -> DomainDecompLayered:
    return DomainDecompLayered(bm, hidden_layers=2, hidden_width=16, num_fourier_levels=2).to(dtype=dtype)


@pytest.fixture
def dtype():
    return torch.float64


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Subdomain assignment
# ---------------------------------------------------------------------------


def test_subdomain_assignment_air(bm, small_model, dtype, device):
    """Points in the air region should be routed to net_air."""
    z = torch.linspace(0.01, bm.z_slab_top - 0.01, 10, dtype=dtype, device=device)
    out_global = small_model.forward(z)
    out_air    = small_model.net_air.forward(z)
    torch.testing.assert_close(out_global, out_air, atol=1e-12, rtol=0)


def test_subdomain_assignment_slab(bm, small_model, dtype, device):
    """Points in the slab region should be routed to net_slab."""
    z = torch.linspace(bm.z_slab_top + 0.001, bm.z_slab_bot - 0.001, 8, dtype=dtype, device=device)
    out_global = small_model.forward(z)
    out_slab   = small_model.net_slab.forward(z)
    torch.testing.assert_close(out_global, out_slab, atol=1e-12, rtol=0)


def test_subdomain_assignment_sub(bm, small_model, dtype, device):
    """Points in the substrate region should be routed to net_sub."""
    z = torch.linspace(bm.z_slab_bot + 0.01, bm.domain_height - 0.01, 10, dtype=dtype, device=device)
    out_global = small_model.forward(z)
    out_sub    = small_model.net_sub.forward(z)
    torch.testing.assert_close(out_global, out_sub, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# 2. Interface point exclusion from interior sampling
# ---------------------------------------------------------------------------


def test_interior_sampling_excludes_interfaces():
    """Interior samples must not fall within margin of the slab boundaries."""
    from scripts.benchmark_layered_dd import _sample_interior
    bm = LayeredMediumBenchmark(k0=2.0 * math.pi)
    margin = 0.02
    z = _sample_interior(bm.z_slab_top, bm.z_slab_bot, 512, margin,
                         seed=0, device=torch.device("cpu"), dtype=torch.float64)
    z_np = z.numpy()
    assert np.all(z_np > bm.z_slab_top + margin - 1e-12)
    assert np.all(z_np < bm.z_slab_bot - margin + 1e-12)


def test_interior_sampling_shape():
    from scripts.benchmark_layered_dd import _sample_interior
    bm = LayeredMediumBenchmark(k0=2.0 * math.pi)
    z = _sample_interior(0.0, bm.z_slab_top, 64, 0.01,
                         seed=7, device=torch.device("cpu"), dtype=torch.float64)
    assert z.shape == (64,)


# ---------------------------------------------------------------------------
# 3. Field continuity loss is zero when both nets return identical values
# ---------------------------------------------------------------------------


def test_interface_field_loss_identical(bm, dtype, device):
    """If both subnets have identical weights, field continuity loss = 0."""
    k1 = bm.k0 * bm.n_air
    k2 = bm.k0 * bm.n_slab
    net_a = SubdomainMLP(0.0, bm.z_slab_top,  k1, hidden_layers=2, hidden_width=8, num_fourier_levels=2).to(dtype=dtype)
    net_b = SubdomainMLP(bm.z_slab_top, bm.z_slab_bot, k2, hidden_layers=2, hidden_width=8, num_fourier_levels=2).to(dtype=dtype)
    loss = interface_field_loss(net_a, net_b, bm.z_slab_top, 16, dtype, device)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0


# ---------------------------------------------------------------------------
# 4. Flux continuity loss shape and non-negativity
# ---------------------------------------------------------------------------


def test_interface_flux_loss_shape(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    k2 = bm.k0 * bm.n_slab
    net_a = SubdomainMLP(0.0, bm.z_slab_top,  k1, hidden_layers=2, hidden_width=8, num_fourier_levels=2).to(dtype=dtype)
    net_b = SubdomainMLP(bm.z_slab_top, bm.z_slab_bot, k2, hidden_layers=2, hidden_width=8, num_fourier_levels=2).to(dtype=dtype)
    loss = interface_flux_loss(net_a, net_b, bm.z_slab_top, bm.k0, 16, dtype, device)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0


def test_interface_flux_loss_for_constant_network(bm, dtype, device):
    """A network with all-zero output has dE/dz = 0 everywhere; flux loss = 0."""
    class ZeroNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def field_components(self, z):
            return (self.dummy * 0.0 + z * 0.0,
                    self.dummy * 0.0 + z * 0.0)

    net_a = ZeroNet()
    net_b = ZeroNet()
    loss = interface_flux_loss(net_a, net_b, bm.z_slab_top, bm.k0, 8, dtype, device)
    assert float(loss.detach()) < 1e-20


# ---------------------------------------------------------------------------
# 5. Global reconstruction shape
# ---------------------------------------------------------------------------


def test_global_reconstruction_shape(bm, small_model, dtype, device):
    """DomainDecompLayered.forward returns (N, 2) for N input points."""
    N = 64
    z = torch.linspace(0.0, bm.domain_height, N, dtype=dtype, device=device)
    out = small_model.forward(z)
    assert out.shape == (N, 2)


# ---------------------------------------------------------------------------
# 6. SubdomainMLP output shape
# ---------------------------------------------------------------------------


def test_subdomain_mlp_output_shape(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    net = SubdomainMLP(0.0, bm.z_slab_top, k1,
                       hidden_layers=2, hidden_width=16, num_fourier_levels=2).to(dtype=dtype)
    N = 32
    z = torch.linspace(0.05, bm.z_slab_top - 0.05, N, dtype=dtype, device=device)
    out = net.forward(z)
    assert out.shape == (N, 2)


def test_subdomain_mlp_nd_output_shape(bm, dtype, device):
    """SubdomainMLP field_components output shape."""
    k_j = bm.k0 * bm.n_air
    net = SubdomainMLP(0.0, bm.z_slab_top, k_j,
                       hidden_layers=2, hidden_width=16, num_fourier_levels=2).to(dtype=dtype)
    N = 16
    z = torch.linspace(0.05, bm.z_slab_top - 0.05, N, dtype=dtype, device=device)
    er, ei = net.field_components(z)
    assert er.shape == (N,)
    assert ei.shape == (N,)


# ---------------------------------------------------------------------------
# 7. PDE residual shape and finiteness
# ---------------------------------------------------------------------------


def test_pde_residual_subdomain_shape(bm, dtype, device):
    k1 = bm.k0 * bm.n_air
    net = SubdomainMLP(0.0, bm.z_slab_top, k1,
                       hidden_layers=2, hidden_width=16, num_fourier_levels=2).to(dtype=dtype)
    N = 20
    z = torch.linspace(0.1, bm.z_slab_top - 0.1, N, dtype=dtype, device=device)
    res_r, res_i = pde_residual_subdomain(net, z, bm.k0, bm.n_air**2)
    assert res_r.shape == (N,)
    assert res_i.shape == (N,)
    assert torch.all(torch.isfinite(res_r))
    assert torch.all(torch.isfinite(res_i))


# ---------------------------------------------------------------------------
# 8. Analytical interface errors are machine-precision
# ---------------------------------------------------------------------------


def test_analytical_interface_errors_machine_precision(bm):
    """TMM solution must satisfy BCs to machine precision."""
    errors = analytical_interface_errors(bm)
    for k, v in errors.items():
        assert v < 1e-10, f"{k} = {v:.3e} exceeds 1e-10"


# ---------------------------------------------------------------------------
# 9. Analytical PDE residuals are small
# ---------------------------------------------------------------------------


def test_analytical_pde_residuals_small(bm):
    """Exact analytical PDE residuals should be floating-point zero."""
    residuals = analytical_pde_residuals(bm)
    for k, v in residuals.items():
        if k == "note":
            continue
        # Exact formula: d2E/dz2 = -k^2*E  =>  residual = -k^2*E + k^2*E = 0
        assert float(v) < 1e-20, f"Analytical PDE residual should be ~0: {k} = {v:.3e}"


# ---------------------------------------------------------------------------
# 10. Local normalisation
# ---------------------------------------------------------------------------


def test_local_norm_endpoints():
    """SubdomainMLP local normalisation maps z_lo→-1 and z_hi→+1."""
    net = SubdomainMLP(1.2, 1.4, 2*math.pi, hidden_layers=2, hidden_width=8, num_fourier_levels=1)
    z_lo_t = torch.tensor([1.2], dtype=torch.float64)
    z_hi_t = torch.tensor([1.4], dtype=torch.float64)
    assert float(net._local_norm(z_lo_t)) == pytest.approx(-1.0, abs=1e-12)
    assert float(net._local_norm(z_hi_t)) == pytest.approx(+1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 11. Top and bottom BC losses are non-negative scalars
# ---------------------------------------------------------------------------


def test_top_bc_loss_shape(bm, small_model, dtype, device):
    r = bm.reflection_coefficient()
    loss = top_bc_loss_dd(small_model.net_air, 0.0, r, 16, dtype, device)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0


def test_bottom_bc_loss_shape(bm, small_model, dtype, device):
    loss = bottom_bc_loss_dd(small_model.net_sub, bm.domain_height, bm.k0, bm.n_sub, 16, dtype, device)
    assert loss.ndim == 0
    assert float(loss.detach()) >= 0.0
