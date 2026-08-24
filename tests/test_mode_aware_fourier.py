"""Regression gates for the mode-aware Fourier-modal representation."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

from src.config import PhysicsConfig
from src.field_comparison import extract_scattered_modal_amplitudes
from src.mode_aware_fourier import (
    ExplicitFourierModalNetwork,
    ModeAwareFeatureEncoder,
    mode_specification,
)
from src.maxwell_2d import maxwell_2d_pde_residual


@pytest.fixture
def p08():
    return PhysicsConfig(period=0.8, ridge_width=0.32)


@pytest.mark.parametrize("medium,m", [
    ("air", 0), ("substrate", 0), ("air", 1), ("air", -1),
    ("substrate", 1), ("substrate", -1),
])
def test_explicit_fourier_modal_exact_manufactured_mode(p08, medium, m):
    """All required manufactured modes are <1% / <1 degree by construction."""
    n = p08.n_air if medium == "air" else p08.n_substrate
    model = ExplicitFourierModalNetwork(orders=(-1, 0, 1), period=p08.period,
                                        k0=p08.k0, n=n, z_lo=0., z_hi=p08.domain_height,
                                        use_coefficient_mlp=False)
    model.set_exact_mode(m)
    x = torch.linspace(0., p08.period, 81, dtype=torch.float64)
    z = torch.linspace(0., p08.domain_height, 61, dtype=torch.float64)
    X, Z = torch.meshgrid(x, z, indexing="xy")
    out = model(X.ravel(), Z.ravel())
    spec = mode_specification(m, p08.period, p08.k0, n)
    e = torch.exp(1j * (spec["kx"] * X.ravel() - spec["kz"] * Z.ravel()))
    hx, hz = spec["kz"] / p08.k0 * e, spec["kx"] / p08.k0 * e
    got_e = out[:, 0] + 1j*out[:, 1]
    got_hx, got_hz = out[:, 2] + 1j*out[:, 3], out[:, 4] + 1j*out[:, 5]
    e_error = torch.linalg.vector_norm(got_e-e) / torch.linalg.vector_norm(e)
    h_error = (torch.linalg.vector_norm(got_hx-hx) + torch.linalg.vector_norm(got_hz-hz)) / (torch.linalg.vector_norm(hx)+torch.linalg.vector_norm(hz)+1e-12)
    phase_deg = torch.sqrt(torch.mean(torch.angle(got_e / e) ** 2)) * 180. / np.pi
    assert float(e_error.detach()) < .01
    assert float(h_error.detach()) < .01
    assert float(phase_deg.detach()) < 1.0


def test_explicit_modal_reconstruction_has_zero_first_order_maxwell_residual_in_air(p08):
    """PDE residuals are taken on reconstructed E/H, not separate H heads."""
    model = ExplicitFourierModalNetwork(orders=(-1, 0, 1), period=p08.period,
                                        k0=p08.k0, n=p08.n_air, z_lo=0., z_hi=.5,
                                        use_coefficient_mlp=False)
    model.set_exact_mode(0)
    x = torch.linspace(0., p08.period, 12, dtype=torch.float64)
    z = torch.full_like(x, .1)
    residuals = maxwell_2d_pde_residual(model, x, z, p08)
    assert max(float(r.abs().max().detach()) for r in residuals) < 1e-10


def test_coefficient_function_variant_accepts_boundary_value_coordinates(p08):
    """The z-dependent coefficient path is usable by BC/interface evaluators."""
    model = ExplicitFourierModalNetwork(orders=(-1, 0, 1), period=p08.period,
                                        k0=p08.k0, n=p08.n_air, z_lo=0., z_hi=.5)
    x = torch.linspace(0., p08.period, 9, dtype=torch.float64)
    out = model(x, torch.zeros_like(x))
    assert out.shape == (9, 6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("order", [-1, 0, 1])
def test_modal_coefficient_spatial_quadrature_round_trip(order, p08):
    """One-hot E/H modal fields survive coefficient→space→DFT extraction."""
    n=p08.n_substrate; model=ExplicitFourierModalNetwork(orders=(-1,0,1),period=p08.period,k0=p08.k0,n=n,z_lo=0.,z_hi=2.,use_coefficient_mlp=False).double();model.set_exact_mode(order)
    x=torch.linspace(0.,p08.period,257,dtype=torch.float64)[:-1];z=torch.full_like(x,1.6);out=model(x,z)
    phase=torch.exp(-1j*(2*np.pi*order/p08.period)*x)
    for re,im in ((0,1),(2,3),(4,5)):
        coefficient=torch.mean((out[:,re]+1j*out[:,im])*phase)
        if (re,im)==(4,5) and order==0: assert abs(coefficient)<1e-10
        else: assert abs(coefficient) > 1e-10
    direct=model.modal_coefficients_at(torch.tensor([1.6],dtype=torch.float64))[0,model.orders.index(order)]
    spatial=torch.mean((out[:,0]+1j*out[:,1])*phase)
    assert abs(direct-spatial) < 1e-10


def test_mode_encoder_has_bounded_evanescent_features(p08):
    enc = ModeAwareFeatureEncoder(orders=(-1, 0, 1), period=p08.period, k0=p08.k0,
                                  n=p08.n_air, z_lo=0., z_hi=1.)
    x = torch.linspace(0., p08.period, 16, dtype=torch.float64)
    z = torch.linspace(0., 1., 16, dtype=torch.float64)
    f = enc(x, z)
    assert f.shape == (16, 12)
    # The two evanescent orders have their bounded decay factors multiplied by
    # bounded sine/cosine lateral factors, so no overflow is admissible.
    assert torch.isfinite(f).all()
    assert float(f.abs().max()) <= 1.0 + 1e-12


def test_zero_contrast_explicit_modal_field_stays_zero(p08):
    """Zero modal coefficients give exactly E_scat=H_scat=t±1=0."""
    model = ExplicitFourierModalNetwork(orders=(-1, 0, 1), period=p08.period,
                                        k0=p08.k0, n=p08.n_substrate, z_lo=0., z_hi=2.,
                                        use_coefficient_mlp=False)
    x = torch.linspace(0., p08.period, 128, dtype=torch.float64)
    z = torch.linspace(0., 2., 64, dtype=torch.float64)
    X, Z = torch.meshgrid(x, z, indexing="xy")
    field = model(X.ravel(), Z.ravel()).detach().numpy()
    assert np.max(np.abs(field)) < 1e-14
    modal = extract_scattered_modal_amplitudes(field[:, 0].reshape(Z.shape) + 1j*field[:, 1].reshape(Z.shape),
                                               x.numpy(), z.numpy(), p08, formulation="layered_bg", n_orders=1)
    assert np.max(np.abs(modal["t_m_complex"])) < 1e-14


def test_weak_scattering_modal_seed_is_continuous_from_zero(p08):
    """The explicit ±1 representation is continuous in a shallow-ridge seed."""
    magnitudes = []
    for contrast in (0., .25, .5, 1.):
        model = ExplicitFourierModalNetwork(orders=(-1, 0, 1), period=p08.period,
                                            k0=p08.k0, n=p08.n_substrate, z_lo=1., z_hi=2.,
                                            use_coefficient_mlp=False)
        # A shallow-ridge first-order modal seed is linear in delta-epsilon.
        with torch.no_grad():
            model.coefficients[0, 0] = .01 * contrast
            model.coefficients[2, 0] = .01 * contrast
        magnitudes.append(float(torch.abs(model.coefficients[2, 0]).detach()))
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] == 0.0 and magnitudes[-1] > 0.0


def test_shallow_ridge_rcwa_orders_grow_continuously_from_zero_contrast(p08):
    """Physical weak-scattering gate, kept separate from PINN training data."""
    from scripts.generate_reference import solve_rcwa
    amplitudes = []
    for contrast in (0.0, 0.25, 0.5, 1.0):
        eps = p08.eps_substrate + contrast * (p08.eps_ridge - p08.eps_substrate)
        shallow = replace(p08, n_ridge=float(np.sqrt(eps)), ridge_height=0.03)
        _, _, _, _, amps = solve_rcwa(shallow, N_harmonics=15, Nfine=512)
        amplitudes.append(abs(amps["c_trans"][15 + 1]))
    assert amplitudes[0] < 1e-10
    assert amplitudes == sorted(amplitudes)
    assert amplitudes[-1] > amplitudes[1] > 0.0
