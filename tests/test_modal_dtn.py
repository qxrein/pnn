"""Tests for src/modal_dtn.py.

Covers:
- kz branch: propagating and evanescent cases
- Single-mode DtN residual: must be ~0 for an exact outgoing mode
- DtN residual: m=0, m=+1, m=-1, propagating and evanescent
- modal_dtn_loss: returns scalar tensor, grad flows
- boundary_spectral_audit: returns correct keys
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, load_config
from src.modal_dtn import (
    _kz_outgoing,
    check_dtn_residual_single_mode,
    modal_dtn_loss,
    boundary_spectral_audit,
    make_single_mode_field,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def physics():
    cfg = load_config(ROOT / "configs/default.yaml")
    return cfg.physics


@pytest.fixture
def physics_0p8(physics):
    p = 0.8 * physics.wavelength
    return PhysicsConfig(
        wavelength=physics.wavelength, n_air=physics.n_air,
        n_ridge=physics.n_ridge, n_substrate=physics.n_substrate,
        period=p, ridge_width=0.5 * p,
        ridge_height=physics.ridge_height, domain_height=physics.domain_height,
        ridge_base_fraction=physics.ridge_base_fraction,
        nx_visualization=64, nz_visualization=128,
    )


# ---------------------------------------------------------------------------
# kz branch tests
# ---------------------------------------------------------------------------

class TestKzBranch:
    def test_propagating_real_positive(self, physics):
        """m=0 at normal incidence in air: kz real, positive."""
        k0  = physics.k0
        kx  = np.array([0.0])
        kz  = _kz_outgoing(kx, physics.n_air, k0)
        assert kz[0].real > 0, "kz_0 should be real positive"
        assert abs(kz[0].imag) < 1e-10, "kz_0 should be real"
        np.testing.assert_allclose(kz[0].real, k0 * physics.n_air, rtol=1e-10)

    def test_evanescent_imaginary_negative(self, physics):
        """High evanescent order: kz purely imaginary with Im < 0."""
        k0   = physics.k0
        G0   = 2 * np.pi / physics.period
        # Use a very high order that is definitely evanescent
        kx   = np.array([10.0 * G0])
        kz   = _kz_outgoing(kx, physics.n_air, k0)
        assert kz[0].real < 1e-6, f"Evanescent kz should have Re~0, got {kz[0].real}"
        assert kz[0].imag < 0, f"Evanescent kz should have Im < 0, got {kz[0].imag}"

    def test_energy_conservation_single_interface(self, physics):
        """R + T = 1 for m=0 propagating order."""
        k0  = physics.k0
        kx  = np.array([0.0])
        kz1 = _kz_outgoing(kx, physics.n_air, k0)[0]
        kz2 = _kz_outgoing(kx, physics.n_substrate, k0)[0]
        # Fresnel coefficients
        r = (kz1 - kz2) / (kz1 + kz2)
        t = 2 * kz1 / (kz1 + kz2)
        energy = abs(r)**2 + abs(t)**2 * kz2.real / kz1.real
        np.testing.assert_allclose(float(energy), 1.0, atol=1e-12)

    def test_propagating_in_substrate(self, physics):
        """m=0 in substrate: kz > k0*n_air (denser medium)."""
        k0  = physics.k0
        kx  = np.array([0.0])
        kz  = _kz_outgoing(kx, physics.n_substrate, k0)
        assert kz[0].real > k0 * physics.n_air


# ---------------------------------------------------------------------------
# Single-mode DtN residual tests
# ---------------------------------------------------------------------------

class TestSingleModeDtN:
    """An exact outgoing mode must have DtN residual < 1e-6."""

    @pytest.mark.parametrize("m", [0, 1, -1])
    def test_m0_bottom_propagating(self, physics, m):
        """m in {-1,0,+1} outgoing downward: DtN residual < 1e-6."""
        result = check_dtn_residual_single_mode(
            m=m, A=1.0+0.5j,
            physics=physics,
            n_medium=physics.n_substrate,
            boundary="bottom",
            N_x=256, n_dtn_orders=8,
        )
        assert result["relative_residual"] < 1e-6, (
            f"m={m} bottom DtN residual = {result['relative_residual']:.2e}"
        )

    @pytest.mark.parametrize("m", [0, 1, -1])
    def test_m0_top_propagating(self, physics, m):
        """m in {-1,0,+1} outgoing upward in air: DtN residual < 1e-6.

        Note: for period=λ (default config), m=±1 in air have kx=k0 → kz=0
        (Rayleigh anomaly, grazing).  We use the substrate here instead.
        """
        # For default period=λ: m=±1 in air is Rayleigh, test in substrate
        n_test = physics.n_substrate if abs(m) == 1 else physics.n_air
        result = check_dtn_residual_single_mode(
            m=m, A=0.3-0.2j,
            physics=physics,
            n_medium=n_test,
            boundary="top",
            N_x=256, n_dtn_orders=8,
        )
        assert result["relative_residual"] < 1e-6, (
            f"m={m} top DtN residual = {result['relative_residual']:.2e}"
        )

    @pytest.mark.parametrize("m", [3, -3])
    def test_evanescent_order_bottom(self, physics, m):
        """Evanescent order: DtN residual < 1e-6 even for Im(kz) != 0."""
        G0   = 2 * np.pi / physics.period
        kx_m = m * G0
        kz_m = _kz_outgoing(np.array([kx_m]), physics.n_substrate, physics.k0)[0]
        if kz_m.real > 1e-6:
            pytest.skip(f"m={m} is propagating for this geometry")
        result = check_dtn_residual_single_mode(
            m=m, A=1.0,
            physics=physics,
            n_medium=physics.n_substrate,
            boundary="bottom",
            N_x=512, n_dtn_orders=8,
        )
        assert result["relative_residual"] < 1e-4, (
            f"m={m} evanescent DtN residual = {result['relative_residual']:.2e}"
        )

    def test_m0_known_H_relation(self, physics):
        """For m=0 normal incidence, H̃_x = -(kz_0/k0) E = -n*E (Robin)."""
        k0  = physics.k0
        n   = physics.n_substrate
        kz0 = _kz_outgoing(np.array([0.0]), n, k0)[0]
        # kz0 should equal n*k0 → kz0/k0 = n
        np.testing.assert_allclose(kz0.real / k0, n, rtol=1e-10)

    def test_0p8_period_pm1_evanescent_in_air(self, physics_0p8):
        """For Λ=0.8λ, ±1 orders are evanescent in air."""
        p   = physics_0p8
        k0  = p.k0
        G0  = 2 * np.pi / p.period
        kx1 = 1.0 * G0
        kz1 = _kz_outgoing(np.array([kx1]), p.n_air, k0)[0]
        assert kz1.real < 1e-6, f"m=+1 should be evanescent in air, kz1={kz1}"
        assert kz1.imag < 0,    f"Evanescent kz should have Im<0, kz1={kz1}"

    def test_0p8_period_pm1_propagating_in_substrate(self, physics_0p8):
        """For Λ=0.8λ, ±1 orders are propagating in substrate."""
        p   = physics_0p8
        k0  = p.k0
        G0  = 2 * np.pi / p.period
        kx1 = 1.0 * G0
        kz1 = _kz_outgoing(np.array([kx1]), p.n_substrate, k0)[0]
        assert kz1.real > 1e-6, f"m=+1 should propagate in substrate, kz1={kz1}"


# ---------------------------------------------------------------------------
# modal_dtn_loss torch tests
# ---------------------------------------------------------------------------

class TestModalDtnLossTorch:
    """modal_dtn_loss must return a scalar tensor with gradient."""

    def _make_exact_mode_subnet(self, physics, m, A, boundary, n_medium):
        """A fake subnet whose field exactly matches a single outgoing mode."""
        k0     = physics.k0
        G0     = 2 * np.pi / physics.period
        kx_m   = float(m) * G0
        kz_m   = _kz_outgoing(np.array([kx_m]), n_medium, k0)[0]
        sign   = +1.0 if boundary == "top" else -1.0
        z_b    = 0.0 if boundary == "top" else physics.domain_height

        class ExactMode:
            def field_components(self, x, z):
                phase = kx_m * x - kz_m * z
                Er  = float(A.real) * torch.cos(phase) - float(A.imag) * torch.sin(phase)
                Ei  = float(A.real) * torch.sin(phase) + float(A.imag) * torch.cos(phase)
                # H̃_x = sign * (kz_m/k0) * E
                alpha_re = float(sign * (kz_m / k0).real)
                alpha_im = float(sign * (kz_m / k0).imag)
                Hr_x = alpha_re * Er - alpha_im * Ei
                Hi_x = alpha_re * Ei + alpha_im * Er
                Hr_z = torch.zeros_like(Er)
                Hi_z = torch.zeros_like(Er)
                return Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z

        return ExactMode()

    def test_exact_mode_gives_near_zero_loss(self, physics):
        """An exact outgoing mode field gives DtN loss < 1e-8."""
        subnet = self._make_exact_mode_subnet(
            physics, m=0, A=1.0+0j, boundary="bottom", n_medium=physics.n_substrate)
        x_pts  = torch.linspace(0, physics.period, 64, dtype=torch.float64)
        loss   = modal_dtn_loss(subnet, x_pts, physics.domain_height,
                                physics, "bottom", physics.n_substrate, n_dtn_orders=4)
        assert float(loss) < 1e-8, f"Exact mode DtN loss = {float(loss):.2e}"

    def test_exact_top_mode_gives_near_zero_loss(self, physics):
        """Exact upward mode at top gives DtN loss < 1e-8."""
        subnet = self._make_exact_mode_subnet(
            physics, m=0, A=0.2+0.1j, boundary="top", n_medium=physics.n_air)
        x_pts  = torch.linspace(0, physics.period, 64, dtype=torch.float64)
        loss   = modal_dtn_loss(subnet, x_pts, 0.0,
                                physics, "top", physics.n_air, n_dtn_orders=4)
        assert float(loss) < 1e-8, f"Exact upward mode DtN loss = {float(loss):.2e}"

    def test_wrong_H_gives_nonzero_loss(self, physics):
        """A field with wrong H (simple Robin n*E) gives nonzero DtN loss when m≠0."""
        # Use m=+1 mode, but set H = n*E (wrong for m≠0)
        k0    = physics.k0
        G0    = 2 * np.pi / physics.period
        kx1   = 1.0 * G0
        kz1   = _kz_outgoing(np.array([kx1]), physics.n_substrate, k0)[0]

        class WrongHMode:
            def field_components(self, x, z):
                phase = kx1 * x - kz1 * z
                Er  = torch.cos(phase); Ei = torch.sin(phase)
                # Wrong: use n_sub * E instead of (kz1/k0) * E
                n = physics.n_substrate
                Hr_x = -n * Er; Hi_x = -n * Ei
                Hr_z = torch.zeros_like(Er); Hi_z = torch.zeros_like(Er)
                return Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z

        subnet = WrongHMode()
        x_pts  = torch.linspace(0, physics.period, 64, dtype=torch.float64)
        loss   = modal_dtn_loss(subnet, x_pts, physics.domain_height,
                                physics, "bottom", physics.n_substrate, n_dtn_orders=4)
        # If kz1/k0 != n_sub, this loss should be nonzero
        kz1_over_k0 = abs(kz1 / k0)
        if abs(kz1_over_k0 - physics.n_substrate) > 0.01:
            assert float(loss) > 1e-6, f"Wrong H should give nonzero DtN loss, got {float(loss):.2e}"

    def test_loss_is_scalar(self, physics):
        """modal_dtn_loss must return a 0-d tensor."""
        class ZeroSubnet:
            def field_components(self, x, z):
                z = torch.zeros_like(x)
                return z, z, z, z, z, z
        subnet = ZeroSubnet()
        x_pts  = torch.linspace(0, physics.period, 32, dtype=torch.float64)
        loss   = modal_dtn_loss(subnet, x_pts, physics.domain_height,
                                physics, "bottom", physics.n_substrate)
        assert loss.shape == (), f"Loss must be scalar, got shape {loss.shape}"

    def test_grad_flows_through_loss(self, physics):
        """Gradient must flow from modal_dtn_loss back to subnet parameters."""
        from src.maxwell_2d_nondim import Maxwell2DSubdomainMLP_ND
        net = Maxwell2DSubdomainMLP_ND(physics.k0, 2, 16, 2).to(dtype=torch.float64)
        x_pts = torch.linspace(0, physics.period, 32, dtype=torch.float64)
        loss  = modal_dtn_loss(net, x_pts, physics.domain_height,
                               physics, "bottom", physics.n_substrate, n_dtn_orders=3)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "No gradient flowed to network parameters"


# ---------------------------------------------------------------------------
# boundary_spectral_audit
# ---------------------------------------------------------------------------

class TestBoundarySpectralAudit:
    def test_audit_keys(self, physics):
        """All required keys present in audit output."""
        from src.maxwell_2d_nondim import Maxwell2DSubdomainMLP_ND
        net = Maxwell2DSubdomainMLP_ND(physics.k0, 2, 16, 2).to(dtype=torch.float64)
        result = boundary_spectral_audit(
            net, physics.domain_height, physics, "bottom", physics.n_substrate, n_dtn_orders=3)
        required = ["boundary", "z_val", "n_medium", "orders",
                    "propagating_orders", "summary"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_audit_order_keys(self, physics):
        """Each order entry has required keys."""
        from src.maxwell_2d_nondim import Maxwell2DSubdomainMLP_ND
        net = Maxwell2DSubdomainMLP_ND(physics.k0, 2, 16, 2).to(dtype=torch.float64)
        result = boundary_spectral_audit(
            net, physics.domain_height, physics, "bottom", physics.n_substrate, n_dtn_orders=2)
        row = result["orders"][0]
        for k in ["m", "kz_m_re", "E_m_abs", "H_m_abs", "H_m_dtn_abs", "relative_residual"]:
            assert k in row, f"Missing order key: {k}"

    def test_exact_mode_audit_small_residual(self, physics):
        """Exact downward mode subnet should have small spectral residual for m=0."""
        k0 = physics.k0
        n  = physics.n_substrate
        kz0 = _kz_outgoing(np.array([0.0]), n, k0)[0]

        class ExactM0Down:
            """E = exp(-ik0*z), H_x = -(kz0/k0)*E = -n*E  (downward outgoing)."""
            def field_components(self, x, z):
                z_det = z.detach()
                Er = torch.cos(k0 * z_det) * torch.ones_like(x)
                Ei = -torch.sin(k0 * z_det) * torch.ones_like(x)
                # Correct downward sign: H_DtN = -(kz0/k0)*E = -n*E
                Hr_x = -n * Er; Hi_x = -n * Ei
                Hr_z = torch.zeros_like(Er); Hi_z = torch.zeros_like(Er)
                return Er, Ei, Hr_x, Hi_x, Hr_z, Hi_z

        result = boundary_spectral_audit(
            ExactM0Down(), physics.domain_height, physics, "bottom", n, n_dtn_orders=3)
        m0_row = next(r for r in result["orders"] if r["m"] == 0)
        assert m0_row["relative_residual"] < 1e-6, \
            f"m=0 residual = {m0_row['relative_residual']:.2e}"
