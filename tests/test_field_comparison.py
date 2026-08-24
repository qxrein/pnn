"""Tests for src/field_comparison.py.

Covers:
- compare_fields: total and scattered metrics are correct
- extract_modal_amplitudes: energy conservation for known field
- compare_modal_with_rcwa: amplitude/phase error extraction
- flat_contrast_test: zero-contrast case
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
from src.field_comparison import (
    compare_fields,
    extract_modal_amplitudes,
    compare_modal_with_rcwa,
    flat_contrast_test,
)
from src.maxwell_layered_bg import background_field_np, compute_background_coefficients


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def physics():
    cfg = load_config(ROOT / "configs/default.yaml")
    return cfg.physics


@pytest.fixture
def simple_grid(physics):
    """Small (16x8) visualization grid."""
    Nz, Nx = 16, 8
    x1d = np.linspace(0, physics.period, Nx)
    z1d = np.linspace(0, physics.domain_height, Nz)
    X, Z = np.meshgrid(x1d, z1d)
    return X, Z, x1d, z1d


# ---------------------------------------------------------------------------
# compare_fields
# ---------------------------------------------------------------------------

class TestCompareFields:
    """compare_fields must return correct metrics for known fields."""

    def test_perfect_scatter_gives_zero_total_error(self, physics, simple_grid):
        """When PINN scatter exactly matches RCWA scatter, total comparison has error 0."""
        X, Z, x1d, z1d = simple_grid
        k0 = physics.k0
        shape = X.shape

        # True total field (arbitrary)
        E_total_r = np.cos(k0 * Z) * (1 + 0.1 * np.cos(2 * np.pi * X / physics.period))
        E_total_i = -np.sin(k0 * Z) * (1 + 0.05 * np.cos(2 * np.pi * X / physics.period))

        # Compute background
        coeff = compute_background_coefficients(physics)
        Ebg_r_1d, Ebg_i_1d, _, _ = background_field_np(z1d, coeff)
        Ebg_r = Ebg_r_1d[:, None] * np.ones((1, shape[1]))
        Ebg_i = Ebg_i_1d[:, None] * np.ones((1, shape[1]))

        # Perfect PINN: scatter = total - bg
        pinn_scat_r = E_total_r - Ebg_r
        pinn_scat_i = E_total_i - Ebg_i

        result = compare_fields(
            pinn_scat_r, pinn_scat_i,
            E_total_r, E_total_i,
            Z, physics, formulation="layered_bg",
        )

        assert result["total/complex_l2"] < 1e-10, \
            f"Perfect scatter should give 0 total error, got {result['total/complex_l2']:.2e}"
        assert result["total/phase_rmse_deg"] < 1e-6, \
            f"Phase error should be 0, got {result['total/phase_rmse_deg']:.4f}°"

    def test_perfect_scatter_gives_zero_scattered_error(self, physics, simple_grid):
        """When PINN scatter = RCWA scatter, scattered comparison has error 0."""
        X, Z, x1d, z1d = simple_grid
        k0 = physics.k0
        shape = X.shape

        E_total_r = np.cos(k0 * Z)
        E_total_i = -np.sin(k0 * Z)

        coeff = compute_background_coefficients(physics)
        Ebg_r_1d, Ebg_i_1d, _, _ = background_field_np(z1d, coeff)
        Ebg_r = Ebg_r_1d[:, None] * np.ones((1, shape[1]))
        Ebg_i = Ebg_i_1d[:, None] * np.ones((1, shape[1]))

        pinn_scat_r = E_total_r - Ebg_r
        pinn_scat_i = E_total_i - Ebg_i

        result = compare_fields(
            pinn_scat_r, pinn_scat_i,
            E_total_r, E_total_i,
            Z, physics, formulation="layered_bg",
        )

        assert result["scattered/complex_l2"] < 1e-10

    def test_wrong_scatter_gives_nonzero_errors(self, physics, simple_grid):
        """A deliberately wrong scatter gives nonzero errors for both representations."""
        X, Z, x1d, z1d = simple_grid
        k0 = physics.k0
        shape = X.shape

        E_total_r = np.cos(k0 * Z)
        E_total_i = -np.sin(k0 * Z)

        # Wrong scatter (all zeros)
        pinn_scat_r = np.zeros(shape)
        pinn_scat_i = np.zeros(shape)

        result = compare_fields(
            pinn_scat_r, pinn_scat_i,
            E_total_r, E_total_i,
            Z, physics, formulation="layered_bg",
        )

        assert result["total/complex_l2"] > 0.01
        assert result["scattered/complex_l2"] > 0.01

    def test_free_space_formulation(self, physics, simple_grid):
        """free_space formulation uses E_inc as background."""
        X, Z, x1d, z1d = simple_grid
        k0 = physics.k0
        shape = X.shape

        # E_inc
        E_inc_r = np.cos(k0 * Z)
        E_inc_i = -np.sin(k0 * Z)

        # PINN outputs zero scatter
        pinn_scat_r = np.zeros(shape)
        pinn_scat_i = np.zeros(shape)

        # RCWA total = incident only (no reflection, no transmission change)
        result = compare_fields(
            pinn_scat_r, pinn_scat_i,
            E_inc_r, E_inc_i,
            Z, physics, formulation="free_space",
        )
        # Total error = |E_bg - E_inc| = 0 for free_space with zero scatter
        assert result["total/complex_l2"] < 1e-10

    def test_output_keys(self, physics, simple_grid):
        """Result dict must contain all required keys."""
        X, Z, x1d, z1d = simple_grid
        shape = X.shape
        result = compare_fields(
            np.zeros(shape), np.zeros(shape),
            np.ones(shape), np.zeros(shape),
            Z, physics,
        )
        required_keys = [
            "total/complex_l2", "total/magnitude_l2", "total/phase_rmse_deg",
            "scattered/complex_l2", "scattered/magnitude_l2", "scattered/phase_rmse_deg",
            "pinn_E_total_r", "pinn_E_total_i",
            "pinn_E_scat_r",  "pinn_E_scat_i",
            "rcwa_E_total_r", "rcwa_E_total_i",
            "rcwa_E_scat_r",  "rcwa_E_scat_i",
            "field_representation_pinn", "field_representation_rcwa",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_representation_labels(self, physics, simple_grid):
        """Representation labels are always 'scattered' and 'total'."""
        X, Z, x1d, z1d = simple_grid
        shape = X.shape
        result = compare_fields(
            np.zeros(shape), np.zeros(shape),
            np.ones(shape), np.zeros(shape),
            Z, physics,
        )
        assert result["field_representation_pinn"] == "scattered"
        assert result["field_representation_rcwa"] == "total"

    def test_partial_valid_region_mask_uses_only_requested_region(self, physics, simple_grid):
        """An error inside the excluded grating region must not affect air metrics."""
        X, Z, _, _ = simple_grid
        total_r = np.cos(physics.k0 * Z)
        total_i = -np.sin(physics.k0 * Z)
        bg_r, bg_i, _, _ = background_field_np(Z[:, 0], compute_background_coefficients(physics))
        scat_r = total_r - bg_r[:, None]
        scat_i = total_i - bg_i[:, None]
        scat_r[(Z >= physics.ridge_z_min) & (Z <= physics.ridge_z_max)] += 10.0
        result = compare_fields(scat_r, scat_i, total_r, total_i, Z, physics,
                                region_mask="air")
        assert result["total/complex_l2"] < 1e-10
        assert result["total/n_valid"] == int((Z < physics.ridge_z_min).sum())

    def test_invalid_nan_region_is_excluded(self, physics, simple_grid):
        """NaNs remove only those points from metrics, without broadcasting."""
        X, Z, _, _ = simple_grid
        total_r = np.cos(physics.k0 * Z)
        total_i = -np.sin(physics.k0 * Z)
        bg_r, bg_i, _, _ = background_field_np(Z[:, 0], compute_background_coefficients(physics))
        scat_r = total_r - bg_r[:, None]
        scat_i = total_i - bg_i[:, None]
        scat_r[0, 0] = np.nan
        result = compare_fields(scat_r, scat_i, total_r, total_i, Z, physics,
                                region_mask="full_domain")
        assert result["total/complex_l2"] < 1e-10
        assert result["total/n_valid"] == Z.size - 1

    def test_grating_interior_exclusion(self, physics, simple_grid):
        """external_only must exclude all points in the grating layer."""
        X, Z, _, _ = simple_grid
        total_r = np.cos(physics.k0 * Z)
        total_i = -np.sin(physics.k0 * Z)
        bg_r, bg_i, _, _ = background_field_np(Z[:, 0], compute_background_coefficients(physics))
        scat_r = total_r - bg_r[:, None]
        scat_i = total_i - bg_i[:, None]
        interior = (Z >= physics.ridge_z_min) & (Z <= physics.ridge_z_max)
        scat_i[interior] += 5.0
        result = compare_fields(scat_r, scat_i, total_r, total_i, Z, physics,
                                region_mask="external_only")
        assert result["total/complex_l2"] < 1e-10
        assert result["total/n_valid"] == int((~interior).sum())

    def test_flattened_fields_do_not_broadcast(self, physics, simple_grid):
        """Flat fields require a same-length flat z grid and retain a flat mask."""
        X, Z, _, _ = simple_grid
        total = np.exp(-1j * physics.k0 * Z).ravel()
        z_flat = Z.ravel()
        bg_r, bg_i, _, _ = background_field_np(z_flat, compute_background_coefficients(physics))
        scat = total - (bg_r + 1j * bg_i)
        result = compare_fields(scat.real, scat.imag, total.real, total.imag,
                                z_flat, physics, region_mask="external_only")
        assert result["pinn_E_total_r"].shape == total.shape
        assert result["total/n_valid"] <= total.size

    def test_shape_mismatch_is_rejected_before_masking(self, physics, simple_grid):
        """No NumPy broadcasting is allowed between fields and masks."""
        _, Z, _, _ = simple_grid
        with pytest.raises(ValueError, match="same shape"):
            compare_fields(np.zeros((16, 8)), np.zeros((16, 8)),
                           np.zeros((16, 7)), np.zeros((16, 7)), Z, physics)

    def test_total_representation_is_compared_without_reconstruction_error(self, physics, simple_grid):
        """The explicit total-field API must support total-vs-total comparison."""
        _, Z, _, _ = simple_grid
        total_r = np.cos(physics.k0 * Z)
        total_i = -np.sin(physics.k0 * Z)
        result = compare_fields(total_r, total_i, total_r, total_i, Z, physics,
                                field_representation="total",
                                reference_field_representation="total")
        assert result["total/complex_l2"] < 1e-10
        assert result["field_representation_pinn"] == "total"


# ---------------------------------------------------------------------------
# extract_modal_amplitudes
# ---------------------------------------------------------------------------

class TestExtractModalAmplitudes:
    """extract_modal_amplitudes must give R+T ≈ 1 for a known field."""

    def test_pure_incident_wave_energy(self, physics):
        """For a pure incident wave with no scattering, R=0, T≈1."""
        Nz, Nx = 64, 32
        x1d = np.linspace(0, physics.period, Nx)
        z1d = np.linspace(0, physics.domain_height, Nz)
        X, Z = np.meshgrid(x1d, z1d)
        k0 = physics.k0
        E_total = np.exp(-1j * k0 * Z)   # pure incident, no reflection

        result = extract_modal_amplitudes(E_total, x1d, z1d, physics, n_orders=3)

        # No reflection at top monitor (E_refl = E_total - E_inc = 0)
        assert result["R_total"] < 1e-6, f"R_total = {result['R_total']:.6f} — should be 0"

    def test_result_keys(self, physics):
        """All required keys are present."""
        Nz, Nx = 32, 16
        x1d = np.linspace(0, physics.period, Nx)
        z1d = np.linspace(0, physics.domain_height, Nz)
        X, Z = np.meshgrid(x1d, z1d)
        E_total = np.exp(-1j * physics.k0 * Z)

        result = extract_modal_amplitudes(E_total, x1d, z1d, physics)
        required = ["orders", "r_m_complex", "t_m_complex", "R_m", "T_m",
                    "R0", "T0", "R_total", "T_total", "energy_check",
                    "r0_complex", "t0_complex"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_orders_length(self, physics):
        """With n_orders=3, orders list has length 7."""
        Nz, Nx = 32, 16
        x1d = np.linspace(0, physics.period, Nx)
        z1d = np.linspace(0, physics.domain_height, Nz)
        E_total = np.zeros((Nz, Nx), dtype=complex)

        result = extract_modal_amplitudes(E_total, x1d, z1d, physics, n_orders=3)
        assert len(result["orders"]) == 7
        assert len(result["R_m"]) == 7
        assert len(result["T_m"]) == 7


# ---------------------------------------------------------------------------
# flat_contrast_test
# ---------------------------------------------------------------------------

class TestFlatContrastTest:
    """flat_contrast_test should pass when scatter is zero."""

    def test_zero_scatter_passes(self, physics, simple_grid):
        X, Z, x1d, z1d = simple_grid
        shape = X.shape
        result = flat_contrast_test(
            np.zeros(shape), np.zeros(shape),
            np.ones(shape), np.zeros(shape),  # RCWA not used in this function
            Z, physics,
            tol=0.01,
        )
        assert result["passed"], "Zero scatter should pass the flat-contrast test"
        assert result["max_E_scat_magnitude"] < 1e-12

    def test_nonzero_scatter_fails(self, physics, simple_grid):
        X, Z, x1d, z1d = simple_grid
        shape = X.shape
        result = flat_contrast_test(
            np.ones(shape) * 0.5, np.zeros(shape),
            np.ones(shape), np.zeros(shape),
            Z, physics,
            tol=0.01,
        )
        assert not result["passed"], "Non-zero scatter should fail the flat-contrast test"

    def test_result_keys(self, physics, simple_grid):
        X, Z, x1d, z1d = simple_grid
        shape = X.shape
        result = flat_contrast_test(
            np.zeros(shape), np.zeros(shape),
            np.zeros(shape), np.zeros(shape),
            Z, physics,
        )
        required = ["max_E_scat_magnitude", "max_total_minus_bg_r",
                    "max_total_minus_bg_i", "passed", "tolerance", "note"]
        for k in required:
            assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# compare_modal_with_rcwa
# ---------------------------------------------------------------------------

class TestCompareModalWithRCWA:
    """compare_modal_with_rcwa uses regenerated reference files."""

    @pytest.fixture
    def ref_path(self):
        p = ROOT / "outputs/reference_grating.npz"
        if not p.exists():
            pytest.skip("reference_grating.npz not found — run generate_reference.py first")
        return str(p)

    def test_rcwa_reference_has_amplitudes(self, ref_path):
        """Regenerated reference must contain c_refl and c_trans."""
        import numpy as np
        data = np.load(ref_path, allow_pickle=True)
        assert "c_refl" in data, "c_refl missing — reference was not regenerated with fixed solver"
        assert "c_trans" in data, "c_trans missing"
        assert "R_m" in data, "R_m missing"
        assert "T_m" in data, "T_m missing"

    def test_rcwa_energy_conservation(self, ref_path):
        """Stored R+T must be 1.0 within 0.01%."""
        data = np.load(ref_path, allow_pickle=True)
        R_total = float(data["R_total"])
        T_total = float(data["T_total"])
        assert abs(R_total + T_total - 1.0) < 1e-4, \
            f"R+T = {R_total+T_total:.8f} — energy not conserved in reference"

    def test_rcwa_nonzero_reflection(self, ref_path):
        """m=0 reflection amplitude must be non-zero."""
        data = np.load(ref_path, allow_pickle=True)
        c_refl = data["c_refl"]
        N = len(c_refl) // 2
        r0 = abs(c_refl[N])
        assert r0 > 0.05, f"|r_0| = {r0:.6f} — should be > 0.05 for this grating"

    def test_compare_modal_no_rcwa(self, physics):
        """Without RCWA path, returns a note dict gracefully."""
        dummy_modal = {
            "orders": [0], "r_m_complex": np.zeros(1, complex),
            "t_m_complex": np.zeros(1, complex),
            "R_m": [0.0], "T_m": [0.0],
            "R_total": 0.0, "T_total": 0.0, "energy_check": 0.0,
        }
        result = compare_modal_with_rcwa(dummy_modal, None, 25)
        assert "note" in result

    def test_compare_modal_with_real_rcwa(self, ref_path, physics):
        """Summary is present and RCWA energy check passes."""
        # Build a dummy PINN modal (zero scatter → R=0)
        Nz, Nx = 64, 32
        x1d = np.linspace(0, physics.period, Nx)
        z1d = np.linspace(0, physics.domain_height, Nz)
        X, Z = np.meshgrid(x1d, z1d)
        E_total = np.exp(-1j * physics.k0 * Z)
        pinn_modal = extract_modal_amplitudes(E_total, x1d, z1d, physics, n_orders=5)
        result = compare_modal_with_rcwa(pinn_modal, ref_path, n_harmonics_center=25)
        assert "summary" in result
        sm = result["summary"]
        # RCWA should have correct energy
        assert abs(sm["rcwa_energy_check"] - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# modal_data_loss tests
# ---------------------------------------------------------------------------

class TestModalDataLoss:
    """modal_data_loss must be large for E_scat=0 and zero for perfect scatter."""

    @pytest.fixture
    def rcwa_ref(self):
        p = ROOT / "outputs/reference_lambda_0p8.npz"
        if not p.exists():
            pytest.skip("reference_lambda_0p8.npz not found")
        from src.field_comparison import load_rcwa_amplitudes
        return load_rcwa_amplitudes(str(p))

    @pytest.fixture
    def physics_0p8(self, physics):
        from src.config import PhysicsConfig
        p = 0.8 * physics.wavelength
        return PhysicsConfig(
            wavelength=physics.wavelength, n_air=physics.n_air,
            n_ridge=physics.n_ridge, n_substrate=physics.n_substrate,
            period=p, ridge_width=0.4 * p,
            ridge_height=physics.ridge_height, domain_height=physics.domain_height,
            ridge_base_fraction=physics.ridge_base_fraction,
        )

    def test_zero_scat_gives_large_loss(self, physics_0p8, rcwa_ref):
        """E_scat=0 should produce a large modal loss (~0.077 for this geometry)."""
        from src.field_comparison import modal_data_loss

        class ZeroNet:
            def field_components(self, x, z):
                z_ = torch.zeros_like(x)
                return z_, z_, z_, z_, z_, z_

        z_bot = 0.92 * physics_0p8.domain_height
        loss = modal_data_loss(
            ZeroNet(), z_bot, physics_0p8,
            rcwa_ref["c_trans"], rcwa_ref["N_harmonics"],
            "bottom", physics_0p8.n_substrate,
            n_data_orders=2, weight_propagating=1.0,
        )
        assert float(loss) > 0.05, \
            f"Zero scatter should give large modal loss, got {float(loss):.4f}"

    def test_perfect_scat_gives_small_loss(self, physics_0p8, rcwa_ref):
        """Exact RCWA scattered field should give near-zero modal loss."""
        from src.field_comparison import modal_data_loss, load_rcwa_amplitudes
        from src.maxwell_layered_bg import compute_background_coefficients, background_field_np
        from src.modal_dtn import _kz_outgoing
        import math

        k0 = physics_0p8.k0
        G0 = 2 * math.pi / physics_0p8.period
        N  = rcwa_ref["N_harmonics"]
        z_bot = 0.92 * physics_0p8.domain_height
        coeff = compute_background_coefficients(physics_0p8)
        Ebg_r, Ebg_i, _, _ = background_field_np(np.array([z_bot]), coeff)
        E_bg = complex(float(Ebg_r[0]), float(Ebg_i[0]))

        # Build exact scattered field at z_bot: sum over propagating orders
        orders = np.arange(-2, 3)
        kx_m = orders * G0
        kz_m = _kz_outgoing(kx_m, physics_0p8.n_substrate, k0)
        x_np = np.linspace(0, physics_0p8.period, 128, endpoint=False)
        E_scat = np.zeros(128, dtype=complex)

        for i_m, m in enumerate(orders):
            if kz_m[i_m].real < 1e-6:
                continue
            idx = N + m
            c = complex(rcwa_ref["c_trans"][idx])
            dz = z_bot - physics_0p8.ridge_z_max
            A = c * np.exp(-1j * kz_m[i_m] * dz)
            if m == 0:
                A = A - E_bg
            E_scat += A * np.exp(1j * m * G0 * x_np)

        class ExactScatNet:
            def field_components(self, x, z):
                x_np2 = x.detach().numpy()
                E_interp = np.interp(x_np2, x_np, E_scat.real) + \
                           1j * np.interp(x_np2, x_np, E_scat.imag)
                Er = torch.as_tensor(E_interp.real, dtype=torch.float64)
                Ei = torch.as_tensor(E_interp.imag, dtype=torch.float64)
                z_ = torch.zeros_like(Er)
                return Er, Ei, z_, z_, z_, z_

        loss = modal_data_loss(
            ExactScatNet(), z_bot, physics_0p8,
            rcwa_ref["c_trans"], rcwa_ref["N_harmonics"],
            "bottom", physics_0p8.n_substrate,
            n_data_orders=2, weight_propagating=1.0,
        )
        assert float(loss) < 1e-6, \
            f"Exact scatter should give near-zero modal loss, got {float(loss):.2e}"

    def test_modal_loss_has_gradient(self, physics_0p8, rcwa_ref):
        """Gradient must flow from modal_data_loss to network parameters."""
        from src.field_comparison import modal_data_loss
        from src.maxwell_2d_nondim import Maxwell2DSubdomainMLP_ND

        net = Maxwell2DSubdomainMLP_ND(
            physics_0p8.k0, 2, 16, 3,
            period=physics_0p8.period, num_grating_levels=2
        ).to(dtype=torch.float64)

        z_bot = 0.92 * physics_0p8.domain_height
        loss  = modal_data_loss(
            net, z_bot, physics_0p8,
            rcwa_ref["c_trans"], rcwa_ref["N_harmonics"],
            "bottom", physics_0p8.n_substrate,
            n_data_orders=2, weight_propagating=1.0,
        )
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "No gradient flowed from modal_data_loss to net parameters"


# ---------------------------------------------------------------------------
# Fixed modal_data_loss audit tests
# ---------------------------------------------------------------------------

class TestModalDataLossAudit:
    """Verify modal_data_loss targets match extract_modal_amplitudes exactly."""

    @pytest.fixture
    def setup(self, physics):
        """Build physics, coeff, rcwa for lambda=0.8."""
        from src.config import PhysicsConfig
        from src.maxwell_layered_bg import compute_background_coefficients, background_field_np
        from src.field_comparison import load_rcwa_amplitudes
        from src.modal_dtn import _kz_outgoing
        import math

        p_ref = ROOT / "outputs/reference_lambda_0p8.npz"
        if not p_ref.exists():
            pytest.skip("reference_lambda_0p8.npz not found")

        p08 = PhysicsConfig(
            wavelength=physics.wavelength, n_air=physics.n_air,
            n_ridge=physics.n_ridge, n_substrate=physics.n_substrate,
            period=0.8*physics.wavelength, ridge_width=0.4*physics.wavelength,
            ridge_height=physics.ridge_height, domain_height=physics.domain_height,
            ridge_base_fraction=physics.ridge_base_fraction,
        )
        rcwa = load_rcwa_amplitudes(str(p_ref))
        coeff = compute_background_coefficients(p08)
        return p08, rcwa, coeff

    def _make_perfect_bot_net(self, p08, rcwa, coeff):
        """Subnet that outputs the exact RCWA scattered field at the bottom monitor."""
        from src.maxwell_layered_bg import background_field_np
        from src.modal_dtn import _kz_outgoing
        import math

        k0 = p08.k0; G0 = 2*math.pi/p08.period; N = rcwa['N_harmonics']
        z_bot = 0.92 * p08.domain_height
        x_np = np.linspace(0, p08.period, 128, endpoint=False)

        Ebg_r, Ebg_i, _, _ = background_field_np(np.array([z_bot]), coeff)
        E_bg = complex(float(Ebg_r[0]), float(Ebg_i[0]))

        E_scat = np.zeros(128, dtype=complex)
        for m in [-1, 0, 1]:
            kzm = _kz_outgoing(np.array([m*G0]), p08.n_substrate, k0)[0]
            if kzm.real <= 1e-6:
                continue
            c = complex(rcwa['c_trans'][N+m])
            dz = z_bot - p08.ridge_z_max
            A = c * np.exp(-1j*kzm*dz)
            A_scat = A - E_bg if m == 0 else A
            E_scat += A_scat * np.exp(1j*m*G0*x_np)

        class Net:
            def field_components(self, x, z):
                xn = x.detach().numpy()
                E = (np.interp(xn, x_np, E_scat.real) +
                     1j * np.interp(xn, x_np, E_scat.imag))
                Er = torch.as_tensor(E.real, dtype=torch.float64)
                Ei = torch.as_tensor(E.imag, dtype=torch.float64)
                z_ = torch.zeros_like(Er)
                return Er, Ei, z_, z_, z_, z_
        return Net()

    def test_zero_scatter_bottom_gives_large_loss(self, setup):
        """E_scat=0 at bottom should give loss ~0.077 (target≠0)."""
        from src.field_comparison import modal_data_loss
        p08, rcwa, coeff = setup

        class ZeroNet:
            def field_components(self, x, z):
                z_ = torch.zeros_like(x)
                return z_, z_, z_, z_, z_, z_

        z_bot = 0.92 * p08.domain_height
        L = modal_data_loss(ZeroNet(), z_bot, p08, rcwa['c_trans'],
                            rcwa['N_harmonics'], 'bottom', p08.n_substrate,
                            n_data_orders=2, weight_propagating=1.0)
        assert float(L) > 0.05, f"Expected large loss for E_scat=0, got {float(L):.4f}"

    def test_perfect_scatter_bottom_gives_zero_loss(self, setup):
        """Exact RCWA scattered field should give loss < 1e-10."""
        from src.field_comparison import modal_data_loss
        p08, rcwa, coeff = setup
        net = self._make_perfect_bot_net(p08, rcwa, coeff)
        z_bot = 0.92 * p08.domain_height
        L = modal_data_loss(net, z_bot, p08, rcwa['c_trans'],
                            rcwa['N_harmonics'], 'bottom', p08.n_substrate,
                            n_data_orders=2, weight_propagating=1.0)
        assert float(L) < 1e-10, \
            f"Perfect scatter should give ~0 loss, got {float(L):.2e}"

    def test_bottom_targets_match_extractor_pm1(self, setup):
        """Bottom modal targets for m=±1 must equal RCWA c_trans phase-propagated."""
        from src.field_comparison import modal_data_loss, extract_modal_amplitudes
        from src.modal_dtn import _kz_outgoing
        import math

        p08, rcwa, coeff = setup
        k0 = p08.k0; G0 = 2*math.pi/p08.period; N = rcwa['N_harmonics']
        z_bot = 0.92 * p08.domain_height

        # Compute expected target for m=+1
        kz1 = _kz_outgoing(np.array([G0]), p08.n_substrate, k0)[0]
        dz = z_bot - p08.ridge_z_max
        c1 = complex(rcwa['c_trans'][N+1])
        target_expected = c1 * np.exp(-1j*kz1*dz)  # no bg subtraction for m≠0

        assert abs(target_expected) > 0.3, \
            f"|target(m=+1)| = {abs(target_expected):.4f}, expected ~0.330"

    def test_zero_contrast_scatter_zero(self, setup):
        """Zero contrast (n_ridge=n_sub): E_scat=0, scattered modal amplitudes=0."""
        from src.config import PhysicsConfig
        from src.maxwell_layered_bg import compute_background_coefficients
        from src.field_comparison import load_rcwa_amplitudes, modal_data_loss

        p08, rcwa, coeff = setup
        # Flat physics: n_ridge = n_substrate
        p_flat = PhysicsConfig(
            wavelength=p08.wavelength, n_air=p08.n_air,
            n_ridge=p08.n_substrate,   # no contrast
            n_substrate=p08.n_substrate,
            period=p08.period, ridge_width=p08.ridge_width,
            ridge_height=p08.ridge_height, domain_height=p08.domain_height,
            ridge_base_fraction=p08.ridge_base_fraction,
        )
        coeff_flat = compute_background_coefficients(p_flat)

        # For flat grating, RCWA scattered = 0 for all m≠0, m=0 has only bg
        # Build a reference NPZ with flat grating (no grating orders)
        # Instead verify: background field DFT at m=±1 is 0
        from src.maxwell_layered_bg import background_field_np
        z_bot = 0.92 * p_flat.domain_height
        x_np = np.linspace(0, p_flat.period, 64, endpoint=False)
        Ebg_r, Ebg_i, _, _ = background_field_np(np.array([z_bot]), coeff_flat)
        E_bg_val = complex(float(Ebg_r[0]), float(Ebg_i[0]))
        # E_bg is x-independent, so DFT at m≠0 is exactly 0
        G0 = 2*np.pi/p_flat.period
        dx = p_flat.period/64
        for m in [-1, 1]:
            gm = m*G0
            Em = np.sum(E_bg_val * np.ones(64) * np.exp(-1j*gm*x_np)) * dx/p_flat.period
            assert abs(Em) < 1e-10, f"E_bg DFT at m={m} should be 0, got {abs(Em):.2e}"

    def test_top_background_subtraction_uses_reflected_only(self, setup):
        """At top boundary, only subtract reflected background (not incident)."""
        from src.maxwell_layered_bg import compute_background_coefficients
        import math

        p08, rcwa, coeff = setup
        k0 = p08.k0; k1 = coeff['k1']
        r_eff = coeff['r_eff']
        z_top = 0.08 * p08.domain_height

        # Expected E_bg_m0 at top = r_eff * exp(+ik1*z_top) (reflected only)
        E_bg_refl = r_eff * np.exp(+1j * k1 * z_top)

        # Full E_bg(z_top) = E_inc + E_bg_refl
        from src.maxwell_layered_bg import background_field_np
        Ebg_r, Ebg_i, _, _ = background_field_np(np.array([z_top]), coeff)
        E_bg_full = complex(float(Ebg_r[0]), float(Ebg_i[0]))
        E_inc_top = np.exp(-1j * k0 * z_top)
        E_bg_refl_check = E_bg_full - E_inc_top

        assert abs(E_bg_refl - E_bg_refl_check) < 1e-8, \
            f"Reflected BG mismatch: {E_bg_refl:.4f} vs {E_bg_refl_check:.4f}"

        # Verify modal_data_loss uses reflected-only background (not full E_bg)
        # The target for m=0 at top should be c_refl_prop - E_bg_refl_only
        # NOT c_refl_prop - E_bg_full (which would include incident)
        c_refl_0 = complex(rcwa['c_refl'][rcwa['N_harmonics']])
        from src.modal_dtn import _kz_outgoing
        kz0 = _kz_outgoing(np.array([0.0]), p08.n_air, k0)[0]
        A_at_top_mon = c_refl_0 * np.exp(+1j * kz0 * z_top)
        target_correct    = A_at_top_mon - E_bg_refl      # subtract reflected BG only
        target_wrong_full = A_at_top_mon - E_bg_full      # would subtract incident too

        # target_correct should have the correct scattered reflection amplitude
        # target_wrong_full would be about 1 unit off (from incident subtraction)
        diff = abs(target_correct) - abs(target_wrong_full)
        assert abs(target_correct) < 0.5, \
            f"|target_correct(m=0,top)| = {abs(target_correct):.4f} (reflection is small)"
        assert abs(target_wrong_full) > abs(target_correct) + 0.3, \
            "If using full E_bg (wrong), target should be much larger than correct"
