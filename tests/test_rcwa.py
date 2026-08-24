"""Tests for the RCWA (Rigorous Coupled-Wave Analysis) solver.

Covers:
- Redheffer star product correctness (identity, commutativity of identity, energy)
- Single-interface S-matrix
- Propagation S-matrix
- Energy conservation for the full grating RCWA
- Correct non-zero reflection amplitudes
- Field physical validity (standing wave in air, no overflow)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_reference import (
    _kx_orders,
    _kz_branch,
    _kz_uniform,
    _s_interface,
    _s_propagate,
    _star,
    _amplitudes,
    solve_rcwa,
)
from src.config import PhysicsConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def physics():
    cfg = load_config(ROOT / "configs/default.yaml")
    return cfg.physics


@pytest.fixture
def physics_0p8(physics):
    new_period = 0.8 * physics.wavelength
    return PhysicsConfig(
        wavelength=physics.wavelength, n_air=physics.n_air,
        n_ridge=physics.n_ridge, n_substrate=physics.n_substrate,
        period=new_period, ridge_width=0.5 * new_period,
        ridge_height=physics.ridge_height, domain_height=physics.domain_height,
        ridge_base_fraction=physics.ridge_base_fraction,
        nx_visualization=64, nz_visualization=128,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_interface_S_1d(na: float, nb: float) -> np.ndarray:
    """2×2 S-matrix for a single planar interface (scalar TE, 1 mode)."""
    r_fwd = (na - nb) / (na + nb)
    t_fwd = 2 * na / (na + nb)
    r_bwd = -r_fwd
    t_bwd = 2 * nb / (na + nb)
    return np.array([[t_fwd, r_bwd], [r_fwd, t_bwd]], dtype=complex)


def make_prop_S_1d(n: float, h: float, k0: float) -> np.ndarray:
    phi = np.exp(-1j * n * k0 * h)
    return np.array([[phi, 0], [0, phi]], dtype=complex)


# ---------------------------------------------------------------------------
# _star: identity tests
# ---------------------------------------------------------------------------

class TestStarIdentity:
    """Star product with identity matrix should be a no-op."""

    def test_identity_left_1x1(self):
        n1, n2 = 1.0, 1.5
        S = make_interface_S_1d(n1, n2)
        S_I = np.eye(2, dtype=complex)
        result = _star(S_I, S)
        np.testing.assert_allclose(result, S, atol=1e-12)

    def test_identity_right_1x1(self):
        n1, n2 = 1.0, 1.5
        S = make_interface_S_1d(n1, n2)
        S_I = np.eye(2, dtype=complex)
        result = _star(S, S_I)
        np.testing.assert_allclose(result, S, atol=1e-12)

    def test_identity_left_3x3(self):
        """3-mode system: identity star S = S."""
        N = 3
        rng = np.random.default_rng(0)
        # Random unitary-like S-matrix (not physically meaningful, just algebraic)
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        B = rng.standard_normal((N, N)) * 0.1 + 1j * rng.standard_normal((N, N)) * 0.1
        C = rng.standard_normal((N, N)) * 0.1 + 1j * rng.standard_normal((N, N)) * 0.1
        D = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        S = np.block([[A, B], [C, D]])
        S_I = np.eye(2 * N, dtype=complex)
        result = _star(S_I, S)
        np.testing.assert_allclose(result, S, atol=1e-12)

    def test_identity_right_3x3(self):
        N = 3
        rng = np.random.default_rng(1)
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        B = rng.standard_normal((N, N)) * 0.1 + 1j * rng.standard_normal((N, N)) * 0.1
        C = rng.standard_normal((N, N)) * 0.1 + 1j * rng.standard_normal((N, N)) * 0.1
        D = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        S = np.block([[A, B], [C, D]])
        S_I = np.eye(2 * N, dtype=complex)
        result = _star(S, S_I)
        np.testing.assert_allclose(result, S, atol=1e-12)


# ---------------------------------------------------------------------------
# _star: physical tests
# ---------------------------------------------------------------------------

class TestStarPhysics:
    """Star product of physical S-matrices must give correct amplitudes."""

    def test_prop_then_interface_transmission(self):
        """Propagation slab + interface: transmission = t12 * exp(-i k1 h)."""
        n1, n2 = 1.0, 1.5
        k0 = 2 * np.pi
        h = 0.15
        r12 = (n1 - n2) / (n1 + n2)
        t12 = 2 * n1 / (n1 + n2)
        phi = np.exp(-1j * n1 * k0 * h)

        S_prop = make_prop_S_1d(n1, h, k0)
        S_int  = make_interface_S_1d(n1, n2)
        S_comb = _star(S_prop, S_int)

        # S[0,0] = forward transmission
        np.testing.assert_allclose(S_comb[0, 0], t12 * phi, atol=1e-12)
        # S[1,0] = reflection seen at input (phase doubled: round-trip through slab)
        np.testing.assert_allclose(S_comb[1, 0], r12 * phi**2, atol=1e-12)

    def test_prop_then_interface_energy(self):
        """Energy conservation: |r|^2 + |t|^2*(n2/n1) = 1."""
        n1, n2 = 1.0, 1.5
        k0 = 2 * np.pi
        h = 0.3
        S_prop = make_prop_S_1d(n1, h, k0)
        S_int  = make_interface_S_1d(n1, n2)
        S_comb = _star(S_prop, S_int)
        r = S_comb[1, 0]
        t = S_comb[0, 0]
        energy = abs(r)**2 + abs(t)**2 * (n2 / n1)
        np.testing.assert_allclose(energy, 1.0, atol=1e-12)

    def test_two_interfaces_energy(self):
        """Two interfaces n1->n2->n3: energy conservation."""
        n1, n2, n3 = 1.0, 1.5, 2.0
        k0 = 2 * np.pi
        h = 0.5
        S_12   = make_interface_S_1d(n1, n2)
        S_prop = make_prop_S_1d(n2, h, k0)
        S_23   = make_interface_S_1d(n2, n3)
        S_all  = _star(_star(S_12, S_prop), S_23)
        r = S_all[1, 0]
        t = S_all[0, 0]
        energy = abs(r)**2 + abs(t)**2 * (n3 / n1)
        np.testing.assert_allclose(energy, 1.0, atol=1e-12)

    def test_slab_energy_conservation(self):
        """Slab between two half-spaces: energy conserved for several thicknesses."""
        n1, n2 = 1.0, 2.5
        k0 = 2 * np.pi
        S_in  = make_interface_S_1d(n1, n2)
        S_out = make_interface_S_1d(n2, n1)
        for h in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            S_prop = make_prop_S_1d(n2, h, k0)
            S_all  = _star(_star(S_in, S_prop), S_out)
            r = S_all[1, 0]
            t = S_all[0, 0]
            energy = abs(r)**2 + abs(t)**2
            np.testing.assert_allclose(energy, 1.0, atol=1e-11,
                                       err_msg=f"h={h}: energy={energy}")

    def test_associativity(self):
        """Star product is associative: (A★B)★C = A★(B★C)."""
        n1, n2, n3, n4 = 1.0, 1.5, 2.0, 1.2
        k0 = 2 * np.pi
        S1 = make_interface_S_1d(n1, n2)
        S2 = _star(make_interface_S_1d(n2, n3), make_prop_S_1d(n3, 0.2, k0))
        S3 = make_interface_S_1d(n3, n4)
        left  = _star(_star(S1, S2), S3)
        right = _star(S1, _star(S2, S3))
        np.testing.assert_allclose(left, right, atol=1e-12)


# ---------------------------------------------------------------------------
# Full RCWA solver tests
# ---------------------------------------------------------------------------

class TestRCWAEnergyConservation:
    """Full RCWA stack must satisfy energy conservation and yield non-zero r."""

    def test_energy_conservation_period_1(self, physics):
        """period=λ grating: R+T = 1 within 0.01%."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        energy = amps["energy_check"]
        assert abs(energy - 1.0) < 1e-4, f"R+T = {energy:.8f} (not 1.0)"

    def test_energy_conservation_period_0p8(self, physics_0p8):
        """period=0.8λ grating: R+T = 1 within 0.01%."""
        x, z, Er, Ei, amps = solve_rcwa(physics_0p8, N_harmonics=25)
        energy = amps["energy_check"]
        assert abs(energy - 1.0) < 1e-4, f"R+T = {energy:.8f} (not 1.0)"

    def test_nonzero_reflection_period_1(self, physics):
        """Reflection amplitude must be non-zero for a grating."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        r0 = abs(amps["c_refl"][15])  # N=15 → index 15 is m=0
        assert r0 > 0.05, f"|r_0| = {r0:.6f} — expected > 0.05 for grating"

    def test_nonzero_transmission_period_1(self, physics):
        """Transmission amplitude must carry most of the power into substrate."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        t0 = abs(amps["c_trans"][15])
        assert t0 > 0.5, f"|t_0| = {t0:.6f} — expected > 0.5"

    def test_reflection_increases_with_contrast(self, physics):
        """Higher index contrast → higher total reflection R_total."""
        # Low contrast: n_ridge = 1.1 (close to air n=1.0)
        p_low = PhysicsConfig(
            wavelength=physics.wavelength,
            n_air=physics.n_air,
            n_ridge=1.1,
            n_substrate=physics.n_substrate,
            period=physics.period,
            ridge_width=physics.ridge_width,
            ridge_height=physics.ridge_height,
            domain_height=physics.domain_height,
            ridge_base_fraction=physics.ridge_base_fraction,
            nx_visualization=32, nz_visualization=64,
        )
        # High contrast: n_ridge = 3.0
        p_high = PhysicsConfig(
            wavelength=physics.wavelength,
            n_air=physics.n_air,
            n_ridge=3.0,
            n_substrate=physics.n_substrate,
            period=physics.period,
            ridge_width=physics.ridge_width,
            ridge_height=physics.ridge_height,
            domain_height=physics.domain_height,
            ridge_base_fraction=physics.ridge_base_fraction,
            nx_visualization=32, nz_visualization=64,
        )
        _, _, _, _, amps_low  = solve_rcwa(p_low,  N_harmonics=10)
        _, _, _, _, amps_high = solve_rcwa(p_high, N_harmonics=10)
        R_low  = amps_low["R_total"]
        R_high = amps_high["R_total"]
        assert R_low < R_high, (
            f"Higher contrast should give more reflection: "
            f"R(n=1.1)={R_low:.4f} vs R(n=3.0)={R_high:.4f}"
        )


class TestRCWAFieldPhysics:
    """RCWA field must be physically sensible."""

    def test_field_finite_no_overflow(self, physics):
        """No NaN or Inf in the field arrays."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        assert np.all(np.isfinite(Er)), "E_real contains non-finite values"
        assert np.all(np.isfinite(Ei)), "E_imag contains non-finite values"

    def test_field_finite_no_overflow_0p8(self, physics_0p8):
        """No NaN or Inf for sub-wavelength period (evanescent orders present)."""
        x, z, Er, Ei, amps = solve_rcwa(physics_0p8, N_harmonics=25)
        assert np.all(np.isfinite(Er)), "E_real contains non-finite values (evanescent overflow?)"
        assert np.all(np.isfinite(Ei)), "E_imag contains non-finite values"

    def test_standing_wave_in_air(self, physics):
        """Air region must show a standing wave: max|E| > 1, min|E| < 1."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        mag = np.sqrt(Er**2 + Ei**2)
        iz_air = z < physics.ridge_z_min
        mag_air = mag[iz_air, :]
        assert mag_air.max() > 1.05, f"max|E| in air = {mag_air.max():.4f} — no standing wave"
        assert mag_air.min() < 0.95, f"min|E| in air = {mag_air.min():.4f} — no standing wave"

    def test_not_pure_incident_wave(self, physics):
        """Field must NOT equal exp(-ik0*z) everywhere (old bug check)."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        k0 = physics.k0
        Z2d = z[:, None] * np.ones((1, len(x)))
        Er_inc = np.cos(k0 * Z2d)
        Ei_inc = -np.sin(k0 * Z2d)
        diff_r = np.max(np.abs(Er - Er_inc))
        diff_i = np.max(np.abs(Ei - Ei_inc))
        # If the old bug is present, diff would be ~0
        assert diff_r > 0.1 or diff_i > 0.1, (
            f"Field looks like incident wave only! "
            f"max|E_r - E_inc_r|={diff_r:.4e}, max|E_i - E_inc_i|={diff_i:.4e}"
        )

    def test_field_mean_magnitude_near_one(self, physics):
        """Mean |E| should be near 1 for unit-amplitude incidence."""
        x, z, Er, Ei, amps = solve_rcwa(physics, N_harmonics=15)
        mag = np.sqrt(Er**2 + Ei**2)
        mean_mag = mag.mean()
        assert 0.7 < mean_mag < 1.3, f"Mean |E| = {mean_mag:.4f} — far from 1.0"


class TestRCWAConvergence:
    """RCWA solution must converge with increasing harmonics."""

    def test_harmonic_convergence(self, physics):
        """Relative |E| difference between N=15 and N=25 < 1%."""
        _, _, Er1, Ei1, _ = solve_rcwa(physics, N_harmonics=15)
        _, _, Er2, Ei2, _ = solve_rcwa(physics, N_harmonics=25)
        mag1 = np.sqrt(Er1**2 + Ei1**2)
        mag2 = np.sqrt(Er2**2 + Ei2**2)
        rel = np.linalg.norm(mag1 - mag2) / (np.linalg.norm(mag2) + 1e-30)
        assert rel < 0.01, f"Convergence diff N=15 vs N=25: {rel:.4e} > 1%"

    def test_energy_convergence(self, physics):
        """Energy check improves or stays at <0.01% for N=25."""
        _, _, _, _, amps = solve_rcwa(physics, N_harmonics=25)
        assert abs(amps["energy_check"] - 1.0) < 1e-4
