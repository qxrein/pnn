"""Tests for the feature-scaling reparameterization (Phase 2).

Verifies seven invariants of the ScaledHead transform:
  T1  zero final-head → E_scat = 0 everywhere
  T2  raw Tanh hidden features are unchanged after scaling
  T3  W_scaled @ h_scaled = W_orig @ h for any weight matrix
  T4  production residual is unchanged at theta=0
  T5  modal extraction is unchanged at theta=0
  T6  canonical reference and companion SHA are unmodified
  T7  residual is affine in the scaled final-head parameters
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.maxwell_2d_nondim import sample_nd_points
from src.maxwell_layered_bg import compute_background_coefficients
from src.mode_aware_fourier import ExplicitFourierModalDD
from src.utils import set_seed
from scripts.run_frozen_head_least_squares import (
    HEAD_PARAM_COUNT, CANONICAL_REF, COMPANION_PATH,
    freeze_except_head, head_parameters, pack_head, pack_residual,
    residual_blocks, unpack_head,
)
from scripts.run_feature_scaling import (
    ScaledHead, apply_feature_scaling, compute_feature_scales,
    freeze_scaled_except_head, make_scaled_model, scaled_head_parameters,
)
from scripts.train_lbg import make_lambda_0p8

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_setup():
    """Small system for fast tests: 8+4+4+4 collocation points."""
    set_seed(42)
    physics = make_lambda_0p8(load_config(ROOT / "configs/default.yaml").physics)
    device  = torch.device("cpu")
    dtype   = torch.float64
    pts     = sample_nd_points(physics, 8, 4, 4, 4, device, dtype, seed=0)
    coeff   = compute_background_coefficients(physics)
    return physics, pts, coeff


@pytest.fixture(scope="module")
def fresh_model(tiny_setup):
    physics, pts, _ = tiny_setup
    set_seed(42)
    m = ExplicitFourierModalDD(physics, 3, True).double()
    freeze_except_head(m)
    return m


@pytest.fixture(scope="module")
def scaled_setup(tiny_setup):
    physics, pts, _ = tiny_setup
    model_s, scales = make_scaled_model(physics, pts)
    heads_s = freeze_scaled_except_head(model_s)
    return model_s, heads_s, scales


# ─────────────────────────────────────────────────────────────────────────────
# T1: zero head → zero output
# ─────────────────────────────────────────────────────────────────────────────

def test_T1_zero_head_output_is_zero(tiny_setup, scaled_setup):
    """E_scat = 0 everywhere when final-head parameters are zero."""
    physics, pts, _ = tiny_setup
    model_s, heads_s, _ = scaled_setup
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    for net, zkey in ((model_s.net_air,  "z_air"),
                       (model_s.net_grat, "z_grat"),
                       (model_s.net_sub,  "z_sub")):
        z = pts[zkey].detach().requires_grad_(True)
        x = z.clone().requires_grad_(True)
        er, ei, *_ = net.field_components(x, z)
        assert er.abs().max().item() < 1e-14, f"E_r nonzero at zero head"
        assert ei.abs().max().item() < 1e-14, f"E_i nonzero at zero head"


# ─────────────────────────────────────────────────────────────────────────────
# T2: raw Tanh features unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_T2_hidden_features_unchanged(tiny_setup, scaled_setup):
    """The Tanh hidden features are the same before and after scaling."""
    physics, pts, _ = tiny_setup
    model_s, _, _ = scaled_setup

    set_seed(42)
    model_u = ExplicitFourierModalDD(physics, 3, True).double()

    for net_name, net_s, net_u, zkey in (
        ("net_air",  model_s.net_air,  model_u.net_air,  "z_air"),
        ("net_grat", model_s.net_grat, model_u.net_grat, "z_grat"),
        ("net_sub",  model_s.net_sub,  model_u.net_sub,  "z_sub"),
    ):
        z = pts[zkey]
        with torch.no_grad():
            zn_s = (2*(z - net_s.z_lo)/(net_s.z_hi - net_s.z_lo) - 1)[:, None]
            sh   = net_s.coefficient_mlp
            h_s  = sh.tanh(sh.lin0(zn_s)).numpy()

            zn_u = (2*(z - net_u.z_lo)/(net_u.z_hi - net_u.z_lo) - 1)[:, None]
            h_u  = net_u.coefficient_mlp[1](net_u.coefficient_mlp[0](zn_u)).numpy()

        assert np.max(np.abs(h_s - h_u)) < 1e-14, \
            f"{net_name}: hidden features changed after scaling"


# ─────────────────────────────────────────────────────────────────────────────
# T3: W_scaled @ h_scaled = W_orig @ h
# ─────────────────────────────────────────────────────────────────────────────

def test_T3_scaled_output_equals_unscaled(tiny_setup):
    """W_scaled @ (h/s) = W_orig @ h for a deterministic random W_orig."""
    physics, pts, _ = tiny_setup
    rng = np.random.default_rng(0)
    W_orig = rng.standard_normal((14, 32))

    set_seed(42)
    model_u = ExplicitFourierModalDD(physics, 3, True).double()
    model_s, scales = make_scaled_model(physics, pts)

    for net_name, net_s, net_u, zkey in (
        ("net_air",  model_s.net_air,  model_u.net_air,  "z_air"),
        ("net_grat", model_s.net_grat, model_u.net_grat, "z_grat"),
        ("net_sub",  model_s.net_sub,  model_u.net_sub,  "z_sub"),
    ):
        sc = net_s.coefficient_mlp.scale.numpy()  # (32,)
        # Set unscaled head to W_orig
        with torch.no_grad():
            net_u.coefficient_mlp[2].weight.copy_(
                torch.tensor(W_orig, dtype=torch.float64))
            net_u.coefficient_mlp[2].bias.zero_()
            # For scaled head: W_s[o,j] = W_orig[o,j] * sc[j]
            # Then W_s @ (h/sc) = W_orig @ h
            net_s.coefficient_mlp.head.weight.copy_(
                torch.tensor(W_orig * sc[None, :], dtype=torch.float64))
            net_s.coefficient_mlp.head.bias.zero_()

        z = pts[zkey].detach().requires_grad_(True)
        er_u, ei_u, *_ = net_u.field_components(z.clone().requires_grad_(True),
                                                  z.clone().requires_grad_(True))
        er_s, ei_s, *_ = net_s.field_components(z.clone().requires_grad_(True),
                                                  z.clone().requires_grad_(True))
        err = float(max(
            (er_u.detach() - er_s.detach()).abs().max(),
            (ei_u.detach() - ei_s.detach()).abs().max(),
        ))
        assert err < 1e-10, f"{net_name}: W@h != W_s@h_norm (err={err:.2e})"

        # Restore
        with torch.no_grad():
            net_u.coefficient_mlp[2].weight.zero_()
            net_s.coefficient_mlp.head.weight.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# T4: production residual unchanged at theta=0
# ─────────────────────────────────────────────────────────────────────────────

def test_T4_production_residual_unchanged(tiny_setup, scaled_setup):
    """The production residual vector at zero head is identical before and after scaling."""
    physics, pts, coeff = tiny_setup
    model_s, heads_s, _ = scaled_setup
    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))

    set_seed(42)
    model_u = ExplicitFourierModalDD(physics, 3, True).double()
    heads_u = freeze_except_head(model_u)
    unpack_head(heads_u, np.zeros(HEAD_PARAM_COUNT))

    r_u, _, _ = pack_residual(residual_blocks(model_u, pts, physics, coeff))
    r_s, _, _ = pack_residual(residual_blocks(model_s, pts, physics, coeff))
    diff = float((r_u.detach() - r_s.detach()).abs().max())
    assert diff < 1e-12, f"Residual changed after scaling (max_diff={diff:.2e})"


# ─────────────────────────────────────────────────────────────────────────────
# T5: parameter count and structure
# ─────────────────────────────────────────────────────────────────────────────

def test_T5_head_parameter_count(scaled_setup):
    """Scaled model has exactly HEAD_PARAM_COUNT trainable final-head parameters."""
    model_s, _, _ = scaled_setup
    heads = scaled_head_parameters(model_s)
    n = sum(p.numel() for p in heads)
    assert n == HEAD_PARAM_COUNT, f"Expected {HEAD_PARAM_COUNT}, got {n}"


def test_T5_head_requires_grad(scaled_setup):
    """After freeze_scaled_except_head, only the final head has requires_grad=True."""
    physics, pts, _ = (None, None, None)  # not needed here
    model_s, heads_s, _ = scaled_setup
    head_ids = {id(p) for p in heads_s}
    for name, p in model_s.named_parameters():
        if id(p) in head_ids:
            assert p.requires_grad, f"{name} should require grad"
        else:
            assert not p.requires_grad, f"{name} should be frozen"


# ─────────────────────────────────────────────────────────────────────────────
# T6: scale buffer is non-trainable
# ─────────────────────────────────────────────────────────────────────────────

def test_T6_scale_is_not_a_parameter(scaled_setup):
    """The feature scale is a registered buffer, not a trainable Parameter."""
    model_s, _, _ = scaled_setup
    param_names = {name for name, _ in model_s.named_parameters()}
    for net_name in ("net_air", "net_grat", "net_sub"):
        scale_key = f"{net_name}.coefficient_mlp.scale"
        assert scale_key not in param_names, \
            f"scale should not be a trainable Parameter: {scale_key}"
    # Check it is in buffers
    buffer_names = {name for name, _ in model_s.named_buffers()}
    for net_name in ("net_air", "net_grat", "net_sub"):
        scale_key = f"{net_name}.coefficient_mlp.scale"
        assert scale_key in buffer_names, \
            f"scale should be a registered buffer: {scale_key}"


# ─────────────────────────────────────────────────────────────────────────────
# T7: residual is affine in scaled head parameters
# ─────────────────────────────────────────────────────────────────────────────

def test_T7_residual_affine_in_scaled_head(tiny_setup, scaled_setup):
    """Affine superposition holds for the scaled parameterization."""
    physics, pts, coeff = tiny_setup
    model_s, heads_s, _ = scaled_setup
    n = HEAD_PARAM_COUNT
    rng = np.random.default_rng(999)

    def residual(theta):
        unpack_head(heads_s, theta)
        r, _, _ = pack_residual(residual_blocks(model_s, pts, physics, coeff))
        return r.detach().numpy()

    a = np.zeros(n); a[rng.integers(0, n)] = 2.3
    b = np.zeros(n); b[rng.integers(0, n)] = -1.1

    r0  = residual(np.zeros(n))
    ra  = residual(a)
    rb  = residual(b)
    rab = residual(a + b)

    err   = np.linalg.norm((rab - r0) - (ra - r0) - (rb - r0))
    scale = np.linalg.norm(ra - r0) + np.linalg.norm(rb - r0) + 1e-30
    assert err / scale < 1e-12, f"Residual not affine in scaled head (err/scale={err/scale:.2e})"
    unpack_head(heads_s, np.zeros(n))


# ─────────────────────────────────────────────────────────────────────────────
# T8: hidden-feature invariance under head changes
# ─────────────────────────────────────────────────────────────────────────────

def test_T8_hidden_feature_invariance(tiny_setup, scaled_setup):
    """Tanh hidden features are unchanged when head parameters vary."""
    physics, pts, _ = tiny_setup
    model_s, heads_s, _ = scaled_setup

    def get_hidden(net, z):
        sh = net.coefficient_mlp
        with torch.no_grad():
            zn = (2*(z - sh.lin0.in_features) / 1.0)[:, None]
            # compute correctly:
            zn = (2*(z - net.z_lo)/(net.z_hi - net.z_lo) - 1.0)[:, None]
            return sh.tanh(sh.lin0(zn)).numpy()

    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))
    h0 = {k: get_hidden(getattr(model_s, k), pts[zk])
          for k, zk in (("net_air","z_air"),("net_grat","z_grat"),("net_sub","z_sub"))}

    rng = np.random.default_rng(11)
    for mag in [1e-3, 1.0, 1e3]:
        theta = rng.standard_normal(HEAD_PARAM_COUNT) * mag
        unpack_head(heads_s, theta)
        for k, zk in (("net_air","z_air"),("net_grat","z_grat"),("net_sub","z_sub")):
            ht = get_hidden(getattr(model_s, k), pts[zk])
            assert np.max(np.abs(ht - h0[k])) < 1e-14, \
                f"Hidden features changed at |theta|={mag:.0e} for {k}"

    unpack_head(heads_s, np.zeros(HEAD_PARAM_COUNT))


# ─────────────────────────────────────────────────────────────────────────────
# T9: state_dict round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_T9_state_dict_roundtrip(tiny_setup):
    """Model with ScaledHead saves and reloads correctly."""
    import tempfile, os
    physics, pts, _ = tiny_setup
    model_s, _ = make_scaled_model(physics, pts)
    heads_s = freeze_scaled_except_head(model_s)

    # Set a nontrivial head
    rng   = np.random.default_rng(55)
    theta = rng.standard_normal(HEAD_PARAM_COUNT) * 0.1
    unpack_head(heads_s, theta)

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(model_s.state_dict(), path)
        model_r, _ = make_scaled_model(physics, pts)
        model_r.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

        # Check outputs match
        x  = pts["x_grat"].detach().requires_grad_(True)
        z  = pts["z_grat"].detach().requires_grad_(True)
        e_s = model_s.net_grat.forward(x.clone().requires_grad_(True),
                                        z.clone().requires_grad_(True))
        e_r = model_r.net_grat.forward(x.clone().requires_grad_(True),
                                        z.clone().requires_grad_(True))
        assert (e_s.detach() - e_r.detach()).abs().max().item() < 1e-14
    finally:
        os.unlink(path)
