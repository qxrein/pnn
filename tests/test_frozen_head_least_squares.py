"""Frozen final-head parameterization and affine residual tests."""

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
    HEAD_PARAM_COUNT,
    freeze_except_head,
    pack_head,
    pack_residual,
    residual_blocks,
    unpack_head,
)
from scripts.train_lbg import make_lambda_0p8

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_system():
    set_seed(42)
    physics = make_lambda_0p8(load_config(ROOT / "configs/default.yaml").physics)
    device = torch.device("cpu")
    dtype = torch.float64
    pts = sample_nd_points(physics, 8, 4, 4, 4, device, dtype, seed=0)
    coeff = compute_background_coefficients(physics)
    model = ExplicitFourierModalDD(physics, 3, True).double()
    heads = freeze_except_head(model)
    return model, heads, pts, physics, coeff


def test_frozen_head_parameter_count(tiny_system):
    _, heads, *_ = tiny_system
    assert sum(p.numel() for p in heads) == HEAD_PARAM_COUNT


def test_hidden_layers_are_frozen(tiny_system):
    model, heads, *_ = tiny_system
    head_ids = {id(p) for p in heads}
    for name, parameter in model.named_parameters():
        if id(parameter) in head_ids:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad, name


def test_residual_affine_in_final_head(tiny_system):
    model, heads, pts, physics, coeff = tiny_system
    n = HEAD_PARAM_COUNT
    rng = np.random.default_rng(0)
    a = np.zeros(n); b = np.zeros(n)
    a[rng.integers(0, n)] = 1.7
    b[rng.integers(0, n)] = -0.8

    def residual(theta):
        unpack_head(heads, theta)
        packed, _, _ = pack_residual(residual_blocks(model, pts, physics, coeff))
        return packed.detach().cpu().numpy()

    r0 = residual(np.zeros(n))
    ra = residual(a)
    rb = residual(b)
    rab = residual(a + b)
    err = np.linalg.norm((rab - r0) - (ra - r0) - (rb - r0))
    scale = np.linalg.norm(ra - r0) + np.linalg.norm(rb - r0) + 1e-30
    assert err / scale < 1e-12
    unpack_head(heads, np.zeros(n))
    assert pack_head(heads).abs().max().item() == 0.0
