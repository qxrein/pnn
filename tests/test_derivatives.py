"""Tests for autograd spatial derivatives."""

from __future__ import annotations

import torch

from src.config import ModelConfig, PhysicsConfig
from src.derivatives import field_laplacian, prepare_coords, second_derivative
from src.model import FieldMLP


def test_model_output_shape() -> None:
    physics = PhysicsConfig()
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics)
    x = torch.linspace(0, physics.period, 32)
    z = torch.linspace(0, physics.domain_height, 32)
    out = model(x, z)
    assert out.shape == (32, 2)


def test_first_and_second_derivatives_finite() -> None:
    physics = PhysicsConfig()
    dtype = torch.float64
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics).to(dtype=dtype)
    x = torch.linspace(0.1, physics.period - 0.1, 8, dtype=dtype)
    z = torch.linspace(0.1, physics.domain_height - 0.1, 8, dtype=dtype)
    x, z = prepare_coords(x, z)
    e_real, e_imag = model.field_components(x, z)
    d2x = second_derivative(e_real, x)
    d2z = second_derivative(e_real, z)
    assert d2x.shape == x.shape
    assert d2z.shape == z.shape
    assert torch.isfinite(d2x).all()
    assert torch.isfinite(d2z).all()


def test_laplacian_shapes() -> None:
    physics = PhysicsConfig()
    dtype = torch.float64
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics).to(dtype=dtype)
    x = torch.linspace(0.1, physics.period - 0.1, 4, dtype=dtype, requires_grad=True)
    z = torch.linspace(0.1, physics.domain_height - 0.1, 4, dtype=dtype, requires_grad=True)
    e_real, e_imag = model.field_components(x, z)
    result = field_laplacian(e_real, e_imag, x, z)
    assert len(result) == 6
    for t in result:
        assert t.shape == x.shape
        assert torch.isfinite(t).all()
