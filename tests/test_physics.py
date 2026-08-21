"""Tests for Helmholtz physics residual and loss dictionary."""

from __future__ import annotations

import torch

from src.config import LossWeights, PhysicsConfig, SamplingConfig
from src.losses import compute_losses
from src.model import FieldMLP
from src.config import ModelConfig
from src.physics import helmholtz_residual, physics_loss
from src.sampling import sample_points


def test_physics_residual_shape() -> None:
    physics = PhysicsConfig()
    dtype = torch.float64
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics).to(dtype=dtype)
    x = torch.linspace(0.2, physics.period - 0.2, 6, dtype=dtype, requires_grad=True)
    z = torch.linspace(0.2, physics.domain_height - 0.2, 6, dtype=dtype, requires_grad=True)
    res_r, res_i = helmholtz_residual(model, x, z, physics)
    assert res_r.shape == x.shape
    assert res_i.shape == x.shape
    assert torch.isfinite(res_r).all()
    assert torch.isfinite(res_i).all()


def test_physics_loss_scalar() -> None:
    physics = PhysicsConfig()
    dtype = torch.float64
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics).to(dtype=dtype)
    x = torch.linspace(0.2, physics.period - 0.2, 6, dtype=dtype)
    z = torch.linspace(0.2, physics.domain_height - 0.2, 6, dtype=dtype)
    loss = physics_loss(model, x, z, physics)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_dictionary_keys() -> None:
    physics = PhysicsConfig()
    sampling = SamplingConfig(n_interior=64, n_periodic=16, n_top=8, n_bottom=8, n_interface=0)
    device = torch.device("cpu")
    dtype = torch.float64
    samples = sample_points(physics, sampling, device, dtype, seed=0)
    model = FieldMLP(ModelConfig(hidden_layers=2, hidden_width=16), physics).to(dtype=dtype)
    weights = LossWeights()
    losses = compute_losses(model, samples, physics, weights)
    expected = {
        "pde", "periodic", "top", "bottom", "interface", "data", "total",
        "weighted_pde", "weighted_periodic", "weighted_top", "weighted_bottom",
        "weighted_interface", "weighted_data",
    }
    assert expected.issubset(set(losses.keys()))
