"""Tests for geometry and permittivity map."""

from __future__ import annotations

import torch

from src.config import PhysicsConfig
from src.geometry import epsilon_r, normalize_coordinates


def test_epsilon_air_above_grating() -> None:
    physics = PhysicsConfig()
    x = torch.tensor([0.5])
    z = torch.tensor([0.1])  # well above substrate
    eps = epsilon_r(x, z, physics)
    assert torch.allclose(eps, torch.tensor([physics.eps_air]))


def test_epsilon_substrate_below_base() -> None:
    physics = PhysicsConfig()
    x = torch.tensor([0.1])  # outside ridge horizontally
    z = torch.tensor([physics.domain_height * 0.9])
    eps = epsilon_r(x, z, physics)
    assert torch.allclose(eps, torch.tensor([physics.eps_substrate]))


def test_epsilon_ridge_interior() -> None:
    physics = PhysicsConfig()
    x = torch.tensor([physics.period / 2.0])
    z = torch.tensor([physics.ridge_base_z + physics.ridge_height / 2.0])
    eps = epsilon_r(x, z, physics)
    assert torch.allclose(eps, torch.tensor([physics.eps_ridge]))


def test_normalize_coordinates_shape() -> None:
    physics = PhysicsConfig()
    x = torch.linspace(0, physics.period, 10)
    z = torch.linspace(0, physics.domain_height, 20)
    x_n, z_n = normalize_coordinates(x, z, physics)
    assert x_n.shape == x.shape
    assert z_n.shape == z.shape
    assert float(x_n.min()) >= -1.0 and float(x_n.max()) <= 1.0
