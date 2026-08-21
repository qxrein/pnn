"""Collocation point sampling for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.stats import qmc

from src.geometry import interface_mask

if TYPE_CHECKING:
    from src.config import PhysicsConfig, SamplingConfig


@dataclass
class SampleBatch:
    """Container for collocation points used during training."""

    interior_x: torch.Tensor
    interior_z: torch.Tensor
    periodic_left_x: torch.Tensor
    periodic_left_z: torch.Tensor
    periodic_right_x: torch.Tensor
    periodic_right_z: torch.Tensor
    top_x: torch.Tensor
    top_z: torch.Tensor
    bottom_x: torch.Tensor
    bottom_z: torch.Tensor
    interface_x: torch.Tensor
    interface_z: torch.Tensor


def _unit_samples(n: int, dim: int, sampler_type: str, seed: int) -> np.ndarray:
    """Generate points in [0, 1]^dim."""
    sampler_type = sampler_type.lower()
    if sampler_type == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))
    if sampler_type == "sobol":
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        return engine.random(n)
    raise ValueError(f"Unknown sampler_type: {sampler_type}")


def _affine_map(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


def sample_points(
    physics: PhysicsConfig,
    sampling: SamplingConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> SampleBatch:
    """Sample collocation points for all boundary and interior sets."""
    margin = physics.interface_margin

    # Interior: reject points near interfaces
    interior_x_list: list[np.ndarray] = []
    interior_z_list: list[np.ndarray] = []
    needed = sampling.n_interior
    batch_size = needed
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(interior_x_list) * batch_size < needed and attempts < 50:
        u = _unit_samples(batch_size, 2, sampling.sampler_type, seed + attempts)
        x_c = _affine_map(u[:, 0], 0.0, physics.period)
        z_c = _affine_map(u[:, 1], 0.0, physics.domain_height)
        x_t = torch.as_tensor(x_c, dtype=dtype)
        z_t = torch.as_tensor(z_c, dtype=dtype)
        mask = interface_mask(x_t, z_t, physics, margin)
        keep = ~mask.numpy()
        interior_x_list.append(x_c[keep])
        interior_z_list.append(z_c[keep])
        attempts += 1

    interior_x = np.concatenate(interior_x_list)[:needed]
    interior_z = np.concatenate(interior_z_list)[:needed]

    # Periodic boundaries (same z on left/right)
    u_p = _unit_samples(sampling.n_periodic, 1, sampling.sampler_type, seed + 1)
    periodic_z = _affine_map(u_p[:, 0], 0.0, physics.domain_height)
    periodic_left_x = np.zeros_like(periodic_z)
    periodic_right_x = np.full_like(periodic_z, physics.period)

    # Top boundary z=0
    u_top = _unit_samples(sampling.n_top, 1, sampling.sampler_type, seed + 2)
    top_x = _affine_map(u_top[:, 0], 0.0, physics.period)
    top_z = np.zeros_like(top_x)

    # Bottom boundary z=domain_height
    u_bot = _unit_samples(sampling.n_bottom, 1, sampling.sampler_type, seed + 3)
    bottom_x = _affine_map(u_bot[:, 0], 0.0, physics.period)
    bottom_z = np.full_like(bottom_x, physics.domain_height)

    # Interface points
    interface_x_list: list[np.ndarray] = []
    interface_z_list: list[np.ndarray] = []
    if sampling.n_interface > 0:
        u_if = _unit_samples(sampling.n_interface * 4, 2, sampling.sampler_type, seed + 4)
        x_if = _affine_map(u_if[:, 0], 0.0, physics.period)
        z_if = _affine_map(u_if[:, 1], 0.0, physics.domain_height)
        x_t = torch.as_tensor(x_if, dtype=dtype)
        z_t = torch.as_tensor(z_if, dtype=dtype)
        mask = interface_mask(x_t, z_t, physics, margin * 0.5)
        interface_x_list.append(x_if[mask.numpy()])
        interface_z_list.append(z_if[mask.numpy()])

    if interface_x_list:
        interface_x = np.concatenate(interface_x_list)[: sampling.n_interface]
        interface_z = np.concatenate(interface_z_list)[: sampling.n_interface]
    else:
        interface_x = np.empty(0)
        interface_z = np.empty(0)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, device=device, dtype=dtype)

    return SampleBatch(
        interior_x=_t(interior_x),
        interior_z=_t(interior_z),
        periodic_left_x=_t(periodic_left_x),
        periodic_left_z=_t(periodic_z),
        periodic_right_x=_t(periodic_right_x),
        periodic_right_z=_t(periodic_z),
        top_x=_t(top_x),
        top_z=_t(top_z),
        bottom_x=_t(bottom_x),
        bottom_z=_t(bottom_z),
        interface_x=_t(interface_x),
        interface_z=_t(interface_z),
    )
