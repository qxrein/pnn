"""Automatic-differentiation utilities for spatial derivatives."""

from __future__ import annotations

import torch


def first_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute dy/dx with autograd, preserving graph for higher derivatives."""
    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad


def second_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute d²y/dx²."""
    dy = first_derivative(y, x)
    # If dy has no gradient function (e.g. was zero from allow_unused),
    # the second derivative is also zero.
    if dy.grad_fn is None:
        return torch.zeros_like(x)
    grad2 = torch.autograd.grad(
        dy,
        x,
        grad_outputs=torch.ones_like(dy),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad2 is None:
        return torch.zeros_like(x)
    return grad2


def field_laplacian(
    e_real: torch.Tensor,
    e_imag: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Laplacian components for real and imaginary field parts.

    Returns
    -------
    d2E_real_dx2, d2E_real_dz2, d2E_imag_dx2, d2E_imag_dz2,
    lap_real, lap_imag
    """
    d2_e_real_dx2 = second_derivative(e_real, x)
    d2_e_real_dz2 = second_derivative(e_real, z)
    d2_e_imag_dx2 = second_derivative(e_imag, x)
    d2_e_imag_dz2 = second_derivative(e_imag, z)
    lap_real = d2_e_real_dx2 + d2_e_real_dz2
    lap_imag = d2_e_imag_dx2 + d2_e_imag_dz2
    return d2_e_real_dx2, d2_e_real_dz2, d2_e_imag_dx2, d2_e_imag_dz2, lap_real, lap_imag


def prepare_coords(x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure coordinate tensors require gradients."""
    x = x.detach().clone().requires_grad_(True)
    z = z.detach().clone().requires_grad_(True)
    return x, z
