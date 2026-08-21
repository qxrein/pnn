"""MLP field network for complex scalar Helmholtz PINN.

Coordinate handling
-------------------
The network uses two coordinate representations:

1. **Normalised** (x_n, z_n) ∈ [-1, 1] — network inputs.
   x_n = 2*x/period - 1,   z_n = 2*z/domain_height - 1.
   This keeps weights well-conditioned regardless of k0.

2. **Nondimensional** (x̃, z̃) — PDE derivatives.
   x̃ = k0*x,   z̃ = k0*z.
   Helmholtz residual in nondimensional form:
       ∂²E/∂x̃² + ∂²E/∂z̃² + εr E = 0
   The conversion x̃ → x_n is applied inside ``field_components_nd``
   so that autograd differentiates through the full graph.

Fourier features
----------------
Optional powers-of-two encoding applied to *normalised* coordinates:
    [sin(2^l π c), cos(2^l π c)]  for l = 0 .. L-1, c ∈ {x_n, z_n}
Disabled by default (``fourier_features: false``).

Activations
-----------
- ``tanh`` (default) — smooth, bounded, stable.
- ``sin``  (SIREN)   — good for oscillatory solutions; uses SIREN init.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import ModelConfig, PhysicsConfig


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return SinActivation()
    raise ValueError(f"Unsupported activation '{name}'. Use 'tanh' or 'sin'.")


class SinActivation(nn.Module):
    """Sine activation (SIREN-style)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# ---------------------------------------------------------------------------
# Fourier feature encoding
# ---------------------------------------------------------------------------


class FourierFeatureEncoding(nn.Module):
    """Powers-of-two Fourier features on normalised [-1, 1] coordinates.

    Output dimension = 4 * num_levels  (sin+cos for each of x_n, z_n).
    """

    def __init__(self, num_levels: int = 8) -> None:
        super().__init__()
        self.num_levels = num_levels

    def forward(self, x_n: torch.Tensor, z_n: torch.Tensor) -> torch.Tensor:
        parts = []
        for l in range(self.num_levels):
            freq = (2.0 ** l) * math.pi
            parts += [
                torch.sin(freq * x_n),
                torch.cos(freq * x_n),
                torch.sin(freq * z_n),
                torch.cos(freq * z_n),
            ]
        return torch.stack(parts, dim=-1)  # (N, 4*num_levels)


class RandomFourierFeatures(nn.Module):
    """Random Fourier features (retained for backwards compatibility)."""

    def __init__(self, in_dim: int, num_features: int, scale: float) -> None:
        super().__init__()
        B = torch.randn(in_dim, num_features) * scale
        self.register_buffer("B", B)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = coords @ self.B  # type: ignore[operator]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FieldMLP(nn.Module):
    """MLP mapping (x, z) → (E_real, E_imag).

    Network inputs are normalised to [-1, 1].
    PDE derivatives are computed w.r.t. nondimensional coordinates x̃, z̃
    via ``field_components_nd``.
    """

    def __init__(self, model_cfg: "ModelConfig", physics: "PhysicsConfig") -> None:
        super().__init__()
        self._k0 = physics.k0
        self._period = physics.period
        self._domain_height = physics.domain_height
        self.use_fourier = model_cfg.fourier_features

        if self.use_fourier:
            num_levels = max(1, model_cfg.num_fourier_features // 4)
            self.fourier_enc = FourierFeatureEncoding(num_levels=num_levels)
            in_dim = 4 * num_levels
        else:
            in_dim = 2  # (x_n, z_n) in [-1,1]

        width = model_cfg.hidden_width
        act = _activation(model_cfg.activation)
        layers: list[nn.Module] = [nn.Linear(in_dim, width), act]
        for _ in range(model_cfg.hidden_layers - 1):
            layers += [nn.Linear(width, width), act]
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights(model_cfg.activation)

    def _init_weights(self, activation: str) -> None:
        is_siren = activation.lower() == "sin"
        for i, module in enumerate(self.net):
            if not isinstance(module, nn.Linear):
                continue
            if is_siren:
                fan_in = module.weight.shape[1]
                if i == 0:
                    nn.init.uniform_(module.weight, -1.0 / fan_in, 1.0 / fan_in)
                else:
                    bound = math.sqrt(6.0 / fan_in)
                    nn.init.uniform_(module.weight, -bound, bound)
            else:
                nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    # --- coordinate helpers ---

    def _phys_to_norm(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Physical coords → normalised [-1, 1]."""
        return 2.0 * x / self._period - 1.0, 2.0 * z / self._domain_height - 1.0

    def _nd_to_norm(self, x_tilde: torch.Tensor, z_tilde: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nondimensional coords (k0*x, k0*z) → normalised [-1, 1].

        Preserves the autograd graph so derivatives w.r.t. x̃/z̃ propagate.
        """
        # x_n = 2*x/period - 1 = 2*(x̃/k0)/period - 1 = 2*x̃/(k0*period) - 1
        x_n = 2.0 * x_tilde / (self._k0 * self._period) - 1.0
        z_n = 2.0 * z_tilde / (self._k0 * self._domain_height) - 1.0
        return x_n, z_n

    def _encode(self, x_n: torch.Tensor, z_n: torch.Tensor) -> torch.Tensor:
        if self.use_fourier:
            return self.fourier_enc(x_n, z_n)
        return torch.stack([x_n, z_n], dim=-1)

    # --- forward interfaces ---

    def field_components_nd(
        self, x_tilde: torch.Tensor, z_tilde: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(E_real, E_imag) from nondimensional coords; graph preserved."""
        x_n, z_n = self._nd_to_norm(x_tilde, z_tilde)
        out = self.net(self._encode(x_n, z_n))
        return out[:, 0], out[:, 1]

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Shape (N, 2) from physical coordinates."""
        x_n, z_n = self._phys_to_norm(x, z)
        return self.net(self._encode(x_n, z_n))

    def field_components(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(E_real, E_imag) from physical coordinates."""
        out = self.forward(x, z)
        return out[:, 0], out[:, 1]
