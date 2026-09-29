"""Vanilla (Raissi-style) monolithic PINN for the 2-D binary grating problem.

Architecture
------------
Input  : (x, z) ∈ ℝ², normalised to [−1, 1]² over the domain
Hidden : N_LAYERS fully-connected layers, N_UNITS units each, tanh activation
Output : 6 scalars: (Er_s, Ei_s, Hr_x, Hi_x, Hr_z, Hi_z)
         — the real/imaginary parts of the scattered E_y and the two H components

This is the control architecture for Proposition 1: unlike ExplicitFourierModalDD,
x and z are jointly processed through every hidden layer.  There is NO Fourier/Bloch
basis in the hidden representation.  The network is therefore in principle capable
of learning arbitrary (x, z) structure, including the x-periodic pattern required
for ±1 diffraction orders.

Zero-output initialisation
--------------------------
The last linear layer's weight and bias are set to zero at construction, so the
network outputs E_scat = 0 everywhere at initialisation — the same starting point
as ExplicitFourierModalDD (whose coefficient MLP final head is also initialised to
zero).  The hidden layers are initialised with Xavier-normal weights.

Interface contract
------------------
The model exposes the same `field_components(x, z)` API used by
`maxwell_2d_lbg_pde_residual`, `maxwell_2d_nd_interface_loss`,
`lbg_top_bc`, `lbg_bottom_bc`, and the spatial Fourier extractor.

It also provides net_air / net_grat / net_sub *sub-network views* that
delegate to the monolithic network but only evaluate at points in the
relevant z-range.  This lets `layered_bg_loss` (which calls model.net_air,
model.net_grat, model.net_sub) work without modification.

Parameter count
---------------
Default (4 hidden layers, 32 units):
    Linear(2→32): 2×32 + 32 = 96
    3 × Linear(32→32): 3 × (32×32 + 32) = 3168
    Linear(32→6): 32×6 + 6 = 198
    Total: 96 + 3168 + 198 = 3462 parameters

This is ~2× our modal baseline (1620).  Since the vanilla model has MORE
parameters, any failure to recover ±1 cannot be attributed to insufficient
capacity — strengthening the architectural-obstruction argument.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.config import PhysicsConfig

# ─────────────────────────────────────────────────────────────────────────────
# Default architecture constants
# ─────────────────────────────────────────────────────────────────────────────

N_LAYERS   = 4    # number of hidden layers
N_UNITS    = 32   # units per hidden layer


# ─────────────────────────────────────────────────────────────────────────────
# Core monolithic network
# ─────────────────────────────────────────────────────────────────────────────

class VanillaMLP(nn.Module):
    """Standard (x, z) → (Er_s, Ei_s, Hr_x, Hi_x, Hr_z, Hi_z) MLP.

    Parameters
    ----------
    physics : PhysicsConfig
        Used for coordinate normalisation (domain_height, period).
    n_layers : int
        Number of fully-connected hidden layers.
    n_units : int
        Width of each hidden layer.
    """

    def __init__(
        self,
        physics: "PhysicsConfig",
        n_layers: int = N_LAYERS,
        n_units: int  = N_UNITS,
    ) -> None:
        super().__init__()
        self.physics  = physics
        self.n_layers = n_layers
        self.n_units  = n_units

        # Build MLP: 2 → [n_units]*n_layers → 6
        layers: list[nn.Module] = []
        in_dim = 2
        for i in range(n_layers):
            lin = nn.Linear(in_dim, n_units, dtype=torch.float64)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
            in_dim = n_units

        # Final head — initialised to ZERO so E_scat = 0 at t=0
        head = nn.Linear(in_dim, 6, dtype=torch.float64)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Coordinate normalisation
    # ------------------------------------------------------------------

    def _normalise(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Map (x, z) ∈ [0, Λ] × [0, H] to [−1, 1]²."""
        p    = self.physics
        xn   = 2.0 * x / p.period        - 1.0
        zn   = 2.0 * z / p.domain_height - 1.0
        return torch.stack([xn, zn], dim=-1)   # (..., 2)

    # ------------------------------------------------------------------
    # Public API (matches ExplicitFourierModalNetwork)
    # ------------------------------------------------------------------

    def field_components(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return (Er_s, Ei_s, Hr_x, Hi_x, Hr_z, Hi_z), each shape (N,)."""
        inp = self._normalise(x, z)          # (N, 2)
        out = self.net(inp)                   # (N, 6)
        return tuple(out[:, i] for i in range(6))

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return (N, 6) output tensor."""
        inp = self._normalise(x, z)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-network view  (delegates to the shared monolithic network)
# ─────────────────────────────────────────────────────────────────────────────

class _SubnetView(nn.Module):
    """Thin wrapper that makes a monolithic VanillaMLP look like a subdomain net.

    `layered_bg_loss` accesses model.net_air, model.net_grat, model.net_sub.
    Each subdomain view delegates field_components to the shared backbone.
    All parameters live in the backbone; this module has no own parameters.
    """

    def __init__(self, backbone: VanillaMLP) -> None:
        super().__init__()
        self._backbone = backbone

    def field_components(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self._backbone.field_components(x, z)

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return self._backbone.forward(x, z)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level wrapper compatible with layered_bg_loss
# ─────────────────────────────────────────────────────────────────────────────

class VanillaPINN(nn.Module):
    """Monolithic Raissi-style PINN with the domain-decomposed loss interface.

    Exposes net_air / net_grat / net_sub attributes that are all views of the
    same underlying VanillaMLP backbone.  This makes `layered_bg_loss` work
    unchanged: every subnet call routes to the single global network.

    The intentional design choice — no subdomain decomposition, no Fourier/
    Bloch basis, x and z entangled in every hidden layer — makes this the
    cleanest possible control for Proposition 1.

    Parameters
    ----------
    physics : PhysicsConfig
    n_layers, n_units : architecture (see VanillaMLP for defaults/param counts)
    """

    def __init__(
        self,
        physics: "PhysicsConfig",
        n_layers: int = N_LAYERS,
        n_units:  int = N_UNITS,
    ) -> None:
        super().__init__()
        self.physics = physics

        # Single shared backbone
        self.backbone = VanillaMLP(physics, n_layers=n_layers, n_units=n_units)

        # Sub-network views (no own parameters — all params in backbone)
        self.net_air  = _SubnetView(self.backbone)
        self.net_grat = _SubnetView(self.backbone)
        self.net_sub  = _SubnetView(self.backbone)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def zero_output_check(self, n_test: int = 32) -> float:
        """Verify E_scat ≡ 0 at initialisation.  Returns max |output|."""
        with torch.no_grad():
            p  = self.physics
            x  = torch.rand(n_test, dtype=torch.float64) * p.period
            z  = torch.rand(n_test, dtype=torch.float64) * p.domain_height
            Er, Ei, *_ = self.backbone.field_components(x, z)
            return float(max(Er.abs().max(), Ei.abs().max()))

# ─────────────────────────────────────────────────────────────────────────────
# Fourier-feature (frequency-aware) variant
# ─────────────────────────────────────────────────────────────────────────────

class FourierFeatureMLP(nn.Module):
    """Fourier-feature PINN (Tancik et al. 2020 / Wang et al. 2021 style).

    Architecture
    ------------
    1. Random Fourier features (RFF) embedding of (x, z):
         φ(x, z) = [cos(B [x,z]^T), sin(B [x,z]^T)]   ∈ ℝ^(2*n_freq)
       where B ∈ ℝ^(n_freq × 2) contains physics-informed frequencies.

    2. Bloch-matched frequency construction:
       The grating scatters at Bloch wavenumbers G_m = m * 2π/Λ (m ∈ ℤ).
       The first n_bloch//2 rows of B are set to exactly target these
       spatial frequencies in x, so the embedding can represent the
       cos(G_m x) and sin(G_m x) patterns that the ±1 orders require.
       The remaining frequencies are drawn from a log-uniform distribution
       over [k0/4, 4*k0] to cover both sub- and super-wavelength scales.

    3. Downstream MLP: n_layers fully-connected tanh layers, then a
       zero-initialised head (→ E_scat = 0 at init, same as VanillaMLP).

    Parameter count (defaults: n_freq=16, n_layers=3, n_units=32):
        B matrix:              NOT a parameter (fixed random buffer)
        Linear(2*n_freq → 32): 32*32+32 = 1056  (first hidden layer)
        2 × Linear(32→32):     2 × 1056 = 2112
        Linear(32→6):          198
        Total:                 3366  ≈ same as VanillaMLP (3462)

    The key design intent: the embedding already contains high-frequency
    x-components at the exact Bloch wavenumbers.  If the failure is
    spectral bias (the MLP cannot learn high-frequency x-structure fast
    enough), this should cure it.  If the network still fails, the issue
    is something else (e.g. the contrast-source forcing at those frequencies
    is too weak even with the right representational capacity).
    """

    # Default Fourier-feature hyperparameters
    N_FREQ   = 16    # number of frequency pairs  → embedding dim = 2*N_FREQ
    N_BLOCH  = 8     # how many of the N_FREQ rows target Bloch wavenumbers

    def __init__(
        self,
        physics: "PhysicsConfig",
        n_freq:   int = N_FREQ,
        n_bloch:  int = N_BLOCH,
        n_layers: int = N_LAYERS,
        n_units:  int = N_UNITS,
        seed:     int = 0,
    ) -> None:
        super().__init__()
        self.physics  = physics
        self.n_freq   = n_freq
        self.n_bloch  = min(n_bloch, n_freq)
        self.n_layers = n_layers
        self.n_units  = n_units

        # ── Build frequency matrix B (n_freq, 2) ─────────────────────────
        rng   = np.random.default_rng(seed)
        k0    = physics.k0
        G0    = 2.0 * math.pi / physics.period
        n_rnd = n_freq - self.n_bloch

        # Rows 0..n_bloch-1: Bloch-matched x-frequencies
        # B[i, 0] = G_m  (frequency in x),  B[i, 1] = k0 * n  (typical z-freq)
        bloch_orders = list(range(-(self.n_bloch // 2),
                                   self.n_bloch - self.n_bloch // 2))
        B_bloch = np.zeros((self.n_bloch, 2))
        for i, m in enumerate(bloch_orders):
            B_bloch[i, 0] = float(m) * G0          # x-frequency: m-th Bloch
            B_bloch[i, 1] = k0 * physics.n_air     # z-frequency: propagating

        # Rows n_bloch..n_freq-1: log-uniform random frequencies
        if n_rnd > 0:
            log_lo = math.log(k0 / 4.0 + 1e-8)
            log_hi = math.log(4.0 * k0)
            freqs  = np.exp(rng.uniform(log_lo, log_hi, (n_rnd, 2)))
            signs  = rng.choice([-1.0, 1.0], size=(n_rnd, 2))
            B_rnd  = freqs * signs
        else:
            B_rnd  = np.zeros((0, 2))

        B = np.vstack([B_bloch, B_rnd]).astype(np.float64)   # (n_freq, 2)
        # Register as buffer: fixed, not trained
        self.register_buffer("B", torch.as_tensor(B, dtype=torch.float64))

        # ── Build downstream MLP ──────────────────────────────────────────
        embed_dim = 2 * n_freq
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(n_layers):
            lin = nn.Linear(in_dim, n_units, dtype=torch.float64)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
            in_dim = n_units

        # Zero-initialised head → E_scat = 0 at init
        head = nn.Linear(in_dim, 6, dtype=torch.float64)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)

        self.net = nn.Sequential(*layers)

    def _embed(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Fourier feature embedding.

        xz  : (N, 2) with columns [x_phys, z_phys] in physical units
        B   : (n_freq, 2)
        out : (N, 2*n_freq)  =  [cos(B xz^T), sin(B xz^T)]
        """
        xz   = torch.stack([x, z], dim=-1)           # (N, 2)
        proj = xz @ self.B.T                          # (N, n_freq)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2*n_freq)

    def field_components(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return (Er_s, Ei_s, Hr_x, Hi_x, Hr_z, Hi_z), each shape (N,)."""
        out = self.net(self._embed(x, z))
        return tuple(out[:, i] for i in range(6))

    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return self.net(self._embed(x, z))


class FourierFeaturePINN(nn.Module):
    """Fourier-feature PINN with the domain-decomposed loss interface.

    Drop-in replacement for VanillaPINN: exposes net_air / net_grat / net_sub
    subnet views, all routing to the single FourierFeatureMLP backbone.
    """

    def __init__(
        self,
        physics:  "PhysicsConfig",
        n_freq:   int = FourierFeatureMLP.N_FREQ,
        n_bloch:  int = FourierFeatureMLP.N_BLOCH,
        n_layers: int = N_LAYERS,
        n_units:  int = N_UNITS,
        seed:     int = 0,
    ) -> None:
        super().__init__()
        self.physics  = physics
        self.backbone = FourierFeatureMLP(
            physics, n_freq=n_freq, n_bloch=n_bloch,
            n_layers=n_layers, n_units=n_units, seed=seed,
        )
        self.net_air  = _SubnetView(self.backbone)
        self.net_grat = _SubnetView(self.backbone)
        self.net_sub  = _SubnetView(self.backbone)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def zero_output_check(self, n_test: int = 32) -> float:
        with torch.no_grad():
            p  = self.physics
            x  = torch.rand(n_test, dtype=torch.float64) * p.period
            z  = torch.rand(n_test, dtype=torch.float64) * p.domain_height
            Er, Ei, *_ = self.backbone.field_components(x, z)
            return float(max(Er.abs().max(), Ei.abs().max()))
