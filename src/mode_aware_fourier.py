"""Mode-aware feature and explicit Fourier-modal field representations.

The classes in this module retain the repository's ``exp(+i omega t)``
convention.  A downward/outgoing basis wave is
``exp(i*kx*x - i*kz*z)``.  Its first-order TE magnetic fields are derived,
not independently guessed: ``Hx = (kz/k0) E`` and ``Hz = (kx/k0) E``.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn


def outgoing_kz(kx: float | np.ndarray, n: float, k0: float) -> np.ndarray:
    """Return the downward/outgoing branch for ``exp(-i*kz*z)``.

    Positive real ``kz`` propagates towards +z.  For evanescent modes the
    returned branch is ``-i*alpha``; substituting it in the stated exponential
    gives ``exp(-alpha*z)``.
    """
    kx_arr = np.asarray(kx, dtype=float)
    q = (n * k0) ** 2 - kx_arr ** 2
    return np.where(q >= 0.0, np.sqrt(np.maximum(q, 0.0)) + 0j,
                    -1j * np.sqrt(np.maximum(-q, 0.0)))


def mode_specification(order: int, period: float, k0: float, n: float, kx_inc: float = 0.0) -> dict:
    """Metadata for a Bloch diffraction order in one homogeneous medium."""
    g0 = 2.0 * math.pi / period
    kx = kx_inc + order * g0
    kz = complex(outgoing_kz(kx, n, k0))
    return {
        "order": int(order), "G0": g0, "kx": kx, "kz": kz,
        "status": "propagating" if abs(kz.real) > 1e-12 else "evanescent",
        "propagation_direction": "+z outgoing" if abs(kz.real) > 1e-12 else "+z decaying",
    }


class ModeAwareFeatureEncoder(nn.Module):
    """Exact periodic lateral and material-specific vertical mode features.

    Each order has sin/cos(kx_m x) factors.  Propagating modes use sin/cos of
    its *own* material kz.  Evanescent modes use two bounded decays measured
    from the two subdomain boundaries, permitting either interface to seed a
    decaying field.
    """
    def __init__(self, *, orders: Iterable[int], period: float, k0: float, n: float,
                 z_lo: float, z_hi: float, kx_inc: float = 0.0) -> None:
        super().__init__()
        self.orders = tuple(int(m) for m in orders)
        self.period, self.k0, self.n = period, k0, n
        self.z_lo, self.z_hi, self.kx_inc = z_lo, z_hi, kx_inc
        specs = [mode_specification(m, period, k0, n, kx_inc) for m in self.orders]
        self.register_buffer("kx", torch.tensor([s["kx"] for s in specs], dtype=torch.float64))
        self.register_buffer("kz_re", torch.tensor([s["kz"].real for s in specs], dtype=torch.float64))
        self.register_buffer("alpha", torch.tensor([-s["kz"].imag for s in specs], dtype=torch.float64))

    @property
    def output_dim(self) -> int:
        # lateral sin/cos times two vertical terms for every retained order
        return 4 * len(self.orders)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(dtype=x.dtype, device=x.device)
        kz_re = self.kz_re.to(dtype=x.dtype, device=x.device)
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        phase_x = x[:, None] * kx[None, :]
        sx, cx = torch.sin(phase_x), torch.cos(phase_x)
        local_z = z[:, None] - self.z_lo
        propagating = alpha.abs() < torch.finfo(x.dtype).eps * 10
        # sin/cos basis for propagating modes; the two interface decays for
        # evanescent modes.  Each exponential is in [0, 1].
        vz_a = torch.where(propagating, torch.sin(local_z * kz_re),
                           torch.exp(-alpha * torch.clamp(local_z, min=0.0)))
        vz_b = torch.where(propagating, torch.cos(local_z * kz_re),
                           torch.exp(-alpha * torch.clamp(self.z_hi - z[:, None], min=0.0)))
        return torch.cat((sx * vz_a, cx * vz_a, sx * vz_b, cx * vz_b), dim=1)


class ModeAwareFeatureMLP(nn.Module):
    """Variant B: exact grating-x plus material-kz feature MLP."""
    def __init__(self, *, orders: Iterable[int], period: float, k0: float, n: float,
                 z_lo: float, z_hi: float, hidden_layers: int = 4, hidden_width: int = 64) -> None:
        super().__init__()
        self.encoder = ModeAwareFeatureEncoder(orders=orders, period=period, k0=k0, n=n,
                                               z_lo=z_lo, z_hi=z_hi)
        layers: list[nn.Module] = [nn.Linear(self.encoder.output_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers.append(nn.Linear(hidden_width, 6))
        self.net = nn.Sequential(*layers)
        # A linear modal skip makes every retained homogeneous basis wave
        # directly reachable; the nonlinear body remains available for the
        # non-separable grating-region coefficient corrections.
        self.modal_skip = nn.Linear(self.encoder.output_dim, 6, bias=False)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.modal_skip.weight)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, z)
        return self.net(features) + self.modal_skip(features)

    def field_components(self, x: torch.Tensor, z: torch.Tensor):
        out = self.forward(x, z)
        return tuple(out[:, i] for i in range(6))


class ExplicitFourierModalNetwork(nn.Module):
    """Variant C: a truncated Fourier field with complex modal coefficients.

    ``coefficients`` are the trainable complex modal coefficient functions in
    their constant-function limit.  The optional z-MLP supplies a residual
    coefficient function for nonuniform grating regions; it starts at zero so
    an exact homogeneous mode is represented without optimization noise.
    Magnetic fields are reconstructed from E derivatives, preserving the
    first-order Maxwell field convention exactly.
    """
    def __init__(self, *, orders: Iterable[int], period: float, k0: float, n: float,
                 z_lo: float, z_hi: float, use_coefficient_mlp: bool = True) -> None:
        super().__init__()
        self.orders = tuple(int(m) for m in orders)
        self.period, self.k0, self.n, self.z_lo, self.z_hi = period, k0, n, z_lo, z_hi
        specs = [mode_specification(m, period, k0, n) for m in self.orders]
        self.register_buffer("kx", torch.tensor([s["kx"] for s in specs], dtype=torch.float64))
        self.register_buffer("kz_re", torch.tensor([s["kz"].real for s in specs], dtype=torch.float64))
        self.register_buffer("kz_im", torch.tensor([s["kz"].imag for s in specs], dtype=torch.float64))
        self.coefficients = nn.Parameter(torch.zeros(len(self.orders), 2, dtype=torch.float64))
        self.coefficient_mlp = None
        if use_coefficient_mlp:
            self.coefficient_mlp = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 2 * len(self.orders)))
            # Keep the output identically zero at initialisation (the exact
            # zero-source solution), but *do not* zero the hidden map.  Zeroing
            # every layer makes the only reachable first update a constant
            # homogeneous mode; the forced grating residual then has virtually
            # no gradient into ±1.  A live hidden z basis plus a zero final head
            # preserves E_scat=0 while allowing the first gradient step to form
            # non-constant coefficient functions.
            first, last = self.coefficient_mlp[0], self.coefficient_mlp[-1]
            nn.init.xavier_normal_(first.weight); nn.init.zeros_(first.bias)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            # Modal wavenumbers and coefficients are stored in float64 to
            # avoid phase loss for evanescent orders; keep this residual head
            # in the same default precision.
            self.coefficient_mlp.double()

    def set_exact_mode(self, order: int, amplitude: complex = 1.0 + 0j) -> None:
        """Set the analytic homogeneous outgoing mode used by regression tests."""
        with torch.no_grad():
            self.coefficients.zero_()
            i = self.orders.index(int(order))
            self.coefficients[i, 0] = amplitude.real
            self.coefficients[i, 1] = amplitude.imag

    def complex_field(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(dtype=x.dtype, device=x.device)
        kz = self.kz_re.to(dtype=x.dtype, device=x.device) + 1j * self.kz_im.to(dtype=x.dtype, device=x.device)
        c = self.coefficients.to(dtype=x.dtype, device=x.device)
        coeff = c[:, 0] + 1j * c[:, 1]
        if self.coefficient_mlp is not None:
            zn = (2.0 * (z - self.z_lo) / (self.z_hi - self.z_lo) - 1.0)[:, None]
            delta = self.coefficient_mlp(zn).reshape(-1, len(self.orders), 2)
            coeff = coeff[None, :] + delta[..., 0] + 1j * delta[..., 1]
        phase = torch.exp(1j * (x[:, None] * kx[None, :] - z[:, None] * kz[None, :]))
        return torch.sum(phase * coeff, dim=1) if coeff.ndim == 2 else torch.sum(phase * coeff[None, :], dim=1)

    def modal_coefficients_at(self, z: torch.Tensor) -> torch.Tensor:
        """Complex E_y coefficients in the global exp(i*kx_m*x) basis.

        The returned coefficient already carries the vertical phase factor at
        ``z``.  This is the independent coefficient-domain extraction path.
        """
        kz = self.kz_re.to(dtype=z.dtype, device=z.device) + 1j*self.kz_im.to(dtype=z.dtype, device=z.device)
        c = self.coefficients.to(dtype=z.dtype, device=z.device); a=c[:,0]+1j*c[:,1]
        if self.coefficient_mlp is not None:
            zn=(2*(z-self.z_lo)/(self.z_hi-self.z_lo)-1)[:,None]
            d=self.coefficient_mlp(zn).reshape(-1,len(self.orders),2)
            a=a[None,:]+d[...,0]+1j*d[...,1]
        if a.ndim==1: a=a[None,:].expand(len(z),-1)
        return a*torch.exp(-1j*z[:,None]*kz[None,:])

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Use autograd expressions so residual evaluators differentiate the
        # reconstructed truncated Fourier field, rather than unrelated H heads.
        # Boundary/interface calls need values only and commonly pass ordinary
        # tensors.  The coefficient-function derivative still needs a graph;
        # make a local coordinate leaf in that case.  PDE callers already pass
        # coordinate leaves, so their physical derivative path is retained.
        if self.coefficient_mlp is not None and not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        if self.coefficient_mlp is not None and not z.requires_grad:
            z = z.detach().clone().requires_grad_(True)
        e = self.complex_field(x, z)
        kx = self.kx.to(dtype=x.dtype, device=x.device)
        kz = self.kz_re.to(dtype=x.dtype, device=x.device) + 1j * self.kz_im.to(dtype=x.dtype, device=x.device)
        c = self.coefficients.to(dtype=x.dtype, device=x.device); coeff = c[:, 0] + 1j*c[:, 1]
        if self.coefficient_mlp is None:
            phase = torch.exp(1j*(x[:, None]*kx[None, :] - z[:, None]*kz[None, :]))
            hx = torch.sum((kz[None, :] / self.k0) * phase * coeff[None, :], dim=1)
            hz = torch.sum((kx[None, :] / self.k0) * phase * coeff[None, :], dim=1)
        else:
            # For variable coefficient functions, differentiate E so the
            # reconstruction remains Maxwell-consistent by definition.
            de_dx = torch.autograd.grad(e.real, x, torch.ones_like(e.real), create_graph=True, retain_graph=True)[0] + 1j * torch.autograd.grad(e.imag, x, torch.ones_like(e.imag), create_graph=True, retain_graph=True)[0]
            de_dz = torch.autograd.grad(e.real, z, torch.ones_like(e.real), create_graph=True, retain_graph=True)[0] + 1j * torch.autograd.grad(e.imag, z, torch.ones_like(e.imag), create_graph=True, retain_graph=True)[0]
            hx, hz = 1j * de_dz / self.k0, -1j * de_dx / self.k0
        return torch.stack((e.real, e.imag, hx.real, hx.imag, hz.real, hz.imag), dim=1)

    def field_components(self, x: torch.Tensor, z: torch.Tensor):
        out = self.forward(x, z)
        return tuple(out[:, i] for i in range(6))


class ExplicitFourierModalDD(nn.Module):
    """Three-subdomain explicit modal PINN for the layered-background solver."""
    def __init__(self, physics, modal_order_max: int = 3, use_coefficient_mlp: bool = True) -> None:
        super().__init__()
        if modal_order_max < 0:
            raise ValueError("modal_order_max must be nonnegative")
        self.physics = physics
        self.modal_order_max = modal_order_max
        orders = tuple(range(-modal_order_max, modal_order_max + 1))
        kw = dict(orders=orders, period=physics.period, k0=physics.k0,
                  use_coefficient_mlp=use_coefficient_mlp)
        self.net_air = ExplicitFourierModalNetwork(n=physics.n_air, z_lo=0.0,
                                                   z_hi=physics.ridge_z_min, **kw)
        self.net_grat = ExplicitFourierModalNetwork(n=physics.n_ridge,
                                                    z_lo=physics.ridge_z_min,
                                                    z_hi=physics.ridge_z_max, **kw)
        self.net_sub = ExplicitFourierModalNetwork(n=physics.n_substrate,
                                                   z_lo=physics.ridge_z_max,
                                                   z_hi=physics.domain_height, **kw)

    def forward_E(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
        p = self.physics
        for mask, net in ((z <= p.ridge_z_min, self.net_air),
                          ((z > p.ridge_z_min) & (z <= p.ridge_z_max), self.net_grat),
                          (z > p.ridge_z_max, self.net_sub)):
            if mask.any():
                out[mask] = net(x[mask], z[mask])[:, :2]
        return out
