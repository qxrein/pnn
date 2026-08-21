"""Analytical benchmark solutions for PINN validation.

Two reference cases for progressive validation:

1. Homogeneous plane-wave benchmark
   Uniform εr = n².  Analytical solution: E(z) = exp(-i k0 n z).

2. Homogeneous layered-medium (1-D) benchmark
   Air | dielectric slab | substrate — no lateral grating.
   Solved via direct boundary-condition matching (exact TMM).

Convention throughout
---------------------
- exp(+iωt) convention suppressed.
- Wave propagates in +z (downward): E_inc = exp(-i k0 z).
- z = 0 is the top (incident) boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1.  Homogeneous plane-wave benchmark
# ---------------------------------------------------------------------------


def plane_wave_field(
    z: torch.Tensor,
    n: float,
    k0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downward-propagating plane wave in uniform medium of index n.

    E(z) = exp(-i k0 n z)  =>  Re = cos(k0 n z),  Im = -sin(k0 n z).
    """
    phase = k0 * n * z
    return torch.cos(phase), -torch.sin(phase)


@dataclass
class PlaneWaveBenchmark:
    """Parameters for the homogeneous plane-wave benchmark."""

    n: float = 1.0
    k0: float = 2.0 * math.pi
    domain_height: float = 2.0
    period: float = 1.0

    @property
    def eps(self) -> float:
        return self.n**2

    def analytical_field(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return plane_wave_field(z, self.n, self.k0)

    def analytical_field_np(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phase = self.k0 * self.n * z
        return np.cos(phase), -np.sin(phase)


# ---------------------------------------------------------------------------
# 2.  Layered-medium (1-D) benchmark — exact boundary-condition matching
# ---------------------------------------------------------------------------


@dataclass
class LayeredMediumBenchmark:
    """Air | dielectric slab | substrate 1-D benchmark.

    Layers (z increasing downward):
        0 <= z <= z_slab_top              : air,  n = n_air
        z_slab_top <= z <= z_slab_bot     : slab, n = n_slab
        z >= z_slab_bot                   : sub,  n = n_sub

    Incident wave: E_inc = exp(-i k1 z) from above.
    Semi-infinite substrate: forward-propagating wave only.
    """

    n_air: float = 1.0
    n_slab: float = 1.5
    n_sub: float = 1.45
    k0: float = 2.0 * math.pi
    z_slab_top: float = 1.2
    z_slab_bot: float = 1.4
    domain_height: float = 2.0
    period: float = 1.0

    def _tmm_coefficients(self) -> dict[str, complex]:
        """Exact solution via direct boundary-condition matching.

        Fields:
            Air:  E = exp(-i k1 z)   + r exp(+i k1 z)
            Slab: E = A exp(-i k2 z') + B exp(+i k2 z'),  z' = z - z_slab_top
            Sub:  E = t exp(-i k3 z''),                   z'' = z - z_slab_bot

        BCs at z_slab_top: E and dE/dz continuous  =>  2x2 system in (r, t).
        A, B expressed in terms of t from BC at z_slab_bot.
        """
        k1 = self.k0 * self.n_air
        k2 = self.k0 * self.n_slab
        k3 = self.k0 * self.n_sub
        h  = self.z_slab_bot - self.z_slab_top
        z0 = self.z_slab_top

        E0  = np.exp(-1j * k1 * z0)
        E0r = np.exp(+1j * k1 * z0)
        em  = np.exp(+1j * k2 * h)
        ep  = np.exp(-1j * k2 * h)

        P  = ((k2 + k3) * em + (k2 - k3) * ep) / (2.0 * k2)
        Qk = ((k2 + k3) * em - (k2 - k3) * ep) / 2.0

        A_mat = np.array([[E0r, -P], [k1 * E0r, Qk]], dtype=complex)
        b_vec = np.array([-E0, k1 * E0], dtype=complex)
        sol   = np.linalg.solve(A_mat, b_vec)
        r_total, t_total = sol[0], sol[1]

        A_amp = t_total * (k2 + k3) / (2.0 * k2) * em
        B_amp = t_total * (k2 - k3) / (2.0 * k2) * ep

        return {
            "r": r_total, "t": t_total,
            "A": A_amp,   "B": B_amp,
            "k1": k1, "k2": k2, "k3": k3,
        }

    def analytical_field_np(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Analytical E(x, z) — x-independent."""
        c = self._tmm_coefficients()
        r, t   = c["r"], c["t"]
        A, B   = c["A"], c["B"]
        k1, k2, k3 = c["k1"], c["k2"], c["k3"]

        E = np.zeros_like(z, dtype=complex)
        air  = z <= self.z_slab_top
        slab = (z > self.z_slab_top) & (z <= self.z_slab_bot)
        sub  = z > self.z_slab_bot

        E[air]  = np.exp(-1j * k1 * z[air]) + r * np.exp(+1j * k1 * z[air])
        zp      = z[slab] - self.z_slab_top
        E[slab] = A * np.exp(-1j * k2 * zp) + B * np.exp(+1j * k2 * zp)
        zpp     = z[sub] - self.z_slab_bot
        E[sub]  = t * np.exp(-1j * k3 * zpp)

        return np.real(E), np.imag(E)

    def analytical_field(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_np = z.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        e_r, e_i = self.analytical_field_np(x_np, z_np)
        return (
            torch.as_tensor(e_r, dtype=z.dtype, device=z.device),
            torch.as_tensor(e_i, dtype=z.dtype, device=z.device),
        )

    def epsilon_r_np(self, z: np.ndarray) -> np.ndarray:
        eps = np.full_like(z, self.n_air**2)
        eps[(z > self.z_slab_top) & (z <= self.z_slab_bot)] = self.n_slab**2
        eps[z > self.z_slab_bot] = self.n_sub**2
        return eps

    def reflection_coefficient(self) -> complex:
        return self._tmm_coefficients()["r"]

    def transmission_coefficient(self) -> complex:
        return self._tmm_coefficients()["t"]


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def evaluate_benchmark_errors(
    pinn_E_real: np.ndarray,
    pinn_E_imag: np.ndarray,
    ref_E_real: np.ndarray,
    ref_E_imag: np.ndarray,
) -> dict[str, float]:
    """Comprehensive error metrics between PINN and reference complex fields.

    Metrics
    -------
    relative_l2_real
        ||Re(E_pinn) - Re(E_ref)|| / ||Re(E_ref)||
    relative_l2_imag
        ||Im(E_pinn) - Im(E_ref)|| / ||Im(E_ref)||
    relative_l2_magnitude
        || |E_pinn| - |E_ref| || / || |E_ref| ||
    relative_complex_l2
        ||E_pinn - E_ref||_C / ||E_ref||_C
        Phase-sensitive.  This is the primary validation metric.
    relative_l2_phase_aligned
        Complex L2 after removing the best global phase offset.
        Measures field-shape error after compensating a harmless global shift.
    phase_rmse_deg
        RMS phase error in degrees (after global alignment).
    global_phase_offset_deg
        Best-fit global phase offset alpha (degrees).
        Non-zero => the PINN has learned the correct field shape but with
        a global phase shift, not a fundamentally wrong solution.
    """
    eps = 1e-12
    pinn_c = pinn_E_real.astype(complex) + 1j * pinn_E_imag.astype(complex)
    ref_c  = ref_E_real.astype(complex)  + 1j * ref_E_imag.astype(complex)

    l2_real = float(np.linalg.norm((pinn_E_real - ref_E_real).ravel())
                    / (np.linalg.norm(ref_E_real.ravel()) + eps))
    l2_imag = float(np.linalg.norm((pinn_E_imag - ref_E_imag).ravel())
                    / (np.linalg.norm(ref_E_imag.ravel()) + eps))
    l2_mag  = float(np.linalg.norm((np.abs(pinn_c) - np.abs(ref_c)).ravel())
                    / (np.linalg.norm(np.abs(ref_c).ravel()) + eps))
    l2_complex = float(np.linalg.norm((pinn_c - ref_c).ravel())
                       / (np.linalg.norm(ref_c.ravel()) + eps))

    # Global phase alignment: alpha = arg( <E_pinn, E_ref> )
    alpha = np.angle(np.vdot(ref_c.ravel(), pinn_c.ravel()))
    pinn_aligned = pinn_c * np.exp(-1j * alpha)
    l2_aligned = float(np.linalg.norm((pinn_aligned - ref_c).ravel())
                       / (np.linalg.norm(ref_c.ravel()) + eps))

    phase_diff = np.angle(pinn_aligned) - np.angle(ref_c)
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    phase_rmse = float(np.sqrt(np.mean(phase_diff.ravel() ** 2)) * 180.0 / np.pi)

    return {
        "relative_l2_real":          l2_real,
        "relative_l2_imag":          l2_imag,
        "relative_l2_magnitude":     l2_mag,
        "relative_complex_l2":       l2_complex,
        "relative_l2_phase_aligned": l2_aligned,
        "phase_rmse_deg":            phase_rmse,
        "global_phase_offset_deg":   float(alpha * 180.0 / np.pi),
    }


def evaluate_benchmark_errors_by_region(
    pinn_E_real: np.ndarray,
    pinn_E_imag: np.ndarray,
    ref_E_real: np.ndarray,
    ref_E_imag: np.ndarray,
    z: np.ndarray,
    bm: "LayeredMediumBenchmark",
) -> dict[str, dict[str, float]]:
    """Per-region errors (air, slab, substrate) for the layered benchmark."""
    masks = {
        "air":  z <= bm.z_slab_top,
        "slab": (z > bm.z_slab_top) & (z <= bm.z_slab_bot),
        "sub":  z > bm.z_slab_bot,
    }
    return {
        name: evaluate_benchmark_errors(
            pinn_E_real[mask], pinn_E_imag[mask],
            ref_E_real[mask],  ref_E_imag[mask],
        )
        for name, mask in masks.items()
        if np.any(mask)
    }


def plot_complex_field_comparison(
    z: np.ndarray,
    pinn_E_real: np.ndarray,
    pinn_E_imag: np.ndarray,
    ref_E_real: np.ndarray,
    ref_E_imag: np.ndarray,
    bm: "LayeredMediumBenchmark",
    output_path: str,
    title: str = "Field comparison",
) -> None:
    """Five-panel figure: Re, Im, |E|, phase, complex-plane trajectory."""
    import pathlib

    import matplotlib.pyplot as plt

    pinn_c = pinn_E_real + 1j * pinn_E_imag
    ref_c  = ref_E_real  + 1j * ref_E_imag
    alpha  = np.angle(np.vdot(ref_c.ravel(), pinn_c.ravel()))
    aligned = pinn_c * np.exp(-1j * alpha)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
    iv = dict(color="gray", ls=":", lw=0.8)

    def _v(ax):
        ax.axvline(bm.z_slab_top, **iv)
        ax.axvline(bm.z_slab_bot, **iv)

    axes[0].plot(z, ref_E_real,    "b-",  lw=1.5, label="TMM")
    axes[0].plot(z, pinn_E_real,   "r--", lw=1.2, label="PINN")
    axes[0].plot(z, aligned.real,  "g:",  lw=1.0, label="PINN (aligned)")
    _v(axes[0]); axes[0].set_xlabel("z (λ)"); axes[0].set_title("Re{E}"); axes[0].legend(fontsize=7)

    axes[1].plot(z, ref_E_imag,    "b-",  lw=1.5)
    axes[1].plot(z, pinn_E_imag,   "r--", lw=1.2)
    axes[1].plot(z, aligned.imag,  "g:",  lw=1.0)
    _v(axes[1]); axes[1].set_xlabel("z (λ)"); axes[1].set_title("Im{E}")

    axes[2].plot(z, np.abs(ref_c),  "b-",  lw=1.5, label="TMM")
    axes[2].plot(z, np.abs(pinn_c), "r--", lw=1.2, label="PINN")
    _v(axes[2]); axes[2].set_xlabel("z (λ)"); axes[2].set_title("|E|"); axes[2].legend(fontsize=7)

    axes[3].plot(z, np.unwrap(np.angle(ref_c))  * 180/np.pi, "b-",  lw=1.5, label="TMM")
    axes[3].plot(z, np.unwrap(np.angle(pinn_c)) * 180/np.pi, "r--", lw=1.2, label="PINN")
    axes[3].plot(z, np.unwrap(np.angle(aligned)) * 180/np.pi,"g:",  lw=1.0, label="PINN (aligned)")
    _v(axes[3]); axes[3].set_xlabel("z (λ)"); axes[3].set_title("Phase (deg)"); axes[3].legend(fontsize=7)

    axes[4].plot(ref_E_real,    ref_E_imag,    "b-",  lw=1.5, label="TMM")
    axes[4].plot(pinn_E_real,   pinn_E_imag,   "r--", lw=1.2, label="PINN")
    axes[4].plot(aligned.real,  aligned.imag,  "g:",  lw=1.0, label="PINN (aligned)")
    axes[4].set_xlabel("Re{E}"); axes[4].set_ylabel("Im{E}")
    axes[4].set_title("Complex trajectory"); axes[4].legend(fontsize=7)
    axes[4].set_aspect("equal")

    fig.suptitle(title)
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
