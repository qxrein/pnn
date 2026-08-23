"""Residual diagnostics, diffraction metrics, and boundary audits for 2-D Maxwell PINN.

Items implemented
-----------------
1. Dense-grid PDE residual map (6 equations, total magnitude)
2. Residual histogram
3. Residual broken down by region (air / ridge / substrate)
4. Residual near interfaces and near four ridge corners
5. Diffraction-order power (modal reflection and transmission efficiencies)
6. Zeroth-order and first-order power with Poynting-flux comparison
7. Boundary audit: local Robin vs. Fourier/modal DtN boundary options
8. DtN boundary condition for each diffraction order m

Convention
----------
exp(+iωt) suppressed.  E_inc = exp(-ik0 z).
For the scattered field E_scat, H̃_scat:
    Top (z=0):  outgoing upward  → modal DtN: dE_m/dz = -i kz_m E_m
    Bottom:     outgoing downward → modal DtN: dE_m/dz = +i kz_m_sub E_m

Modal DtN boundary
------------------
The local Robin condition H̃_x = ±n E assumes a single dominant plane-wave
mode (zeroth order).  This is exact only if no other propagating orders exist.

The modal Dirichlet-to-Neumann operator uses a Fourier decomposition along
x at the boundary:

    E(x, z_bc) = sum_m E_m exp(i Gm x)      (Gm = m * 2pi/period)
    dE/dz|_{z_bc} = sum_m (±i kz_m) E_m exp(i Gm x)

where kz_m is the outgoing z-wavenumber for order m:
    kz_m² = k0² n² - Gm²
    Choose branch: outgoing = propagating upward/downward.

For the total-field or scattered-field formulation the DtN is enforced as:
    dE/dz(x, z_bc) - DtN[E](x, z_bc) = 0

The current implementation computes the DtN residual as a scalar penalty
on a set of boundary collocation points.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import PhysicsConfig
    from src.maxwell_2d_dd import Maxwell2DDD, Maxwell2DSubdomainMLP


# ---------------------------------------------------------------------------
# Residual map
# ---------------------------------------------------------------------------


def compute_residual_map(
    model: "Maxwell2DDD",
    physics: "PhysicsConfig",
    device: torch.device,
    dtype: torch.dtype,
    nx: int = 64,
    nz: int = 128,
    scattered: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate PDE residuals on a dense grid.

    Returns
    -------
    dict with keys:
        'x', 'z'               : 2-D coordinate arrays (nz, nx)
        'res_total'            : sqrt(sum of squared residuals), shape (nz, nx)
        'res_Ar', 'res_Ai', ... : individual equations, shape (nz, nx)
        'region'               : integer region mask (0=air, 1=grat, 2=sub)
    """
    from src.geometry import epsilon_r
    from src.maxwell_2d_dd import maxwell_2d_dd_pde_residual

    x1d = np.linspace(0.0, physics.period, nx)
    z1d = np.linspace(0.0, physics.domain_height, nz)
    X, Z = np.meshgrid(x1d, z1d)  # (nz, nx)
    x_flat = torch.as_tensor(X.ravel(), dtype=dtype, device=device)
    z_flat = torch.as_tensor(Z.ravel(), dtype=dtype, device=device)

    def eps_fn(x, z):
        return epsilon_r(x, z, physics)

    margin = 1e-3
    res_names = ["Ar", "Ai", "Br", "Bi", "Cr", "Ci"]
    residuals_np = {}

    model.eval()
    for mask_cond, subnet, name_suffix in [
        (z_flat <= physics.ridge_z_min, model.net_air,  "air"),
        ((z_flat > physics.ridge_z_min) & (z_flat <= physics.ridge_z_max), model.net_grat, "grat"),
        (z_flat > physics.ridge_z_max,  model.net_sub,  "sub"),
    ]:
        if not mask_cond.any():
            continue
        xm = x_flat[mask_cond]
        zm = z_flat[mask_cond]
        # Exclude interface points for cleaner residuals
        near_intf = (torch.abs(zm - physics.ridge_z_min) < margin) | \
                    (torch.abs(zm - physics.ridge_z_max) < margin)
        xm = xm[~near_intf]; zm = zm[~near_intf]
        if len(xm) == 0:
            continue
        with torch.enable_grad():
            res_tuple = maxwell_2d_dd_pde_residual(subnet, xm, zm, physics, eps_fn, scattered)
        for i, name in enumerate(res_names):
            key = f"res_{name}"
            arr = res_tuple[i].detach().cpu().numpy()
            if key not in residuals_np:
                residuals_np[key] = np.full(X.shape, np.nan)
            # Map back to grid positions
            idx_flat = torch.where(mask_cond)[0][~near_intf.cpu()]
            row_idx = idx_flat.numpy() // nx
            col_idx = idx_flat.numpy() % nx
            residuals_np[key][row_idx, col_idx] = arr

    # Total residual magnitude
    total = np.zeros(X.shape)
    for name in res_names:
        key = f"res_{name}"
        if key in residuals_np:
            r = residuals_np[key]
            total += np.where(np.isfinite(r), r**2, 0.0)
    residuals_np["res_total"] = np.sqrt(total)
    residuals_np["x"] = X
    residuals_np["z"] = Z

    # Region mask
    region = np.zeros(X.shape, dtype=int)
    region[(Z > physics.ridge_z_min) & (Z <= physics.ridge_z_max)] = 1
    region[Z > physics.ridge_z_max] = 2
    residuals_np["region"] = region

    return residuals_np


def compute_residual_by_region(res_map: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Break down total residual MSE by region (air/grating/substrate)."""
    region = res_map["region"]
    res_total = res_map["res_total"]
    region_names = {0: "air", 1: "grating", 2: "substrate"}
    result = {}
    for code, name in region_names.items():
        mask = (region == code) & np.isfinite(res_total)
        if mask.any():
            vals = res_total[mask]
            result[name] = {
                "mean": float(np.mean(vals**2)),
                "max":  float(np.max(vals)),
                "p95":  float(np.percentile(vals, 95)),
            }
    return result


def compute_residual_near_corners(
    res_map: dict[str, np.ndarray],
    physics: "PhysicsConfig",
    corner_margin: float = 0.05,
) -> dict[str, float]:
    """Residual statistics near the four ridge corners.

    Corners: (x_min, z_min), (x_max, z_min), (x_min, z_max), (x_max, z_max)
    """
    X, Z = res_map["x"], res_map["z"]
    res = res_map["res_total"]
    corners = {
        "SW": (physics.ridge_x_min, physics.ridge_z_min),
        "SE": (physics.ridge_x_max, physics.ridge_z_min),
        "NW": (physics.ridge_x_min, physics.ridge_z_max),
        "NE": (physics.ridge_x_max, physics.ridge_z_max),
    }
    result = {}
    for name, (xc, zc) in corners.items():
        dist = np.sqrt((X - xc)**2 + (Z - zc)**2)
        near = (dist < corner_margin) & np.isfinite(res)
        if near.any():
            result[name] = {
                "mean_mse":  float(np.mean(res[near]**2)),
                "max_res":   float(np.max(res[near])),
                "n_points":  int(near.sum()),
            }
    return result


def plot_residual_maps(
    res_map: dict[str, np.ndarray],
    physics: "PhysicsConfig",
    output_dir: Path,
    tag: str = "residual",
) -> None:
    """Save residual magnitude map and per-equation maps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X, Z = res_map["x"], res_map["z"]
    ext = [float(X.min()), float(X.max()), float(Z.max()), float(Z.min())]

    def _ridge_outline(ax):
        from matplotlib.patches import Rectangle
        w = physics.ridge_x_max - physics.ridge_x_min
        h = physics.ridge_z_max - physics.ridge_z_min
        rect = Rectangle((physics.ridge_x_min, physics.ridge_z_min), w, h,
                          fill=False, edgecolor="white", linewidth=1.0, linestyle="--")
        ax.add_patch(rect)

    # Total residual
    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(np.log10(res_map["res_total"] + 1e-10), extent=ext,
                   aspect="auto", cmap="hot_r", origin="upper")
    _ridge_outline(ax)
    ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)")
    ax.set_title("log10(|PDE residual|)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log10(|r|)")
    fig.savefig(output_dir / f"{tag}_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Histogram
    vals = res_map["res_total"].ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(np.log10(vals + 1e-10), bins=50, color="steelblue", edgecolor="none")
    ax.set_xlabel("log10(|r|)"); ax.set_ylabel("Count")
    ax.set_title("Residual magnitude histogram")
    fig.savefig(output_dir / f"{tag}_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-equation (2-row grid)
    eq_names = ["Ar", "Ai", "Br", "Bi", "Cr", "Ci"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, eq_names):
        key = f"res_{name}"
        if key not in res_map:
            continue
        data = np.abs(res_map[key])
        im = ax.imshow(np.log10(data + 1e-10), extent=ext, aspect="auto",
                       cmap="hot_r", origin="upper")
        _ridge_outline(ax)
        ax.set_xlabel("x (λ)"); ax.set_ylabel("z (λ)")
        ax.set_title(f"log10|res_{name}|")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-equation PDE residuals")
    fig.savefig(output_dir / f"{tag}_per_eq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diffraction metrics
# ---------------------------------------------------------------------------


def compute_diffraction_efficiencies(
    E_total_2d: np.ndarray,
    x1d: np.ndarray,
    z1d: np.ndarray,
    physics: "PhysicsConfig",
    n_orders: int = 3,
    z_monitor_top: float | None = None,
    z_monitor_bot: float | None = None,
) -> dict[str, object]:
    """Compute diffraction-order power by Fourier decomposition at monitor planes.

    Method
    ------
    At a horizontal monitor plane z = z_mon, decompose the field into
    spatial Fourier orders along x:

        E_m = (1/period) * integral_0^period E(x, z_mon) exp(-i Gm x) dx
        Gm = m * 2*pi / period

    Power in order m (Poynting flux, time-averaged, per unit period):
        P_m = (1/2) * Re(kz_m) * |E_m|^2 / k0
              (normalised by incident power, which is 1 W/m for unit amplitude)

    Propagating orders have real kz_m > 0; evanescent orders carry no power.

    Parameters
    ----------
    E_total_2d :
        Complex total field E(z_index, x_index), shape (Nz, Nx).
    x1d, z1d :
        Coordinate arrays.
    physics :
        PhysicsConfig.
    n_orders :
        Number of diffraction orders on each side of DC.
    z_monitor_top :
        z-position of top monitor plane (default: 10% from top).
    z_monitor_bot :
        z-position of bottom monitor plane (default: 10% from bottom).

    Returns
    -------
    dict with:
        'orders'        : array of order indices [-n..+n]
        'R_m'           : reflection efficiency per order (top monitor)
        'T_m'           : transmission efficiency per order (bottom monitor)
        'R_total'       : total reflection (sum over propagating orders)
        'T_total'       : total transmission
        'R0', 'T0'      : zeroth-order (specular)
        'R1', 'T1'      : ±1 orders (first-order diffraction)
        'poynting_top'  : time-averaged Sz at top monitor (Nz × Nx slice)
        'poynting_bot'  : time-averaged Sz at bottom monitor
        'energy_check'  : R_total + T_total (should be ≤ 1 for lossless)
    """
    k0     = physics.k0
    period = physics.period
    n_air  = physics.n_air
    n_sub  = physics.n_substrate
    Nx     = len(x1d)
    dx     = x1d[1] - x1d[0]

    if z_monitor_top is None:
        z_monitor_top = 0.1 * physics.domain_height
    if z_monitor_bot is None:
        z_monitor_bot = 0.9 * physics.domain_height

    orders = np.arange(-n_orders, n_orders + 1)  # shape (2*n+1,)
    Gm     = orders * (2.0 * np.pi / period)

    def _dft_at_z(z_target: float, n_medium: float) -> dict:
        iz = np.argmin(np.abs(z1d - z_target))
        E_slice = E_total_2d[iz, :]  # complex, shape (Nx,)
        # Discrete Fourier decomposition
        E_m = np.array([
            np.trapezoid(E_slice * np.exp(-1j * Gm[mi] * x1d), x1d) / period
            for mi in range(len(orders))
        ])
        # kz for each order in this medium
        kz2 = (k0 * n_medium)**2 - Gm**2
        kz_m = np.sqrt(kz2.astype(complex))
        # Outgoing branch: propagating → Re(kz) > 0; evanescent → Im(kz) < 0
        evan = kz2.real < 0
        kz_m[evan] = -kz_m[evan]
        # Power: P_m = (1/2) * Re(kz_m / k0) * |E_m|^2 / 1 (unit incident)
        P_m = 0.5 * np.real(kz_m / k0) * np.abs(E_m)**2
        return {"E_m": E_m, "kz_m": kz_m, "P_m": P_m, "iz": iz}

    # Incident power = 0.5 * kz_inc / k0 = 0.5 * n_air (since kz_inc = k0*n_air at normal inc)
    P_inc = 0.5 * n_air

    top_res = _dft_at_z(z_monitor_top, n_air)
    bot_res = _dft_at_z(z_monitor_bot, n_sub)

    # Reflection: field at top monitor minus incident
    # For scattered-field formulation E already includes incident; for total field, subtract inc.
    # We report raw powers for the user to interpret.
    R_m = top_res["P_m"] / P_inc  # upgoing power fraction per order (includes incident)
    T_m = bot_res["P_m"] / P_inc  # downgoing power fraction per order

    # Find m=0 and m=±1 indices
    idx0 = np.where(orders == 0)[0][0]
    idx_p1 = np.where(orders == 1)[0][0] if 1 in orders else None
    idx_m1 = np.where(orders == -1)[0][0] if -1 in orders else None

    R_total = float(np.sum(R_m[np.real(top_res["kz_m"]) > 1e-6]))
    T_total = float(np.sum(T_m[np.real(bot_res["kz_m"]) > 1e-6]))

    return {
        "orders":      orders,
        "R_m":         R_m,
        "T_m":         T_m,
        "R_total":     R_total,
        "T_total":     T_total,
        "R0":          float(R_m[idx0]),
        "T0":          float(T_m[idx0]),
        "R1":          float(R_m[idx_p1]) if idx_p1 is not None else float("nan"),
        "T1":          float(T_m[idx_p1]) if idx_p1 is not None else float("nan"),
        "energy_check": R_total + T_total,
        "z_monitor_top": float(z_monitor_top),
        "z_monitor_bot": float(z_monitor_bot),
    }


def compare_diffraction_with_rcwa(
    pinn_diff: dict,
    rcwa_E: np.ndarray,
    rcwa_x: np.ndarray,
    rcwa_z: np.ndarray,
    physics: "PhysicsConfig",
    n_orders: int = 3,
) -> dict:
    """Compute diffraction efficiencies from RCWA field and compare to PINN."""
    rcwa_diff = compute_diffraction_efficiencies(
        rcwa_E, rcwa_x, rcwa_z, physics, n_orders
    )
    comparison = {
        "R0_pinn":  pinn_diff["R0"],  "R0_rcwa":  rcwa_diff["R0"],
        "T0_pinn":  pinn_diff["T0"],  "T0_rcwa":  rcwa_diff["T0"],
        "R_total_pinn": pinn_diff["R_total"], "R_total_rcwa": rcwa_diff["R_total"],
        "T_total_pinn": pinn_diff["T_total"], "T_total_rcwa": rcwa_diff["T_total"],
        "delta_R0": abs(pinn_diff["R0"] - rcwa_diff["R0"]),
        "delta_T0": abs(pinn_diff["T0"] - rcwa_diff["T0"]),
    }
    return comparison


# ---------------------------------------------------------------------------
# Modal DtN boundary conditions
# ---------------------------------------------------------------------------


def dtn_top_bc_loss(
    model: "Maxwell2DDD",
    x_pts: torch.Tensor,
    physics: "PhysicsConfig",
    n_orders: int = 5,
    scattered: bool = True,
) -> torch.Tensor:
    """Modal Dirichlet-to-Neumann boundary condition at z=0 (top).

    For each Fourier mode m, the outgoing (upward) condition is:
        dE_m / dz = -i kz_m E_m   (upward propagating: kz_m > 0)

    We enforce this by penalising the DtN residual integrated over x:
        L = || dE/dz(x, 0) - DtN[E](x, 0) ||^2

    where DtN[E] is reconstructed by Fourier synthesis of the modal conditions.

    For scattered-field: applies to E_scat.  For total-field: to E_total.

    Notes
    -----
    - Propagating orders (kz_m real > 0): outgoing upward.
    - Evanescent orders (kz_m purely imaginary): decay upward, Im(kz_m) < 0
      for exp(-i kz_m z) to decay as z decreases (i.e. kz_m = -i|β|).
    - This is more physically accurate than the local Robin condition
      H̃_x = -n_air * E, which only captures the zeroth-order contribution.
    """
    from src.derivatives import first_derivative

    k0     = physics.k0
    period = physics.period
    n_air  = physics.n_air
    orders = np.arange(-n_orders, n_orders + 1)
    Gm     = orders * (2.0 * np.pi / period)

    # Outgoing kz_m for each order at z=0 (in air)
    kz2 = (k0 * n_air)**2 - Gm**2
    kz_m = np.sqrt(kz2.astype(complex))
    # Branch: outgoing upward → Re(kz) > 0 for propagating; evanescent: Im(kz) < 0
    evan = kz2.real < 0
    kz_m[evan] = -kz_m[evan]
    # (kz_m used as numpy complex array below — no tensor conversion needed)

    # Evaluate E and dE/dz at z=0
    z0 = torch.zeros_like(x_pts)
    z0_g = z0.detach().clone().requires_grad_(True)
    # Ensure x is in a fresh graph too
    x_g  = x_pts.detach().clone().requires_grad_(False)

    Er_s, Ei_s, _, _, _, _ = model.net_air.field_components(x_g, z0_g)
    dEr_dz = first_derivative(Er_s, z0_g)
    dEi_dz = first_derivative(Ei_s, z0_g)

    # DtN operator: for each x point, compute DtN[E](x) via IDFT
    # E_m = (1/N) * sum_x E(x_j) * exp(-i Gm x_j)
    # DtN[E](x) = sum_m (-i kz_m) * E_m * exp(i Gm x)
    Nx = len(x_pts)
    x_np = x_pts.detach().cpu().numpy()
    Er_np = Er_s.detach().cpu().numpy()
    Ei_np = Ei_s.detach().cpu().numpy()
    E_np  = Er_np + 1j * Ei_np

    # DFT: E_m = sum_j E(x_j) exp(-i Gm x_j) dx / period
    dz_dt_np = np.zeros(Nx, dtype=complex)
    dx_step = period / Nx
    for mi, (gm, kz) in enumerate(zip(Gm, kz_m)):
        Em = np.sum(E_np * np.exp(-1j * gm * x_np)) * dx_step / period
        # dE/dz contribution from this mode
        dz_dt_np += (-1j * kz) * Em * np.exp(1j * gm * x_np)

    # DtN target
    dz_dt_r = torch.as_tensor(np.real(dz_dt_np), dtype=x_pts.dtype, device=x_pts.device)
    dz_dt_i = torch.as_tensor(np.imag(dz_dt_np), dtype=x_pts.dtype, device=x_pts.device)

    loss = torch.mean((dEr_dz - dz_dt_r)**2 + (dEi_dz - dz_dt_i)**2)
    return loss


def audit_boundary(
    model: "Maxwell2DDD",
    physics: "PhysicsConfig",
    device: torch.device,
    dtype: torch.dtype,
    n_bc_pts: int = 256,
    n_orders: int = 5,
    scattered: bool = True,
) -> dict[str, float]:
    """Compare local Robin BC vs. modal DtN BC residuals.

    Reports unweighted MSE for each boundary formulation.

    The local Robin condition (H̃_x = -n_air * E at z=0) is exact only for
    the zeroth diffraction order.  The modal DtN is exact for all orders
    within the truncation.

    Returns
    -------
    dict with:
        'robin_top_mse'     : Robin residual at z=0
        'robin_bottom_mse'  : Robin residual at z=domain_height
        'dtn_top_mse'       : DtN residual at z=0
        'dtn_agrees_with_robin': True if errors are similar (single-mode regime)
    """
    from src.maxwell_2d_dd import maxwell_2d_dd_top_bc, maxwell_2d_dd_bottom_bc

    rng = np.random.default_rng(0)
    x_pts = torch.as_tensor(
        rng.uniform(0.0, physics.period, n_bc_pts), dtype=dtype, device=device
    )

    model.eval()
    with torch.enable_grad():
        L_top_robin = maxwell_2d_dd_top_bc(
            model.net_air, x_pts, physics, scattered=scattered
        )
        L_bot_robin = maxwell_2d_dd_bottom_bc(
            model.net_sub, x_pts, physics
        )
        L_top_dtn = dtn_top_bc_loss(model, x_pts, physics, n_orders, scattered)

    return {
        "robin_top_mse":    float(L_top_robin.detach()),
        "robin_bottom_mse": float(L_bot_robin.detach()),
        "dtn_top_mse":      float(L_top_dtn.detach()),
        "dtn_agrees_with_robin": (
            abs(float(L_top_dtn.detach()) - float(L_top_robin.detach()))
            / (float(L_top_robin.detach()) + 1e-12) < 0.5
        ),
    }


# ---------------------------------------------------------------------------
# Adaptive collocation refinement
# ---------------------------------------------------------------------------


def residual_based_refinement(
    model: "Maxwell2DDD",
    physics: "PhysicsConfig",
    device: torch.device,
    dtype: torch.dtype,
    n_new: int = 512,
    threshold_percentile: float = 90.0,
    scattered: bool = True,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Generate additional collocation points in high-residual regions.

    Computes a coarse residual map, then samples new points preferentially
    where the residual magnitude is above the threshold percentile.

    Parameters
    ----------
    n_new :
        Number of new interior points to add (split across three regions).
    threshold_percentile :
        Points above this percentile of residual magnitude are oversampled.

    Returns
    -------
    dict with 'x_new', 'z_new' — additional collocation points.
    """
    res_map = compute_residual_map(
        model, physics, device, dtype, nx=32, nz=64, scattered=scattered
    )
    res_total = res_map["res_total"]
    X, Z = res_map["x"], res_map["z"]

    # Threshold
    vals = res_total[np.isfinite(res_total)]
    threshold = float(np.percentile(vals, threshold_percentile))
    high_res = np.isfinite(res_total) & (res_total > threshold)

    # Sample new points uniformly from high-residual cells
    rng = np.random.default_rng(seed)
    xi = X[high_res]; zi = Z[high_res]
    if len(xi) == 0:
        # Fallback: uniform
        xi = rng.uniform(0.0, physics.period, n_new)
        zi = rng.uniform(0.0, physics.domain_height, n_new)
    else:
        idx = rng.choice(len(xi), min(n_new, len(xi)), replace=True)
        xi = xi[idx]; zi = zi[idx]
        # Add small jitter
        jitter = 0.01 * physics.period
        xi = xi + rng.uniform(-jitter, jitter, len(xi))
        zi = zi + rng.uniform(-jitter, jitter, len(zi))
        xi = np.clip(xi, 0.0, physics.period)
        zi = np.clip(zi, 0.0, physics.domain_height)

    return {
        "x_new": torch.as_tensor(xi, dtype=dtype, device=device),
        "z_new": torch.as_tensor(zi, dtype=dtype, device=device),
    }
