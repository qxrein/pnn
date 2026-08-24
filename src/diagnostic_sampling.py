"""Deterministic, region-aware samples for Phase 5 diagnostics."""
from __future__ import annotations

import numpy as np


def deterministic_region_samples(physics, n_per_region: int, n_interface: int,
                                 n_vertical: int, n_corner: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Return named physical-coordinate samples, each shaped ``(N, 2)``.

    Interior groups exclude the ridge interfaces by a small deterministic
    margin. Horizontal samples span the material boundaries; vertical samples
    follow both ridge sidewalls; corner samples are local two-dimensional
    patches around the four geometric corners.
    """
    rng = np.random.default_rng(seed)
    p = physics
    margin = min(1e-3, p.ridge_height / 20)
    def rect(x0, x1, z0, z1, n):
        return np.column_stack((rng.uniform(x0, x1, n), rng.uniform(z0, z1, n)))
    samples = {
        "air": rect(0, p.period, margin, p.ridge_z_min - margin, n_per_region),
        "ridge": rect(p.ridge_x_min + margin, p.ridge_x_max - margin,
                      p.ridge_z_min + margin, p.ridge_z_max - margin, n_per_region),
        "substrate": rect(0, p.period, p.ridge_z_max + margin, p.domain_height - margin, n_per_region),
    }
    x = np.linspace(0, p.period, n_interface, endpoint=False)
    samples["horizontal_top"] = np.column_stack((x, np.full_like(x, p.ridge_z_min)))
    samples["horizontal_bottom"] = np.column_stack((x, np.full_like(x, p.ridge_z_max)))
    z = np.linspace(p.ridge_z_min, p.ridge_z_max, n_vertical)
    samples["vertical_left"] = np.column_stack((np.full_like(z, p.ridge_x_min), z))
    samples["vertical_right"] = np.column_stack((np.full_like(z, p.ridge_x_max), z))
    for name, x0, z0 in (("corner_nw", p.ridge_x_min, p.ridge_z_min),
                         ("corner_ne", p.ridge_x_max, p.ridge_z_min),
                         ("corner_sw", p.ridge_x_min, p.ridge_z_max),
                         ("corner_se", p.ridge_x_max, p.ridge_z_max)):
        samples[name] = rect(x0 - margin, x0 + margin, z0 - margin, z0 + margin, n_corner)
    return samples
