import numpy as np

from src.config import PhysicsConfig
from src.diagnostic_sampling import deterministic_region_samples


def test_region_sampling_is_deterministic_and_complete():
    p = PhysicsConfig()
    a = deterministic_region_samples(p, 10, 8, 6, 4, seed=42)
    b = deterministic_region_samples(p, 10, 8, 6, 4, seed=42)
    expected = {"air", "ridge", "substrate", "horizontal_top", "horizontal_bottom",
                "vertical_left", "vertical_right", "corner_nw", "corner_ne", "corner_sw", "corner_se"}
    assert set(a) == expected
    for key in expected:
        assert a[key].ndim == 2 and a[key].shape[1] == 2
        np.testing.assert_allclose(a[key], b[key])


def test_region_sampling_places_interior_points_in_requested_regions():
    p = PhysicsConfig()
    s = deterministic_region_samples(p, 32, 8, 6, 4)
    assert np.all(s["air"][:, 1] < p.ridge_z_min)
    assert np.all((s["ridge"][:, 0] > p.ridge_x_min) & (s["ridge"][:, 0] < p.ridge_x_max))
    assert np.all(s["substrate"][:, 1] > p.ridge_z_max)
