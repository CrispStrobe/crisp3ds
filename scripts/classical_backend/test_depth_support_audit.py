import unittest

import numpy as np

from scripts.classical_backend import depth_support_audit as audit


class DepthSupportGoldenTests(unittest.TestCase):
    def test_simple_radial_inverse_roundtrip_with_pixel_centers(self):
        params = np.array([1000.0, 640.0, 512.0, -0.05])
        rays = np.array([[0, 0], [0.24, -.16], [-.32, .27]], dtype=float)
        r2 = np.sum(rays * rays, axis=1)
        pixels = rays * (1 + params[3] * r2)[:, None] * params[0] + params[1:3]
        np.testing.assert_allclose(audit.radial_inverse(pixels, params), rays, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            audit.radial_inverse(np.array([[1100., 512.]]), [1000, 640, 512, -100])

    def test_native_half_pixel_and_nearest_rounding(self):
        k = np.array([[10., 0., -.5], [0., 10., -.5], [0., 0., 1.]])
        rays = np.array([[0., 0.], [.1, .1], [.049, .049], [-.06, 0.], [1., 0.]])
        x, y, inside = audit.depth_indices(rays, k, (8, 8))
        np.testing.assert_array_equal(inside, [True, True, True, False, False])
        np.testing.assert_array_equal(x[:3], [0, 1, 0])
        np.testing.assert_array_equal(y[:3], [0, 1, 0])

    def test_frozen_original_rgb_probe_lattice(self):
        mask = np.zeros((1024, 1280), dtype=bool)
        mask[2, 2] = mask[6, 6] = mask[2, 3] = True
        np.testing.assert_array_equal(audit.original_probe_pixels(mask), [[2, 2], [6, 6]])
        with self.assertRaisesRegex(ValueError, "original"):
            audit.original_probe_pixels(mask[1:])


if __name__ == "__main__":
    unittest.main()
