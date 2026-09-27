"""Analytic depth-grid to RGB projection checks; no Berkeley files required."""

import unittest

import numpy as np

from scripts.object_motion import berkeley_projection_check as check


class BerkeleyProjectionTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.array([[10000, 10000], [10000, 0]], dtype=np.uint16)
        self.k = np.eye(3, dtype=np.float64)
        self.distortion = np.zeros(5, dtype=np.float64)

    def test_identity_integer_centers(self):
        uv, valid = check.project(self.depth, self.k, self.k, self.distortion, np.eye(4))
        self.assertEqual(uv[0, 0].tolist(), [0, 0])
        self.assertEqual(uv[0, 1].tolist(), [1, 0])
        self.assertEqual(uv[1, 0].tolist(), [0, 1])
        self.assertFalse(valid[1, 1])
        self.assertTrue(np.isnan(uv[1, 1]).all())

    def test_depth_pixel_half_shift_is_explicit(self):
        uv, _ = check.project(self.depth, self.k, self.k, self.distortion, np.eye(4), pixel_shift=.5)
        self.assertEqual(uv[0, 0].tolist(), [.5, .5])

    def test_extrinsic_translation_and_positive_depth(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, -1, 0]
        uv, valid = check.project(self.depth, self.k, self.k, self.distortion, transform)
        self.assertEqual(uv[0, 0].tolist(), [1, -1])
        self.assertTrue(valid[0, 0])
        transform[2, 3] = -2
        _, behind = check.project(self.depth, self.k, self.k, self.distortion, transform)
        self.assertFalse(behind.any())

    def test_rgb_brown_radial_distortion(self):
        distortion = np.array([.1, 0, 0, 0, 0], dtype=np.float64)
        distorted, _ = check.project(self.depth, self.k, self.k, distortion, np.eye(4))
        plain, _ = check.project(self.depth, self.k, self.k, distortion, np.eye(4), apply_distortion=False)
        self.assertAlmostEqual(distorted[0, 1, 0], 1.1, places=12)
        self.assertAlmostEqual(plain[0, 1, 0], 1.0, places=12)

    def test_depth_edges_require_two_valid_neighbors_and_two_cm(self):
        depth = np.array([[10000, 10100, 0], [10000, 10300, 0]], dtype=np.uint16)
        edges = check.depth_edges(depth)
        self.assertFalse(edges[0, 0])  # 1 cm is below the fixed threshold.
        self.assertTrue(edges[1, 0])   # 3 cm exceeds it.
        self.assertFalse(edges[:, 2].any())  # Missing depth is never an edge source.


if __name__ == "__main__":
    unittest.main()
