"""Independent radial-depth geometry checks."""

from collections import namedtuple
import math
import unittest

from scripts.mve_spike.evaluate_depth import truth_depth


class DepthGeometryTests(unittest.TestCase):
    def test_off_axis_plane_is_radial_distance_not_camera_z(self):
        Camera = namedtuple("Camera", "k rotation translation full_size")
        camera = Camera([10, 0, 0.5, 0, 10, 0.5, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0], (10, 10))
        plane = {"x": [5.9, 6.1], "y": [7.9, 8.1], "z": 10}
        self.assertAlmostEqual(truth_depth(6, 8, 10, 10, camera, [plane]), 10 * math.sqrt(2))
        self.assertIsNone(truth_depth(0, 0, 10, 10, camera, [plane]))


if __name__ == "__main__":
    unittest.main()
