"""Geometry and population tests for the independent three-view checker."""

import math
import unittest

import numpy as np

from scripts.three_view_numpy import check


def pose(center, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    center = np.asarray(center, dtype=float)
    return {"R": rotation, "t": -rotation @ center, "C": center}


def observe(point, camera_pose, intrinsics):
    f, cx, cy = intrinsics
    x, y, z = camera_pose["R"] @ point + camera_pose["t"]
    return [f * x / z + cx, f * y / z + cy]


class ThreeViewTests(unittest.TestCase):
    def setUp(self):
        self.camera = (700.0, 384.0, 256.0)
        self.point = np.array([0.2, 0.1, 5.0])
        self.poses = {"a": pose([0, 0, 0], 0), "b": pose([1, 0, 0], 0.1),
                      "c": pose([0.2, 0.6, 0], -0.05)}
        self.features = {name: [observe(self.point, view, self.camera)]
                         for name, view in self.poses.items()}
        self.cycle = (("a", 0), ("b", 0), ("c", 0))

    def test_nonidentity_pose_triangulates_known_point(self):
        result = check.evaluate_cycle(self.cycle, self.poses, self.features, self.camera)
        self.assertEqual(result["status"], "finite")
        self.assertLess(result["error_px"], 1e-10)
        np.testing.assert_allclose(result["point_xyz"], self.point, atol=1e-12)
        self.assertGreater(result["parallax_deg"], 1)

    def test_third_view_outlier_is_measured_without_refitting(self):
        self.features["c"][0][0] += 20.0
        result = check.evaluate_cycle(self.cycle, self.poses, self.features, self.camera)
        self.assertEqual(result["status"], "finite")
        self.assertAlmostEqual(result["error_px"], 20.0, places=9)

    def test_behind_camera_and_parallel_rays_are_reported(self):
        behind = dict(self.poses)
        behind["c"] = pose([0.2, 0.6, 0], math.pi)
        self.assertEqual(check.evaluate_cycle(self.cycle, behind, self.features, self.camera)["status"],
                         "nonpositive_depth")
        parallel = dict(self.poses)
        parallel["b"] = pose([1, 0, 0], 0)
        features = dict(self.features)
        features["b"] = [self.features["a"][0][:]]
        self.assertEqual(check.evaluate_cycle(self.cycle, parallel, features, self.camera)["status"],
                         "low_parallax")

    def test_triangle_enumeration_deduplicates_and_sorts(self):
        rows = [{"image1": a, "feature1": 0, "image2": b, "feature2": 0}
                for a, b in (("a", "b"), ("a", "c"), ("b", "c"), ("b", "a"))]
        self.assertEqual(check.closed_cycles(rows), [self.cycle])


if __name__ == "__main__":
    unittest.main()
