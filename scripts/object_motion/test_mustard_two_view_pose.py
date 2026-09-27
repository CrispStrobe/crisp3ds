"""Synthetic and fail-closed checks for the independent two-view audit."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.object_motion import mustard_two_view_pose as target


class TwoViewPoseTests(unittest.TestCase):
    def test_rotation_and_ray_angle(self):
        identity = np.eye(3)
        self.assertAlmostEqual(target.rotation_difference_degrees(identity, identity), 0)
        ninety = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        self.assertAlmostEqual(target.rotation_difference_degrees(identity, ninety), 90)
        angle = target.ray_angles_degrees(np.array([[0., 0.]]), np.array([[.1, 0.]]),
                                         identity, np.array([True]))
        self.assertEqual(len(angle), 1)
        self.assertGreater(angle[0], 5)

    def test_multiple_essential_solutions_are_unavailable_for_all_seeds(self):
        row = {"stratum": "opposing", "left": "a", "right": "b",
               "cyclic_separation": 24, "verified_correspondences": 25,
               "two_view_config": 3, "points_left": np.zeros((25, 2)),
               "points_right": np.ones((25, 2))}
        with patch.object(target.cv2, "findEssentialMat",
                          return_value=(np.zeros((6, 3)), np.ones((25, 1), "uint8"))), \
                patch.object(target.cv2, "findHomography", return_value=(None, None)), \
                patch.object(target.cv2, "recoverPose", side_effect=AssertionError("must not pick first E")):
            result = target.fit_pair(row)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual([fit["seed"] for fit in result["seeds"]], list(target.SEEDS))
        self.assertTrue(all("unique 3x3" in fit["reason"] for fit in result["seeds"]))

    def test_essential_pose_api_on_synthetic_nonplanar_correspondences(self):
        rng = np.random.default_rng(19)
        points = rng.uniform([-1, -1, 2], [1, 1, 5], size=(80, 3))
        angle = np.deg2rad(18)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        moved = points @ rotation.T + [.5, 0, 0]
        left = points[:, :2] / points[:, 2, None]
        right = moved[:, :2] / moved[:, 2, None]
        row = {"stratum": "near", "left": "synthetic_a", "right": "synthetic_b",
               "cyclic_separation": 1, "verified_correspondences": 80,
               "two_view_config": 3, "points_left": left, "points_right": right}
        result = target.fit_pair(row)
        self.assertEqual(len(result["seeds"]), 3)
        self.assertTrue(all(fit["status"] == "passed" for fit in result["seeds"]))
        self.assertEqual(result["status"], "distinct")
        self.assertAlmostEqual(result["median_rotation_degrees"], 18, delta=2)
        for fit in result["seeds"]:
            self.assertGreater(np.dot(fit["translation_unit"], [1, 0, 0]), .99)
            self.assertGreater(fit["median_inter_ray_parallax_degrees"], 1)

    def test_analytic_planar_scene_withholds_physical_rotation(self):
        rng = np.random.default_rng(21)
        points = np.column_stack((rng.uniform(-1, 1, 80), rng.uniform(-1, 1, 80),
                                  np.full(80, 3.)))
        angle = np.deg2rad(18)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        moved = points @ rotation.T + [.5, 0, 0]
        row = {"stratum": "opposing", "left": "planar_a", "right": "planar_b",
               "cyclic_separation": 23, "verified_correspondences": 80,
               "two_view_config": 6,
               "points_left": points[:, :2] / points[:, 2, None],
               "points_right": moved[:, :2] / moved[:, 2, None]}
        result = target.fit_pair(row)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("homography", result["reason"])
        self.assertEqual(result["homography_inliers"], 80)


if __name__ == "__main__":
    unittest.main()
