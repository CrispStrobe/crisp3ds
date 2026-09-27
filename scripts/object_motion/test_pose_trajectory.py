"""Synthetic geometry diagnostics; no reference scan or real photos required."""

import math
from unittest import TestCase

import numpy as np

from scripts.object_motion.pose_trajectory import trajectory


def orbit(*, folded=False, raised=None):
    rows = []
    for angle in range(0, 360, 30):
        theta = math.radians(angle * (2 if folded else 1))
        center = np.asarray([math.cos(theta), math.sin(theta),
                             0.8 if angle == raised else 0.0])
        forward = np.asarray([-math.cos(theta), -math.sin(theta), 0.0])
        up = np.asarray([0.0, 0.0, 1.0])
        right = np.cross(up, forward)
        rotation = np.column_stack([right, up, forward])
        rows.append({"name": f"NP3_{angle:03d}.jpg", "center": center,
                     "camera_to_world_rotation": rotation})
    return rows


class PoseTrajectoryTests(TestCase):
    def test_correct_circle_has_opposing_centers_and_rotations(self):
        result = trajectory(orbit())
        self.assertEqual(result["registered"], 12)
        self.assertAlmostEqual(result["median_planar_radius_model_units"], 1.0, places=8)
        self.assertLess(result["plane_residual_over_median_radius"]["max"], 1e-8)
        self.assertAlmostEqual(result["opposing_center_distance_over_median_radius"]["median"],
                               2.0, places=8)
        self.assertAlmostEqual(result["opposing_full_orientation_angle_degrees"]["median"],
                               180.0, places=8)
        self.assertAlmostEqual(result["adjacent_full_orientation_step_degrees"]["median"],
                               30.0, places=8)
        self.assertEqual(result["closing_pair"]["wrapped_order_label_step"], 30)
        self.assertAlmostEqual(result["closing_pair"]["full_orientation_step_degrees"],
                               30.0, places=8)

    def test_deliberate_180_fold_changes_center_and_orientation_diagnostics(self):
        result = trajectory(orbit(folded=True))
        self.assertLess(result["opposing_center_distance_over_median_radius"]["max"], 1e-8)
        self.assertLess(result["opposing_full_orientation_angle_degrees"]["max"], 1e-5)
        self.assertAlmostEqual(result["adjacent_full_orientation_step_degrees"]["median"],
                               60.0, places=8)

    def test_outlier_and_invalid_pose_are_exposed(self):
        result = trajectory(orbit(raised=90))
        self.assertEqual(result["largest_plane_residuals"][0]["name"], "NP3_090.jpg")
        self.assertGreater(result["plane_residual_over_median_radius"]["max"], 0.5)
        invalid = orbit()
        invalid[0]["camera_to_world_rotation"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            trajectory(invalid)
