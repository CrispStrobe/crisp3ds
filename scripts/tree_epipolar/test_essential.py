#!/usr/bin/env python3
"""Known synthetic camera pose, outliers, and frozen holdout for essential.py."""

import unittest

import numpy as np

from essential import assess, direction_angle_deg, rotation_angle_deg


class EssentialTests(unittest.TestCase):
    def test_known_camera_with_outliers_and_heldout_isolation(self):
        rng = np.random.default_rng(23)
        k = np.array([[500., 0, 384.], [0, 500., 256.], [0, 0, 1.]])
        views = [dict(id=1, K=k.tolist(), R=np.eye(3).tolist(), t=[0., 0., 0.]),
                 dict(id=4, K=k.tolist(), R=np.eye(3).tolist(), t=[-.5, 0., 0.])]
        points = rng.uniform([-1., -.7, 2.], [1., .7, 6.], (100, 3))
        a = (k @ points.T).T
        b = (k @ (points + np.array([-.5, 0., 0.])).T).T
        a, b = a[:, :2]/a[:, 2:], b[:, :2]/b[:, 2:]
        a += rng.normal(0, .08, a.shape)
        b += rng.normal(0, .08, b.shape)
        b[[2, 13, 24, 37, 58]] += [0, 45]
        rows = [dict(order=i, a_xy=a[i].tolist(), b_xy=b[i].tolist(),
                     split="holdout" if i % 5 == 4 else "train") for i in range(len(points))]
        result = assess(views, rows)
        self.assertGreater(result["essential_ransac_training_inliers"], 65)
        self.assertGreater(result["recover_pose_positive_depth_training_count"], 50)
        raw = result["raw_triangulation_training_inliers"]
        self.assertEqual(raw["total"], result["essential_ransac_training_inliers"])
        self.assertEqual(raw["positive_both"] + raw["nonpositive_either"], raw["finite"])
        self.assertGreaterEqual(raw["positive_both"], result["recover_pose_positive_depth_training_count"])
        self.assertLess(result["rotation_difference_deg"], 3.)
        self.assertLess(result["translation_direction_difference_deg"], 10.)
        self.assertLess(result["calibrated_heldout"]["median_px"], 1.)
        self.assertGreater(result["training_positive_depth_ray_angle_median_deg"], 2.)
        rows[4]["b_xy"] = [100., 100.]
        changed = assess(views, rows)
        self.assertTrue(np.allclose(result["calibrated_F"], changed["calibrated_F"]))

    def test_angles(self):
        self.assertAlmostEqual(rotation_angle_deg(np.eye(3), np.eye(3)), 0.)
        self.assertAlmostEqual(direction_angle_deg([1,0,0], [-1,0,0]), 180.)


if __name__ == "__main__":
    unittest.main()
