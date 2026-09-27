"""Analytic camera-center and proper Sim(3) checks."""

import unittest

import numpy as np

from scripts.upstream_control import camera_compare


class CameraCompareTests(unittest.TestCase):
    def test_world_to_camera_center_and_quaternion(self):
        rotation = camera_compare.quaternion_rotation([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        world_center = np.array([2., -3., 4.])
        translation = -rotation @ world_center
        np.testing.assert_allclose(camera_compare.camera_center(rotation, translation), world_center, atol=1e-12)

    def test_named_camera_sim3_with_scale_rotation_translation(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                           [0., 0., 3.], [2., 3., 4.]])
        rotation = camera_compare.quaternion_rotation([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        mapped = 2.5 * points @ rotation.T + [4., -2., 1.]
        names = [f"{i:05}.jpg" for i in range(len(points))]
        source = dict(zip(names, points))
        target = dict(zip(names, mapped))
        matrix, report = camera_compare.fit_centers(source, target)
        np.testing.assert_allclose(matrix[:3, :3], 2.5 * rotation, atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 3], [4., -2., 1.], atol=1e-12)
        self.assertLess(report["fit_rms"], 1e-12)
        self.assertLess(report["leave_one_out_rms"], 1e-12)
        self.assertAlmostEqual(report["fitted_scale"], 2.5)

    def test_name_mismatch_and_nonrigid_camera_rejected(self):
        source = {"a.jpg": [0., 0., 0.], "b.jpg": [1., 0., 0.], "c.jpg": [0., 1., 0.]}
        target = {"a.jpg": [0., 0., 0.], "b.jpg": [1., 0., 0.], "d.jpg": [0., 1., 0.]}
        with self.assertRaises(ValueError):
            camera_compare.fit_centers(source, target)
        with self.assertRaises(ValueError):
            camera_compare.camera_center(np.diag([2., 1., 1.]), [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
