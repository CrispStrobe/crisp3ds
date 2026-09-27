"""Synthetic named-pose and proper-similarity checks for Sceaux diagnostics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.upstream_control import colmap_camera_compare as diagnostic


def pose(center, rotation):
    return (np.asarray(center, dtype=float), np.asarray(rotation, dtype=float))


class ColmapCameraCompareTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                                 [0., 0., 3.], [2., 3., 4.], [-1., 1., 2.]])
        self.names = [f"{index:05}.jpg" for index in range(len(self.centers))]

    def test_known_sim3_and_orientation_convention(self):
        angle = np.deg2rad(35)
        c, s = np.cos(angle), np.sin(angle)
        world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        camera = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        source = {name: pose(center, camera) for name, center in zip(self.names, self.centers)}
        target = {name: pose(2.4 * world @ center + [3., -2., 5.], camera @ world.T)
                  for name, center in zip(self.names, self.centers)}
        report = diagnostic.compare(source, target)
        self.assertEqual(report["paired_camera_count"], 6)
        self.assertTrue(report["all_reference_cameras_paired"])
        self.assertLess(report["center_rms_over_reference_radius"], 1e-12)
        self.assertLess(report["orientation_p95_degrees"], 1e-5)
        self.assertAlmostEqual(report["similarity_source_to_reference"]["scale"], 2.4)
        np.testing.assert_allclose(report["similarity_source_to_reference"]["rotation"], world, atol=1e-12)

    def test_relative_orientation_error_is_detected(self):
        source = {name: pose(center, np.eye(3)) for name, center in zip(self.names, self.centers)}
        target = {name: pose(center, np.eye(3)) for name, center in zip(self.names, self.centers)}
        target[self.names[0]] = pose(self.centers[0], np.diag([1., -1., -1.]))
        report = diagnostic.compare(source, target)
        self.assertLess(report["center_rms"], 1e-12)
        self.assertGreater(report["orientation_p95_degrees"], 1)
        self.assertEqual(report["per_name"][self.names[0]]["orientation_error_degrees"], 180)

    def test_reflection_and_collinear_centers_rejected(self):
        with self.assertRaisesRegex(ValueError, "reflection"):
            diagnostic.proper_sim3(self.centers, self.centers * [-1, 1, 1])
        line = np.array([[i, 0., 0.] for i in range(6)])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            diagnostic.proper_sim3(line, line)

    def test_missing_names_and_basename_collisions(self):
        source = {name: pose(center, np.eye(3)) for name, center in zip(self.names, self.centers)}
        target = dict(source)
        del source[self.names[0]]
        report = diagnostic.compare(source, target)
        self.assertEqual(report["paired_camera_count"], 5)
        self.assertEqual(report["source_camera_count"], 5)
        self.assertEqual(report["reference_camera_count"], 6)
        self.assertEqual(report["missing_reference_names"], [self.names[0]])
        self.assertFalse(report["all_reference_cameras_paired"])
        self.assertIn("RMS distance", report["reference_camera_radius_definition"])
        poses = {}
        diagnostic.add_pose(poses, "images/a.jpg", np.eye(3), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "collision"):
            diagnostic.add_pose(poses, "other/a.jpg", np.eye(3), [0, 0, 0])

    def test_reference_text_quaternion_and_world_to_camera_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text("# COLMAP\n1 1 0 0 0 -1 -2 -3 1 images/a.jpg\n\n"
                            "2 1 0 0 0 0 0 0 1 b.jpg\n\n")
            poses = diagnostic.read_reference(path)
            np.testing.assert_allclose(poses["a.jpg"][0], [1, 2, 3])
            self.assertEqual(set(poses), {"a.jpg", "b.jpg"})
            path.write_text(path.read_text() + "3 1 0 0 0 0 0 0 1 other/a.jpg\n\n")
            with self.assertRaisesRegex(ValueError, "collision"):
                diagnostic.read_reference(path)


if __name__ == "__main__":
    unittest.main()
