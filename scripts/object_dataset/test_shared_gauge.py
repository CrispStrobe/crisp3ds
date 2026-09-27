"""Analytic principal-axis and transform sensitivity checks."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from scripts.object_dataset import shared_gauge


class SharedGaugeTests(unittest.TestCase):
    def test_sparse_hashes_bind_all_three_files_and_reject_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in shared_gauge.SPARSE_FILES:
                (directory / name).write_bytes(name.encode())
            hashes = shared_gauge.sparse_hashes(directory)
            self.assertEqual(set(hashes), set(shared_gauge.SPARSE_FILES))
            (directory / "images.bin").write_bytes(b"changed")
            self.assertNotEqual(shared_gauge.sparse_hashes(directory), hashes)
            (directory / "images.bin").unlink()
            try:
                (directory / "images.bin").symlink_to(directory / "cameras.bin")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "linked"):
                shared_gauge.sparse_hashes(directory)

    def test_principal_axis_rotation_and_flip_equivalence(self):
        points = np.array([[-3., 0., 0.], [3., 0., 0.], [0., -1., 0.],
                           [0., 1., 0.], [0., 0., -.5], [0., 0., .5]])
        values, axes = shared_gauge.pca_points(points)
        theta = np.deg2rad(45)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotated_values, rotated_axes = shared_gauge.pca_points(points @ rotation.T)
        np.testing.assert_allclose(values, rotated_values)
        self.assertAlmostEqual(shared_gauge.axis_angle_degrees(axes[:, 0], rotated_axes[:, 0]), 45)
        self.assertAlmostEqual(shared_gauge.axis_angle_degrees(axes[:, 0], -axes[:, 0]), 0)

    def test_relative_transform_rotation_ignores_uniform_scale(self):
        parent = np.eye(4)
        theta = np.deg2rad(45)
        candidate = np.eye(4)
        candidate[:3, :3] = 2 * np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        self.assertAlmostEqual(shared_gauge.relative_rotation_degrees(parent, candidate), 45)


if __name__ == "__main__":
    unittest.main()
