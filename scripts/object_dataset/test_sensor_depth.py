import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from scripts.object_dataset import sensor_depth as sd


class RayDepthTests(unittest.TestCase):
    def setUp(self):
        self.square_vertices = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
        self.square_faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def test_plane_first_hit_and_miss(self):
        rays = np.array([[0, 0, 1], [.5, .3, 1], [2, 0, 1]], float)
        got, degenerate = sd.first_hit_depths(self.square_vertices, self.square_faces, rays)
        np.testing.assert_allclose(got[:2], [1, 1], atol=1e-14)
        self.assertTrue(math.isnan(got[2]))
        self.assertEqual(degenerate, 0)

    def test_occlusion_double_sided_and_zero_area(self):
        vertices = np.vstack((self.square_vertices, self.square_vertices * [.5, .5, .5],
                              [[0, 0, .8], [0, 0, .8], [0, 0, .8]]))
        faces = np.vstack((self.square_faces, self.square_faces[:, ::-1] + 4, [[8, 9, 10]]))
        got, degenerate = sd.first_hit_depths(vertices, faces, [[0, 0, 1]])
        self.assertAlmostEqual(got[0], .5)
        self.assertEqual(degenerate, 1)

    def test_parallel_box_axis_and_far_boundary(self):
        vertices = self.square_vertices * [1, 1, 1.5]
        got, _ = sd.first_hit_depths(vertices, self.square_faces, [[0, 0, 1], [3, 0, 1]])
        self.assertAlmostEqual(got[0], 1.5)
        self.assertTrue(math.isnan(got[1]))

    def test_known_proper_transform_and_reflection_rejection(self):
        transform = np.eye(4)
        transform[:3, :3] = [[0, -2, 0], [2, 0, 0], [0, 0, 2]]
        transform[:3, 3] = [0, 0, -.9]
        transformed = sd.transform_vertices(self.square_vertices, transform)
        got, _ = sd.first_hit_depths(transformed, self.square_faces, [[0, 0, 1]])
        self.assertAlmostEqual(got[0], 1.1)
        transform[0, 0] = -1
        with self.assertRaises(ValueError):
            sd.proper_similarity(transform)

    def test_depth_ray_integer_address(self):
        k = np.array([[100, 0, 20], [0, 100, 10], [0, 0, 1]], float)
        np.testing.assert_array_equal(sd.depth_directions([20, 30], [10, 20], k),
                                      [[0, 0, 1], [.1, .1, 1]])


class SupportTests(unittest.TestCase):
    def test_calibration_zero_is_missing_and_bias_unknown_rejected(self):
        got = sd.calibrated_depth_metres(np.array([0, 10000], np.uint16), 1.0016965552242483, 0)
        self.assertTrue(math.isnan(got[0]))
        self.assertAlmostEqual(got[1], 1.0016965552242483)
        with self.assertRaises(ValueError):
            sd.calibrated_depth_metres(np.array([1], np.uint16), 1, .001)
        frame = np.zeros((480, 640), np.uint16)
        frame[2, 3] = 10000
        selected, metadata = sd.selected_depth_indices(sd.calibrated_depth_metres(frame, 1, 0),
                                                       np.ones((480, 640), bool), 0)
        np.testing.assert_array_equal(selected, [2 * 640 + 3])
        self.assertEqual(metadata["valid_interval_depth_pixels"], 1)

    def test_projection_identity_and_raster_no_clipping(self):
        k = np.array([[100, 0, 2], [0, 100, 1], [0, 0, 1]], float)
        directions = sd.depth_directions([2, 3, 500], [1, 1, 1], k)
        xy, valid = sd.project_depth_to_rgb(directions, np.ones(3), np.eye(4), k, np.zeros(5))
        np.testing.assert_allclose(xy, [[2, 1], [3, 1], [500, 1]])
        mask = np.zeros((3, 5), bool)
        mask[1, 2] = mask[1, 3] = True
        np.testing.assert_array_equal(sd.raster_support(xy, valid, mask), [True, True, False])
        bad_h = np.eye(4)
        bad_h[0, 0] = -1
        with self.assertRaises(ValueError):
            sd.project_depth_to_rgb(directions, np.ones(3), bad_h, k, np.zeros(5))

    def test_erosion_and_deterministic_shared_sample(self):
        mask = np.ones((40, 40), bool)
        eroded = sd.erode_square(mask, 16)
        self.assertTrue(eroded[20, 20])
        self.assertFalse(eroded[0, 0])
        self.assertEqual(int(eroded.sum()), 64)
        depth = np.ones((480, 640))
        support = np.ones((480, 640), bool)
        a, meta_a = sd.selected_depth_indices(depth, support, 120)
        b, meta_b = sd.selected_depth_indices(depth, support, 120)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(meta_a, meta_b)
        self.assertEqual(len(a), 2048)
        self.assertEqual(meta_a["eligible_count"], 307200)
        self.assertTrue(np.all(np.diff(a) > 0))
        depth[0, 0] = np.nan
        _, meta_missing = sd.selected_depth_indices(depth, support, 120)
        self.assertEqual(meta_missing["eligible_count"], 307199)

    def test_residuals_exclude_missing_prediction(self):
        got = sd.residual_summary([1, 1.1, 1.2], [1.01, np.nan, 1.19])
        self.assertEqual(got["supported_rays"], 3)
        self.assertEqual(got["hits"], 2)
        self.assertEqual(got["missing_predictions"], 1)
        self.assertAlmostEqual(got["signed_m"]["mean"], 0)
        self.assertAlmostEqual(got["absolute_m"]["mean"], .01)
        self.assertAlmostEqual(got["within_m"]["0.02"]["among_supported"], 2 / 3)
        self.assertEqual(sd.nullable_depths(np.array([1.01, np.nan, 1.19])),
                         [1.01, None, 1.19])

    def test_dense_camera_gauge_mocked_without_pycolmap_dependency(self):
        class FakeImage:
            def __init__(self, index, delta):
                self.name = f"view{index}.jpg"
                matrix = np.column_stack((np.eye(3), np.array([delta, 0, 0])))
                self.cam_from_world = types.SimpleNamespace(matrix=lambda: matrix)

        class FakeReconstruction:
            def __init__(self, path):
                self.images = {index: FakeImage(index, 0 if path.endswith("source") else self.delta)
                               for index in range(60)}

            delta = 0

        with tempfile.TemporaryDirectory(dir=sd.ROOT / ".local-tools/tmp") as folder:
            dense = Path(folder) / "dense"
            dense.mkdir()
            (dense / "images.bin").write_bytes(b"mock images")
            with mock.patch.dict(sys.modules, {"pycolmap": types.SimpleNamespace(Reconstruction=FakeReconstruction)}):
                good = sd.verify_dense_camera_gauge(dense, Path(folder) / "source")
                self.assertEqual(good["named_registered_images"], 60)
                FakeReconstruction.delta = .01
                with self.assertRaisesRegex(ValueError, "poses differ"):
                    sd.verify_dense_camera_gauge(dense, Path(folder) / "source")


if __name__ == "__main__":
    unittest.main()
