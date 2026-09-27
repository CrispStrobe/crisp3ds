import types
import unittest

import numpy as np

from scripts.classical_backend import dense_masks


def camera(model, params, width=11, height=9):
    return types.SimpleNamespace(model=types.SimpleNamespace(name=model), params=params,
                                 width=width, height=height)


class NativeMaskNamingTests(unittest.TestCase):
    def test_full_opencv_rejects_truncated_and_singular_models(self):
        xx = np.asarray([0.1, 0.5])
        yy = np.asarray([-0.2, 0.4])
        with self.assertRaisesRegex(ValueError, "12 finite"):
            dense_masks.full_opencv_project(xx, yy, [1] * 8)
        singular = [1000, 1000, 640, 512, 0, 0, 0, 0, 0, -1, 0, 0]
        with self.assertRaisesRegex(ValueError, "singular"):
            dense_masks.full_opencv_project(np.asarray([1.0]), np.asarray([0.0]), singular)

    def test_openmvs_mask_filename_uses_image_stem_not_jpeg_extension(self):
        self.assertEqual(dense_masks.native_mask_name("NP3_000.jpg"), "NP3_000.mask.png")
        with self.assertRaisesRegex(ValueError, "basename"):
            dense_masks.native_mask_name("sub/NP3_000.jpg")


@unittest.skipIf(dense_masks.cv2 is None, "OpenCV not installed in generic test environment")
class DenseMaskTests(unittest.TestCase):
    def test_full_opencv_five_coefficients_match_independent_cv2_projection(self):
        # Berkeley NP3 RGB 5-coefficient calibration, including nonzero k3.
        params = [1081.1500729507718, 1081.1438315939365,
                  621.3445331259582, 480.58324224028723,
                  -0.035479533764683456, 0.14390968646477606,
                  0.006104733063423312, 0.00008415227830856679,
                  -0.17716948411083983, 0, 0, 0]
        nx = np.asarray([-0.57, -0.32, 0, 0.41, 0.58], dtype=np.float64)
        ny = np.asarray([-0.42, 0.38, 0, -0.33, 0.44], dtype=np.float64)
        got_x, got_y = dense_masks.full_opencv_project(nx, ny, params)
        points = np.stack([nx, ny, np.ones_like(nx)], axis=1).reshape(-1, 1, 3)
        intrinsic = np.asarray([[params[0], 0, params[2]], [0, params[1], params[3]],
                                [0, 0, 1]], dtype=np.float64)
        expected, _ = dense_masks.cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                                     intrinsic, np.asarray(params[4:]))
        np.testing.assert_allclose(np.stack([got_x, got_y], axis=1), expected[:, 0, :], atol=1e-7)
        no_k3 = params.copy()
        no_k3[8] = 0
        dropped_x, _ = dense_masks.full_opencv_project(nx, ny, no_k3)
        self.assertGreater(float(np.max(np.abs(got_x - dropped_x))), 1.0)

    def test_full_opencv_mask_warp_matches_cv2_with_half_pixel_and_rescale(self):
        params = [28.0, 27.0, 10.2, 8.4, -0.05, 0.12, 0.006, -0.002,
                  -0.18, 0.01, -0.004, 0.001]
        source = camera("FULL_OPENCV", params, 21, 17)
        target = camera("PINHOLE", [35.0, 34.0, 12.1, 10.2], 26, 20)
        mask = np.zeros((17, 21), dtype=np.uint8)
        mask[3:15, 4:18] = 255
        mask[6:9, 9:13] = 0
        got = dense_masks.remap_binary_mask(mask, source, target)
        scale_x, scale_y = source.width / target.width, source.height / target.height
        xx, yy = np.meshgrid(np.arange(source.width) + 0.5, np.arange(source.height) + 0.5)
        nx = (xx - target.params[2] * scale_x) / (target.params[0] * scale_x)
        ny = (yy - target.params[3] * scale_y) / (target.params[1] * scale_y)
        points = np.stack([nx, ny, np.ones_like(nx)], axis=-1).reshape(-1, 1, 3)
        intrinsic = np.asarray([[params[0], 0, params[2]], [0, params[1], params[3]],
                                [0, 0, 1]], dtype=np.float64)
        projected, _ = dense_masks.cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                                      intrinsic, np.asarray(params[4:]))
        maps = projected.reshape(source.height, source.width, 2) - 0.5
        expected = dense_masks.cv2.remap(mask, maps[:, :, 0].astype(np.float32),
                                         maps[:, :, 1].astype(np.float32), dense_masks.cv2.INTER_NEAREST,
                                         borderMode=dense_masks.cv2.BORDER_CONSTANT, borderValue=0)
        expected = dense_masks.cv2.resize(expected, (target.width, target.height),
                                          interpolation=dense_masks.cv2.INTER_NEAREST)
        np.testing.assert_array_equal(got, expected)

    def test_identity_warp_preserves_binary_pixels(self):
        original = np.zeros((9, 11), dtype=np.uint8)
        original[2:7, 3:9] = 255
        source = camera("SIMPLE_RADIAL", [10, 5, 4, 0])
        target = camera("PINHOLE", [10, 10, 5, 4])
        np.testing.assert_array_equal(dense_masks.remap_binary_mask(original, source, target), original)

    def test_radial_warp_uses_original_distorted_coordinates(self):
        original = np.zeros((9, 11), dtype=np.uint8)
        original[4, 7] = 255
        source = camera("SIMPLE_RADIAL", [10, 5, 4, 0.5])
        target = camera("PINHOLE", [10, 10, 5, 4])
        warped = dense_masks.remap_binary_mask(original, source, target)
        self.assertEqual(int(warped[4, 7]), 255)
        self.assertEqual(int(warped[4, 8]), 0)
        self.assertTrue(np.all((warped == 0) | (warped == 255)))

    def test_target_resize_uses_colmap_source_resolution_then_nearest_labels(self):
        original = np.zeros((9, 11), dtype=np.uint8)
        original[3:6, 4:7] = 255
        source = camera("SIMPLE_RADIAL", [10, 5.5, 4.5, 0])
        target = camera("PINHOLE", [20, 20, 11, 9], 22, 18)
        warped = dense_masks.remap_binary_mask(original, source, target)
        self.assertEqual(warped.shape, (18, 22))
        np.testing.assert_array_equal(warped, np.repeat(np.repeat(original, 2, axis=0), 2, axis=1))

    def test_half_pixel_center_matters_when_focal_lengths_differ(self):
        original = np.zeros((9, 11), dtype=np.uint8)
        original[4, 4] = 255
        source = camera("SIMPLE_RADIAL", [10, 5.3, 4.5, 0])
        target = camera("PINHOLE", [5, 10, 5.1, 4.5])
        # At target pixel (4,4), COLMAP projects center x=4.5 to source
        # sample x=3.6, hence nearest label index 4. Using x=4 instead
        # incorrectly projects to source index 3.1.
        warped = dense_masks.remap_binary_mask(original, source, target)
        self.assertEqual(int(warped[4, 4]), 255)

    def test_rejects_wrong_dimensions_nonbinary_and_unsupported_model(self):
        original = np.zeros((9, 11), dtype=np.uint8)
        source = camera("SIMPLE_RADIAL", [10, 5, 4, 0])
        target = camera("PINHOLE", [10, 10, 5, 4])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            dense_masks.remap_binary_mask(original[:, :-1], source, target)
        original[4, 5] = 127
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            dense_masks.remap_binary_mask(original, source, target)
        original[4, 5] = 0
        with self.assertRaisesRegex(ValueError, "unsupported"):
            dense_masks.remap_binary_mask(original, camera("OPENCV", [10, 10, 5, 4]), target)


if __name__ == "__main__":
    unittest.main()
