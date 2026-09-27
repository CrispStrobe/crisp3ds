import types
import unittest

import numpy as np

from scripts.classical_backend import dense_masks


def camera(model, params, width=11, height=9):
    return types.SimpleNamespace(model=types.SimpleNamespace(name=model), params=params,
                                 width=width, height=height)


class NativeMaskNamingTests(unittest.TestCase):
    def test_openmvs_mask_filename_uses_image_stem_not_jpeg_extension(self):
        self.assertEqual(dense_masks.native_mask_name("NP3_000.jpg"), "NP3_000.mask.png")
        with self.assertRaisesRegex(ValueError, "basename"):
            dense_masks.native_mask_name("sub/NP3_000.jpg")


@unittest.skipIf(dense_masks.cv2 is None, "OpenCV not installed in generic test environment")
class DenseMaskTests(unittest.TestCase):
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
