import unittest

import numpy as np

from scripts.object_motion.mustard_photo_refine import refine


class TestMustardPhotoRefine(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((1024, 1280, 3), [175, 195, 220], np.uint8)
        self.prior = np.zeros((1024, 1280), np.uint8)
        self.prior[370:590, 550:640] = 255
        self.rgb[370:590, 550:640] = [135, 65, 24]
        self.rgb[440:480, 570:620] = [50, 30, 100]  # dark label

    def test_label_core_and_no_new_background(self):
        result, stats = refine(self.rgb, self.prior)
        self.assertTrue(np.all(result[445:475, 575:615] == 255))
        self.assertTrue(np.array_equal(result[:435], self.prior[:435]))
        self.assertFalse(np.any((result > 0) & (self.prior == 0)))
        self.assertEqual(stats['prior_pixels'] - stats['trimmed_pixels'], stats['retained_pixels'])

    def test_reject_nonbinary_prior(self):
        self.prior[400, 600] = 127
        with self.assertRaises(ValueError):
            refine(self.rgb, self.prior)

    def test_reject_mask_outside_training_roi(self):
        self.prior[100, 100] = 255
        with self.assertRaises(ValueError):
            refine(self.rgb, self.prior)


if __name__ == '__main__':
    unittest.main()
