"""Analytic, dataset-free registration overlay contracts."""

import unittest

import numpy as np

from scripts.object_dataset import overlay_preview


class OverlayPreviewTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                                  [0., 1., 0.], [0., 0., 1.]])
        self.faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    def test_three_views_share_bounds_and_show_both_roles(self):
        transform = np.eye(4)
        transform[:3, 3] = [2., -3., 1.]
        candidate = self.vertices - transform[:3, 3]
        image, bounds = overlay_preview.make_overlay(
            (self.vertices, self.faces), (candidate, self.faces), transform, count=256)
        self.assertEqual(image.size, (560, 1680))
        self.assertEqual([item["view"] for item in bounds], ["XY", "XZ", "YZ"])
        self.assertTrue(all(item["radius"] > 0 for item in bounds))
        pixels = np.asarray(image)
        self.assertGreater(np.count_nonzero((pixels[:, :, 1] > pixels[:, :, 0]) &
                                            (pixels[:, :, 2] > pixels[:, :, 0])), 0)
        self.assertGreater(np.count_nonzero((pixels[:, :, 0] > pixels[:, :, 1]) &
                                            (pixels[:, :, 1] > pixels[:, :, 2])), 0)

    def test_bounded_samples_and_degenerate_projection(self):
        with self.assertRaisesRegex(ValueError, "sample count"):
            overlay_preview.make_overlay((self.vertices, self.faces),
                                         (self.vertices, self.faces), np.eye(4), count=10_001)
        zero = np.zeros_like(self.vertices)
        with self.assertRaises(ValueError):
            overlay_preview.make_overlay((zero, self.faces), (zero, self.faces),
                                         np.eye(4), count=16)


if __name__ == "__main__":
    unittest.main()
