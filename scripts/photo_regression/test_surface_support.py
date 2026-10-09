import unittest
import numpy as np
from surface_support import support_counts, sample_surface


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.row = {'rotation': np.eye(3).tolist(), 'translation': [0, 0, 0]}
        self.stage = {'k': [10, 10, 1.5, 1.5], 'width': 3, 'height': 3}
        self.depth = np.full((3, 3), 2.)

    def test_free_space_is_not_a_surface_observation(self):
        points = [[0, 0, 1], [0, 0, 1.95], [0, 0, 2.05], [0, 0, 2.3], [0, 0, -1], [2, 0, 2]]
        c = support_counts(points, [self.row], [self.stage], [self.depth], .1, .4)
        np.testing.assert_array_equal(c['near'], [0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(c['free'], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(c['behind'], [0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(c['positive'], [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(c['negative'], [0, 0, 1, 0, 0, 0])

    def test_transformed_camera_and_cropped_intrinsics(self):
        row = {'rotation': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], 'translation': [0, 0, 3]}
        # World x=1 is camera Z=2, y=.2 projects to cropped pixel (1,2).
        d = np.zeros((3, 3));d[2, 1] = 2
        c = support_counts([[1, .2, 0]], [row, row], [self.stage]*2, [d, d], .1, .4)
        self.assertEqual(c['near'].tolist(), [2])

    def test_missing_depth_is_not_behind_or_free_evidence(self):
        c = support_counts([[0, 0, 2]], [self.row], [self.stage], [np.zeros((3, 3))], .1, .4)
        self.assertTrue(all(a[0] == 0 for a in c.values()))

    def test_area_sampling_preserves_large_triangle_share(self):
        triangles = np.array([[[0, 0, 2], [1, 0, 2], [0, 1, 2]], [[10, 0, 2], [12, 0, 2], [10, 2, 2]]])
        points = sample_surface(triangles, 20000, 236)
        self.assertAlmostEqual(float(np.mean(points[:, 0] > 5)), .8, delta=.01)
        np.testing.assert_array_equal(points, sample_surface(triangles, 20000, 236))

    def test_malformed_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            support_counts([[0, 0, 2]], [self.row], [], [], .1, .4)
        with self.assertRaises(ValueError):
            support_counts([[0, 0, 2]], [self.row], [self.stage], [np.full((3, 3), np.nan)], .1, .4)
        with self.assertRaises(ValueError):
            sample_surface(np.zeros((1, 3, 3)), 10, 1)


if __name__ == '__main__':
    unittest.main()
