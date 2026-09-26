import time
import unittest

import numpy as np

from scripts.object_dataset import align


class AlignmentTests(unittest.TestCase):
    def test_asymmetric_known_scale_rotation_translation(self):
        rng = np.random.default_rng(19)
        output = rng.normal(size=(120, 3)) * [3.0, 1.4, 0.5]
        output[:, 1] += 0.15 * output[:, 0] ** 2
        angle = 0.61
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        scale = 1.7
        translation = np.array([7.0, -2.0, 4.0])
        reference = output @ (scale * rotation).T + translation
        fitted, diagnostics = align.align_points(reference, output,
                                                  deadline=time.monotonic() + 20)
        np.testing.assert_allclose(fitted[:3, :3], scale * rotation, atol=1e-6)
        np.testing.assert_allclose(fitted[:3, 3], translation, atol=1e-6)
        self.assertLess(diagnostics["final_symmetric_rms_nearest_sampled_point"], 1e-5)
        self.assertEqual(diagnostics["initial_candidate_count"], 24)

    def test_degenerate_points_rejected(self):
        points = np.zeros((20, 3))
        with self.assertRaisesRegex(ValueError, "spread|radius"):
            align.align_points(points, points)

    def test_refinement_keeps_a_better_initial_transform(self):
        rng = np.random.default_rng(123)
        points = rng.normal(size=(40, 3)) * [2.1, 1.2, 0.7]
        fitted, value, history = align.refine(points, points, np.eye(4), 3,
                                               time.monotonic() + 10)
        self.assertLess(value, 1e-6)
        np.testing.assert_allclose(fitted, np.eye(4), atol=1e-6)
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
