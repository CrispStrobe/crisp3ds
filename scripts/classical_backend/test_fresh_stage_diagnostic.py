"""Small deterministic checks for the evaluation-only stage diagnostic."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.classical_backend import fresh_stage_diagnostic as target
from scripts.object_dataset import surface_metrics


class FreshStageDiagnosticTests(unittest.TestCase):
    def test_cloud_proximity_is_one_way_exact_triangle_distance(self):
        triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        tree = surface_metrics.TriangleBVH(triangle)
        cloud = np.tile([[.25, .25, .1]], (16, 1))
        with patch.object(target, "COUNT", 16):
            result = target.proximity(cloud, np.eye(4), tree,
                                      (np.array([0., 0., 0.]), np.array([1., 1., 0.])),
                                      [.05, .15])
        self.assertAlmostEqual(result["distance"]["mean"], .1)
        self.assertEqual(result["within_threshold_fraction"], [0., 1.])
        self.assertEqual(result["within_reference_bbox_fraction"], 0.)
        self.assertNotIn("f_score", result)

    def test_bounds_report_all_three_axes_without_clipping(self):
        self.assertEqual(target.bounds([[1, -2, 4], [3, 5, -1]])["extent"],
                         [2., 7., 5.])


if __name__ == "__main__":
    unittest.main()
