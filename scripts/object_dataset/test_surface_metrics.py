"""Analytic checks for point-to-triangle distances and directional scores."""

import unittest

import numpy as np

try:
    from . import surface_metrics as metrics
except ImportError:
    import surface_metrics as metrics


class TriangleDistanceTests(unittest.TestCase):
    def setUp(self):
        self.triangle = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
        self.tree = metrics.TriangleBVH(self.triangle)

    def test_face_edge_vertex_regions(self):
        points = np.array([[.5, .5, 3], [1, -2, 0], [-1, -1, 0], [2, 2, 0]])
        expected = np.array([3, 2, np.sqrt(2), np.sqrt(2)])
        np.testing.assert_allclose(self.tree.distances(points), expected, atol=1e-12)

    def test_bvh_matches_brute_force_and_discards_zero_area(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                             [2., 2., 0.], [5., 5., 5.]])
        triangles, dropped = metrics.positive_triangles(vertices, [[0, 1, 2], [1, 3, 2], [4, 4, 4]])
        self.assertEqual(dropped, 1)
        tree = metrics.TriangleBVH(triangles)
        points = np.random.default_rng(1).uniform(-2, 4, (100, 3))
        brute = np.sqrt([metrics.point_triangle_squared(point, triangles).min() for point in points])
        np.testing.assert_allclose(tree.distances(points), brute, atol=1e-12)

    def test_randomized_multileaf_bvh_and_skinny_triangle(self):
        rng = np.random.default_rng(27)
        triangles = rng.uniform(-10, 10, (129, 3, 3))
        triangles[0] = [[0, 0, 0], [1, 0, 0], [1, 1e-10, 0]]
        points = np.vstack((rng.uniform(-10, 10, (60, 3)), [[.5, 5e-11, 2]]))
        tree = metrics.TriangleBVH(triangles)
        brute = np.sqrt([metrics.point_triangle_squared(point, triangles).min() for point in points])
        np.testing.assert_allclose(tree.distances(points), brute, atol=1e-10)
        self.assertAlmostEqual(float(metrics.point_triangle_squared(points[-1], triangles[:1])[0]), 4)

    def test_invalid_geometry_and_transform_rejected(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        with self.assertRaises(ValueError):
            metrics.positive_triangles(vertices, [[0, 1, 9]])
        with self.assertRaises(ValueError):
            metrics.positive_triangles(vertices, [[0.0, 1.5, 2.0]])
        with self.assertRaises(ValueError):
            metrics.positive_triangles(np.array([[np.nan, 0, 0]]), [[0, 0, 0]])
        huge = np.array([[0., 0., 0.], [1e300, 0., 0.], [0., 1e300, 0.]])
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(ValueError, "overflowed"):
                metrics.positive_triangles(huge, [[0, 1, 2]])
        mesh = (vertices, np.array([[0, 1, 2]]))
        reflected = np.diag([-1., 1., 1., 1.])
        with self.assertRaises(ValueError):
            metrics.compare(mesh, mesh, reflected, thresholds=[.1], count=8)

    def test_parallel_planes_have_exact_offset_and_f_score(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        shifted = vertices + np.array([0., 0., 2.])
        ref = (vertices, np.array([[0, 1, 2]]))
        out = (shifted, np.array([[0, 1, 2]]))
        result = metrics.compare(ref, out, np.eye(4), thresholds=[1, 2, 3], count=64)
        self.assertAlmostEqual(result["output_to_reference_accuracy"]["mean"], 2)
        self.assertAlmostEqual(result["reference_to_output_completeness"]["mean"], 2)
        self.assertEqual([row["f_score"] for row in result["threshold_scores"]], [0, 1, 1])

    def test_identical_surface_self_control_zero(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        mesh = (vertices, np.array([[0, 1, 2]]))
        result = metrics.compare(mesh, mesh, np.eye(4), thresholds=[1e-8], count=64)
        self.assertLess(result["output_to_reference_accuracy"]["rms"], 1e-12)
        self.assertEqual(result["threshold_scores"][0]["f_score"], 1)


if __name__ == "__main__":
    unittest.main()
