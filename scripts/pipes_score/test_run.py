"""Small independent geometry and input-contract checks for the pipes scorer."""

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.pipes_score import run as score


class PipesScoreTest(unittest.TestCase):
    def test_opencv_serial_setting(self):
        self.assertEqual(score.configure_threads(), 1)

    def test_ply_extra_element_rigid_alignment_and_nearest_sqrt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            scan = base / "scan1.ply"
            header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "element camera 1\nproperty int view\nend_header\n").encode()
            scan.write_bytes(header + struct.pack("<9fi", 0, 0, 0, 1, 0, 0,
                                                  0, 2, 0, 17))
            mlp = base / "scan_alignment.mlp"
            mlp.write_text('<MeshLabProject><MeshGroup><MLMesh filename="scan1.ply">'
                           '<MLMatrix44>0 -1 0 2 1 0 0 3 0 0 1 4 0 0 0 1</MLMatrix44>'
                           '</MLMesh></MeshGroup></MeshLabProject>')
            matrix = score.mlp_matrix(mlp, scan)
            self.assertEqual(score.ply_layout(scan)[1], 3)
            world = score.scan_points(scan, matrix, float("inf"))
            np.testing.assert_allclose(world, [[2, 3, 4], [2, 4, 4], [0, 3, 4]])
            queries = np.asarray([[2, 3.5, 4], [0, 3, 4.2], [2, 4, 7]], np.float32)
            got, nearest = score.exact_distances(world, queries, float("inf"))
            expected = [np.min(np.linalg.norm(world.astype(np.float64) - q, axis=1)) for q in queries]
            np.testing.assert_allclose(got, expected, atol=1e-6)
            self.assertEqual(len(nearest), 3)
            self.assertAlmostEqual(got[0], .5, places=6)
            self.assertAlmostEqual(got[1], .2, places=6)
            self.assertAlmostEqual(got[2], 3., places=6)

    def test_full_scan_matches_independent_all_pairs_tiny_random(self):
        rng = np.random.default_rng(314159)
        world = rng.normal(size=(251, 3)).astype(np.float32)
        queries = rng.normal(size=(23, 3)).astype(np.float32)
        got, nearest = score.exact_distances(world, queries, float("inf"))
        all_pairs = [[math.dist(q.tolist(), p.tolist()) for p in world] for q in queries]
        expected = [min(row) for row in all_pairs]
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(nearest, [row.index(min(row)) for row in all_pairs])

    def test_adversarial_nearest_crosses_chunk_boundary_and_never_calls_flann(self):
        world = np.full((250_001, 3), 100., dtype=np.float32)
        world[0] = [.001, 0, 0]  # An approximate search could stop here.
        world[-1] = [0, 0, 0]   # Exact nearest in the next chunk.
        query = np.zeros((1, 3), dtype=np.float32)
        with mock.patch.object(score.cv2, "flann_Index", side_effect=AssertionError("FLANN is approximate")):
            distances, nearest = score.exact_distances(world, query, float("inf"))
        self.assertEqual(nearest.tolist(), [250_000])
        self.assertEqual(distances.tolist(), [0.])

    def test_reject_empty_and_nonfinite_sparse_population(self):
        with self.assertRaisesRegex(ValueError, "empty sparse distance"):
            score.summarise(np.array([], dtype=np.float64))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = base / "report.json"
            points = base / "points.json"
            report.write_text(json.dumps({"schema": "eth3d_pipes_fixed_camera_sparse_v1",
                                          "reference_geometry_used": False,
                                          "accepted_tracks": 0}))
            points.write_text(json.dumps({"schema": "fixed_camera_sparse_points_v1", "points": []}))
            with self.assertRaisesRegex(ValueError, "empty"):
                score.sparse_points(points, report)
            report.write_text(json.dumps({"schema": "eth3d_pipes_fixed_camera_sparse_v1",
                                          "reference_geometry_used": False,
                                          "accepted_tracks": 1}))
            points.write_text(json.dumps({"schema": "fixed_camera_sparse_points_v1",
                                          "points": [{"id": 1, "xyz": [0, 0, float("nan")]}]}))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                score.sparse_points(points, report)


if __name__ == "__main__":
    unittest.main()
