import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.object_dataset import bunny_roi, evaluate


class BunnyROITests(unittest.TestCase):
    def test_all_vertices_strictly_above_cut_and_reindexes(self):
        v = np.array([[0, -24, 0], [1, -23, 0], [0, -23, 1],
                      [1, -22, 1], [0, -22, 2]], dtype=float)
        f = np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int32)
        rv, rf, kept = bunny_roi.select_roi(v, f)
        self.assertEqual(kept, 1)
        np.testing.assert_array_equal(rv, v[[1, 3, 4]])
        np.testing.assert_array_equal(rf, [[0, 1, 2]])

    def test_invalid_indices_rejected_before_lookup(self):
        with self.assertRaises(ValueError):
            bunny_roi.select_roi(np.zeros((3, 3)), np.array([[0, 1, 3]]))

    def test_binary_ply_round_trip(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "roi.ply"
            bunny_roi.write_ply(p, v, f)
            _, rv, rf = evaluate.inspect_ply(p, geometry=True)
            np.testing.assert_array_equal(rv, v)
            np.testing.assert_array_equal(rf, f)


if __name__ == "__main__":
    unittest.main()
