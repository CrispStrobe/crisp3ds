"""Small pure contracts for the read-only YCB seed diagnostic."""

import unittest

import numpy as np

from scripts.object_motion import initialization_diagnostic as diag


def pair(config, values):
    return {"config": config, "matches": np.asarray(values, dtype=np.uint32).reshape(-1, 2)}


class InitializationDiagnosticTests(unittest.TestCase):
    def test_pair_id_decoding_and_bad_order(self):
        self.assertEqual(diag.decode_pair(2 * diag.MAX_IMAGE_ID + 5), (2, 5))
        with self.assertRaises(ValueError):
            diag.decode_pair(5 * diag.MAX_IMAGE_ID + 2)

    def test_two_view_blob_shape_and_empty(self):
        data = np.array([[1, 2], [3, 4]], dtype="<u4")
        np.testing.assert_array_equal(diag.decode_matches(data.tobytes(), 2, 2), data)
        self.assertEqual(diag.decode_matches(None, 0, 2).shape, (0, 2))
        with self.assertRaises(ValueError):
            diag.decode_matches(data.tobytes()[:-1], 2, 2)

    def test_exact_triplet_support_requires_same_third_feature(self):
        pairs = {(1, 2): pair(3, [[0, 0], [1, 1], [2, 2]]),
                 (1, 3): pair(3, [[0, 4], [1, 5], [2, 6]]),
                 (2, 3): pair(3, [[0, 4], [1, 9], [2, 6]])}
        self.assertEqual(diag.triple_support((1, 2), pairs), {3: 2})

    def test_candidate_policy_does_not_gate_by_filename_angle(self):
        names = {1: "NP3_000.jpg", 2: "NP3_006.jpg", 3: "NP3_012.jpg"}
        pairs = {(1, 2): pair(3, [[i, i] for i in range(80)]),
                 (1, 3): pair(3, [[i, i] for i in range(80)]),
                 (2, 3): pair(3, [[i, i] for i in range(80)])}
        rows = diag.candidate_rows(names, pairs)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["best_exact_triplet_support"], 80)
        self.assertIn(6, [row["acquisition_angle_gap_degrees"] for row in rows])

    def test_planar_and_low_support_pairs_not_candidates(self):
        names = {1: "NP3_000.jpg", 2: "NP3_006.jpg", 3: "NP3_012.jpg"}
        pairs = {(1, 2): pair(6, [[i, i] for i in range(100)]),
                 (1, 3): pair(3, [[i, i] for i in range(79)]),
                 (2, 3): pair(3, [[i, i] for i in range(79)])}
        self.assertEqual(diag.candidate_rows(names, pairs), [])


if __name__ == "__main__":
    unittest.main()
