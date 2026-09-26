import math
import unittest

from scripts.colmap_sparse.feature_split import RADIUS_PX, split_features


class FeatureSplitTest(unittest.TestCase):
    def test_six_column_affine_orientations_stay_together(self):
        rows = [[10, 10, 1, 0, 0, 1], [10, 10, 0, -1, 1, 0],
                [20, 10, 1, 0, 0, 1], [30, 10, 1, 0, 0, 1],
                [40, 10, 1, 0, 0, 1], [50, 10, 1, 0, 0, 1]]
        result = split_features(rows, "view.png", "fixed-seed")
        self.assertEqual(result.layout, "affine")
        self.assertEqual(result.group_for_row[0], result.group_for_row[1])
        self.assertEqual(result.heldout_for_row[0], result.heldout_for_row[1])
        self.assertEqual(len(result.groups), 5)
        self.assertEqual(len({result.group_for_row[i] for i in result.heldout_ids}), 1)

    def test_two_and_four_column_layouts(self):
        xy = split_features([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], "a", "s")
        four = split_features([[0, 0, 2, 0], [0, 0, 5, math.pi],
                               [1, 1, 2, 0], [2, 2, 2, 0],
                               [3, 3, 2, 0], [4, 4, 2, 0]], "a", "s")
        self.assertEqual(xy.layout, "xy")
        self.assertEqual(four.layout, "scale_orientation")
        self.assertEqual(four.group_for_row[0], four.group_for_row[1])

    def test_rounding_boundary_does_not_split_clones(self):
        rows = [[0.249, 0], [0.251, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
        result = split_features(rows, "a", "s")
        self.assertEqual(result.group_for_row[0], result.group_for_row[1])
        self.assertEqual(result.heldout_for_row[0], result.heldout_for_row[1])

    def test_transitive_spatial_chain_is_one_component(self):
        rows = [[0, 0], [RADIUS_PX*0.8, 0], [RADIUS_PX*1.6, 0],
                [2, 0], [4, 0], [6, 0], [8, 0]]
        result = split_features(rows, "a", "s")
        self.assertEqual(len({result.group_for_row[i] for i in (0, 1, 2)}), 1)
        self.assertGreater(rows[2][0]-rows[0][0], RADIUS_PX)

    def test_permutation_keeps_location_assignment(self):
        rows = [[x, y, 1, 0, 0, 1] for x, y in
                [(0, 0), (0.02, 0), (2, 0), (4, 0), (6, 0), (8, 0), (10, 0)]]
        first = split_features(rows, "view", "seed")
        order = [4, 6, 1, 5, 0, 2, 3]
        shuffled = split_features([rows[i] for i in order], "view", "seed")
        by_location = {tuple(row[:2]): first.heldout_for_row[i]
                       for i, row in enumerate(rows)}
        self.assertEqual(by_location,
                         {tuple(rows[original][:2]): shuffled.heldout_for_row[i]
                          for i, original in enumerate(order)})
        self.assertIsInstance(first.train_ids, tuple)
        with self.assertRaises(Exception):
            first.train_ids += (99,)

    def test_malformed_rows_rejected(self):
        invalid = [[], [[1, 2, 3]], [[1, 2], [3, 4, 5, 6]],
                   [[math.nan, 2]], [[math.inf, 2]], [["bad", 2]],
                   ["1 2"], [[1, 2, 3, 4, 5, 6, 7]]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                split_features(rows, "a", "s")
        with self.assertRaises(ValueError):
            split_features([[0, 0]], "", "seed")
        with self.assertRaises(ValueError):
            split_features([[0, 0]], "view", "")


if __name__ == "__main__":
    unittest.main()
