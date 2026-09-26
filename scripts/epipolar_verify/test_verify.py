"""Independent arithmetic and tamper fixtures for verify.py."""

import math
import unittest

from verify import (f_difference_up_to_scale, matrix, near, sqrt_sampson, supplied_f,
                    summary, verify_split)


class GeometryTests(unittest.TestCase):
    def test_arbitrary_f_and_scale_sign(self):
        f = [[0, -2, 3], [1, 0, -4], [-5, 6, 7]]
        a, b = [2.5, -1], [4, 3]
        got = sqrt_sampson(f, a, b)
        numerator = abs(sum(bx * ax for bx, ax in zip([*b, 1], [5, -1.5, -11.5])))
        denominator = math.sqrt(5**2 + (-1.5)**2 + (-2)**2 + (-2)**2)
        self.assertAlmostEqual(got, numerator / denominator)
        self.assertAlmostEqual(got, sqrt_sampson([[10*x for x in row] for row in f], a, b))
        self.assertAlmostEqual(got, sqrt_sampson([[-x for x in row] for row in f], a, b))
        for scale in (1e-12, 1e12):
            self.assertAlmostEqual(got, sqrt_sampson([[scale*x for x in row] for row in f], a, b))
        self.assertLess(f_difference_up_to_scale(f, [[-7*x for x in row] for row in f]), 1e-14)

    def test_camera_pose_and_rotation(self):
        ident = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        rot_z = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        a = {"K": [2, 4, 3, 5], "R": ident, "t": [0, 0, 0]}
        b = {"K": [2, 4, 3, 5], "R": rot_z, "t": [1, 2, 3]}
        f = supplied_f(a, b)
        # World point (1,2,10): camera b sees (-1,3,13).
        pa = [3 + 2*1/10, 5 + 4*2/10]
        pb = [3 - 2/13, 5 + 12/13]
        self.assertAlmostEqual(sqrt_sampson(f, pa, pb), 0, places=12)
        self.assertLess(f_difference_up_to_scale(f, [[-2*x for x in row] for row in f]), 1e-14)

    def test_degeneracies(self):
        with self.assertRaises(ValueError):
            f_difference_up_to_scale([[0]*3]*3, [[1, 0, 0]]*3)
        with self.assertRaises(ValueError):
            matrix([float("nan")] + [0]*8)
        self.assertIsNone(sqrt_sampson([[0]*3]*3, [1, 2], [3, 4]))
        self.assertEqual(summary([[0]*3]*3, [{"a_xy": [1, 2], "b_xy": [3, 4]}])["degenerate_count"], 1)

    def test_split_tamper(self):
        matches = [{"order": i, "query_idx": i, "train_idx": 100+i,
                    "a_xy": [i, 0], "b_xy": [i, 0],
                    "split": "holdout" if (i+1)%5 == 0 else "train"} for i in range(11)]
        train, heldout = verify_split(matches, 9, 2)
        self.assertEqual(len(train), 9)
        self.assertEqual(len(heldout), 2)
        matches[4]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "split changed"):
            verify_split(matches, 9, 2)
        matches[4]["split"] = "holdout"
        matches[3]["query_idx"] = matches[2]["query_idx"]
        with self.assertRaisesRegex(ValueError, "query descriptor order"):
            verify_split(matches, 9, 2)

    def test_reported_metric_tamper(self):
        with self.assertRaisesRegex(ValueError, "reported"):
            near(0.75, 0.5, "within_2px_fraction")
        with self.assertRaisesRegex(ValueError, "reported nan"):
            near(float("nan"), 0.5, "median_px")


if __name__ == "__main__":
    unittest.main()
