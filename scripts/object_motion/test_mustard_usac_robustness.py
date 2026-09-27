"""Analytic and deterministic checks for the pinned two-view method comparison."""

import unittest

import numpy as np

from scripts.object_motion import mustard_usac_robustness as target


def scene(planar: bool) -> dict:
    rng = np.random.default_rng(21 if planar else 19)
    points = rng.uniform([-1, -1, 2], [1, 1, 5], size=(80, 3))
    if planar:
        points[:, 2] = 3
    angle = np.deg2rad(18)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    moved = points @ rotation.T + [.5, 0, 0]
    return {"stratum": "middle", "left": "synthetic_a", "right": "synthetic_b",
            "cyclic_separation": 5, "verified_correspondences": 80,
            "two_view_config": 3,
            "points_left": points[:, :2] / points[:, 2, None],
            "points_right": moved[:, :2] / moved[:, 2, None]}


class UsacRobustnessTests(unittest.TestCase):
    @unittest.skipUnless(all(hasattr(target.cv2, method) for method in target.METHODS),
                         "OpenCV wheel lacks one or more USAC essential-matrix methods")
    def test_pinned_api_recovers_nonplanar_rotation_for_every_method(self):
        row = scene(False)
        for method in target.METHODS:
            with self.subTest(method=method):
                fit = target.fit(row, method)
                self.assertEqual(fit["status"], "distinct")
                self.assertAlmostEqual(fit["median_rotation_degrees"], 18, delta=2)
                self.assertEqual(len(fit["seeds"]), 3)
                self.assertTrue(all(seed["essential_inliers"] >= 15 for seed in fit["seeds"]))
                self.assertTrue(all(seed["cheirality_positive"] >= 12 for seed in fit["seeds"]))

    @unittest.skipUnless(hasattr(target.cv2, "USAC_MAGSAC"),
                         "OpenCV wheel lacks USAC_MAGSAC")
    def test_magsac_planar_scene_abstains_under_homography_gate(self):
        fit = target.fit(scene(True), "USAC_MAGSAC")
        self.assertEqual(fit["status"], "unavailable")
        self.assertIn("homography", fit["reason"])
        self.assertEqual(fit["homography_inliers"], 80)

    def test_ordering_and_removal_share_exact_feature_pair(self):
        pairs = np.array([[8, 4], [2, 7], [8, 1], [1, 9], [5, 3]], dtype="<u4")
        row = {"stratum": "middle", "left": "a", "right": "b",
               "feature_index_pairs": pairs,
               "points_left": np.arange(10).reshape(5, 2),
               "points_right": np.arange(10, 20).reshape(5, 2)}
        selected = target.selected_removals(row)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertEqual(selected, target.selected_removals(row))
        np.testing.assert_array_equal(target.ordered_row(row, "canonical")["feature_index_pairs"],
                                      [[1, 9], [2, 7], [5, 3], [8, 1], [8, 4]])
        np.testing.assert_array_equal(target.ordered_row(row, "reverse_canonical")["feature_index_pairs"],
                                      [[8, 4], [8, 1], [5, 3], [2, 7], [1, 9]])
        removed = selected[0]
        blob = target.ordered_row(row, "blob", removed)["feature_index_pairs"]
        canonical = target.ordered_row(row, "canonical", removed)["feature_index_pairs"]
        self.assertEqual(len(blob), 4)
        self.assertEqual({tuple(x) for x in blob}, {tuple(x) for x in canonical})
        self.assertNotIn(tuple(pairs[removed]), {tuple(x) for x in blob})
        np.testing.assert_array_equal(row["feature_index_pairs"], pairs)

    def test_removal_spread_uses_same_order_full_input(self):
        def raw(degrees):
            angle = np.deg2rad(degrees)
            rotation = [[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            return {"status": "distinct", "seeds": [{"rotation": rotation}]}
        baselines = {"blob": raw(0), "canonical": raw(0), "reverse_canonical": raw(90)}
        removals = [{"order": "reverse_canonical", "raw": raw(91)}]
        summary = target.summarise_method(baselines, removals)
        self.assertAlmostEqual(summary["max_baseline_order_rotation_spread_degrees"], 90)
        self.assertAlmostEqual(summary["max_same_order_removal_rotation_spread_degrees"], 1)

    def test_historical_control_calls_unchanged_runner(self):
        row = scene(False)
        self.assertEqual(target.fit(row, "RANSAC"), target.pose.fit_pair(row))


if __name__ == "__main__":
    unittest.main()
