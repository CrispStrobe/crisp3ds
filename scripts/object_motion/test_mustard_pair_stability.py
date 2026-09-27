"""Analytic checks for sealed-pair perturbation construction and reporting."""

import unittest
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from scripts.object_motion import mustard_pair_stability as target


class PairStabilityTests(unittest.TestCase):
    def test_index_loader_reverses_database_order_and_checks_pose_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "fixture.db"
            left = np.column_stack((np.array([[640., 512.], [641., 513.], [642., 514.], [643., 515.]],
                                              dtype="<f4"), np.zeros((4, 4), dtype="<f4")))
            right = left.copy()
            right[:, 0] += 10
            with closing(sqlite3.connect(db)) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
                connection.executemany("INSERT INTO images VALUES (?,?)", [(1, "right"), (2, "left")])
                connection.executemany("INSERT INTO keypoints VALUES (?,?,?,?)",
                                       [(1, 4, 6, right.tobytes()), (2, 4, 6, left.tobytes())])
                pair_id = 1 * target.pose.MAX_IMAGE_ID + 2
                connection.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?)",
                                   (pair_id, 2, 2, np.array([[0, 1], [2, 3]], dtype="<u4").tobytes()))
                connection.commit()
            row = {"left": "left", "right": "right", "verified_correspondences": 2,
                   "points_left": (left[[1, 3], :2].astype(float) - [640, 512]) / 1536,
                   "points_right": (right[[0, 2], :2].astype(float) - [640, 512]) / 1536}
            attached = target.attach_index_details(db, [row])[0]
            np.testing.assert_array_equal(attached["feature_index_pairs"], [[1, 0], [3, 2]])
            changed = row.copy()
            changed["points_left"] = row["points_left"].copy()
            changed["points_left"][0, 0] += .01
            with self.assertRaisesRegex(ValueError, "index-to-coordinate"):
                target.attach_index_details(db, [changed])

    def test_leave_one_out_preserves_order_and_exactly_one_removal(self):
        row = {"points_left": np.arange(8).reshape(4, 2),
               "points_right": np.arange(8, 16).reshape(4, 2),
               "verified_correspondences": 4}
        changed = target.subset(row, [0, 2, 3])
        np.testing.assert_array_equal(changed["points_left"], [[0, 1], [4, 5], [6, 7]])
        np.testing.assert_array_equal(changed["points_right"], [[8, 9], [12, 13], [14, 15]])
        self.assertEqual(changed["verified_correspondences"], 3)
        self.assertEqual(row["verified_correspondences"], 4)

    def test_only_other_arm_verified_pairs_with_identical_keypoints_can_be_added(self):
        points = np.array([[640., 512.], [641., 513.], [642., 514.], [643., 515.]])
        own = {"feature_index_pairs": np.array([[0, 0], [1, 1]], dtype="<u4"),
               "all_keypoints_left": points.copy(), "all_keypoints_right": points.copy(),
               "points_left": (points[:2] - [640, 512]) / 1536,
               "points_right": (points[:2] - [640, 512]) / 1536}
        other = {"feature_index_pairs": np.array([[0, 0], [2, 2], [3, 3], [5, 0]], dtype="<u4"),
                 "all_keypoints_left": points.copy(), "all_keypoints_right": points.copy()}
        other["all_keypoints_right"][3, 0] += 1
        accepted, incompatible = target.addition_candidates(own, other)
        self.assertEqual(accepted, [(2, 2)])
        self.assertEqual(incompatible, [(3, 3), (5, 0)])
        added = target.augmented(own, accepted[0])
        self.assertEqual(added["verified_correspondences"], 3)
        np.testing.assert_array_equal(added["points_left"][-1], (points[2] - [640, 512]) / 1536)
        self.assertEqual(len(own["points_left"]), 2)

    def test_rotation_difference_is_reported_even_when_gate_abstains(self):
        identity = np.eye(3)
        angle = np.deg2rad(90)
        quarter = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        baseline = {"seeds": [{"seed": 17, "status": "unavailable", "rotation": identity.tolist()}]}
        fit = {"status": "unavailable", "reason": "parallax gate failed", "seeds": [
            {"seed": 17, "status": "unavailable", "rotation": quarter.tolist(),
             "essential_inliers": 19, "cheirality_positive": 13,
             "median_inter_ray_parallax_degrees": .3}]}
        compact = target.compact_fit(fit, baseline)
        self.assertAlmostEqual(compact["max_rotation_difference_from_full_input_degrees"], 90)
        self.assertEqual(compact["status"], "unavailable")
        self.assertEqual(compact["seeds"][0]["essential_inliers"], 19)
        self.assertNotIn("rotation", compact["seeds"][0])


if __name__ == "__main__":
    unittest.main()
