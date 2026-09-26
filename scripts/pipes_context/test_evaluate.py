"""Fixed-population evaluation and failure cases for sealed context verdicts."""

import unittest
from itertools import combinations

from scripts.pipes_context.evaluate import distance_summary, evaluate_data


POINT_HASH = "a" * 64
REPORT_HASH = "b" * 64


def fixture(distances=(0.005, 0.03, 0.15), candidate=(True, False, False),
            control=(True, True, False)):
    points = []
    for index in range(len(distances)):
        names = ("a", "b", "c") if index == 1 else ("a", "b")
        points.append({"id": index + 1, "xyz": [float(index), 0.0, 0.0],
                       "observations": [{"image": name, "original_feature_id": index}
                                        for name in names]})
    point_data = {"schema": "fixed_camera_sparse_points_v1", "points": points}
    report = {"schema": "eth3d_pipes_fixed_camera_sparse_v1", "reference_geometry_used": False,
              "accepted_tracks": len(points), "accepted_tracks_three_or_more_views": 1 if len(points) > 1 else 0,
              "accepted_observations": 2 * len(points) + (1 if len(points) > 1 else 0),
              "accepted_observations_by_image": {"a": len(points), "b": len(points),
                                                 "c": 1 if len(points) > 1 else 0},
              "accepted_tracks_by_support_views": {"2": len(points) - (1 if len(points) > 1 else 0),
                                                    "3": 1 if len(points) > 1 else 0},
              "views": [{"name": name} for name in ("a", "b", "c")]}
    score = {"schema": "eth3d_pipes_sparse_laser_proximity_v1", "status": "pass",
             "input_sha256": {"/frozen/points.json": POINT_HASH, "/frozen/report.json": REPORT_HASH},
             "summary": distance_summary(list(range(1, len(points) + 1)),
                                         dict(enumerate(distances, 1)), len(points)),
             "point_distances_m": [{"id": index + 1, "distance_m": distance}
                                   for index, distance in enumerate(distances)]}
    decisions = []
    for index, point in enumerate(points):
        pairs = list(combinations(point["observations"], 2))
        witnesses = [{"images": [a["image"], b["image"]],
                      "original_feature_ids": [index, index],
                      "pass": candidate[index], "missing_descriptor_images": []}
                     for a, b in pairs]
        decisions.append({"original_point_id": index + 1, "retained": candidate[index],
                          "baseline_all_pairs_retained": control[index],
                          "baseline_pairs_passed": len(pairs) if control[index] else 0,
                          "reason": "all_pairs_match" if candidate[index] else "pair_match_failed",
                          "pairs_passed": len(pairs) if candidate[index] else 0,
                          "pairs_total": len(pairs), "pairs_missing_descriptor": 0,
                          "pair_witnesses": witnesses})
    verdicts = {"schema": "pipes_context_frozen_point_verification_v1",
                "reference_geometry_used": False, "geometry_recomputed": False,
                "input_sha256_before": {"points.json": POINT_HASH, "report.json": REPORT_HASH},
                "input_sha256_after": {"points.json": POINT_HASH, "report.json": REPORT_HASH},
                "verdicts": decisions}
    verdicts.update({"frozen_points": len(points), "retained_points": sum(candidate),
                     "baseline_all_pairs_retained_points": sum(control),
                     "rejected_points": len(points) - sum(candidate)})
    return point_data, report, score, verdicts


def evaluate(data):
    return evaluate_data(*data, POINT_HASH, REPORT_HASH)


class EvaluationTests(unittest.TestCase):
    def test_rejection_cost_and_control(self):
        result = evaluate(fixture())
        self.assertEqual(result["retained_points"], 1)
        self.assertEqual(result["candidate"]["discarded_by_threshold"]["0.05"],
                         {"near_reference": 1, "far_from_reference": 1})
        self.assertEqual(result["baseline"]["distance"]["within_threshold_counts"]["0.05"], 2)
        self.assertEqual(result["candidate"]["distance"]["within_threshold_original_fractions"]["0.05"], 1 / 3)
        self.assertEqual(result["candidate"]["distance"]["missing_inclusive_error_fractions"]["0.05"], 2 / 3)
        self.assertEqual(result["unchanged_size_all_pairs_control"]["retained_points"], 2)
        self.assertEqual(result["candidate_minus_control"]["retained_near_reference_counts"]["0.05"], -1)
        self.assertEqual(result["candidate"]["support"]["observations_by_image"],
                         {"a": 1, "b": 1, "c": 0})
        self.assertEqual(result["three_or_more_views_descriptive"]["baseline_points"], 1)

    def test_empty_candidate_has_null_quantiles_and_zero_coverage(self):
        result = evaluate(fixture(candidate=(False, False, False)))
        summary = result["candidate"]["distance"]
        self.assertIsNone(summary["median_m"])
        self.assertIsNone(summary["within_threshold_fractions"]["0.01"])
        self.assertEqual(summary["within_threshold_original_fractions"]["0.01"], 0)
        self.assertEqual(summary["missing_inclusive_error_fractions"]["0.01"], 1)

    def test_all_retained_matches_baseline(self):
        result = evaluate(fixture(candidate=(True, True, True)))
        self.assertEqual(result["candidate"]["distance"], result["baseline"]["distance"])
        self.assertEqual(result["candidate"]["discarded_by_threshold"]["0.10"],
                         {"near_reference": 0, "far_from_reference": 0})

    def test_duplicate_and_missing_ids_rejected(self):
        data = list(fixture())
        data[3]["verdicts"][1]["original_point_id"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(data)
        data = list(fixture())
        data[3]["verdicts"].pop()
        with self.assertRaisesRegex(ValueError, "universe"):
            evaluate(data)

    def test_score_ids_and_hash_rejected(self):
        data = list(fixture())
        data[2]["point_distances_m"][1]["id"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(data)
        data = list(fixture())
        data[2]["input_sha256"]["/frozen/points.json"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate(data)

    def test_verdict_hash_boolean_and_xyz_rejected(self):
        data = list(fixture())
        data[3]["input_sha256_after"]["points.json"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hashes changed"):
            evaluate(data)
        data = list(fixture())
        data[3]["verdicts"][0]["retained"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            evaluate(data)
        data = list(fixture())
        data[3]["verdicts"][0]["baseline_all_pairs_retained"] = "true"
        with self.assertRaisesRegex(ValueError, "boolean"):
            evaluate(data)
        data = list(fixture())
        data[3]["verdicts"][0]["xyz"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "XYZ"):
            evaluate(data)

    def test_witness_and_declared_counts_rejected(self):
        data = list(fixture())
        data[3]["verdicts"][0]["pairs_passed"] = 0
        with self.assertRaisesRegex(ValueError, "pair counts"):
            evaluate(data)
        data = list(fixture())
        data[3]["retained_points"] = 2
        with self.assertRaisesRegex(ValueError, "summary counts"):
            evaluate(data)


if __name__ == "__main__":
    unittest.main()
