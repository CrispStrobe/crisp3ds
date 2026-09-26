"""Population, reprojection and paired-result checks for recovery scoring."""

import copy
import unittest

import numpy as np

from scripts.pipes_recovery import evaluate as ev


def fixture():
    camera = {"model": "PINHOLE", "width": 1024, "height": 682,
              "params": [100., 100., 512., 341.]}
    images = []
    for n in range(634, 648):
        images.append({"name": f"DSC_{n:04d}.JPG", "camera_id": 0,
                       "camera": camera,
                       "qvec": [1., 0., 0., 0.], "tvec": [0., 0., 0.],
                       "pose_convention": "world_to_camera"})
    metadata = {"images": images, "cameras": {"0": {
                 "width": 1024, "height": 682, "params": [100., 100., 512., 341.]}}}
    baseline = {"points": [{"id": i, "xyz": [0., 0., 1.],
                 "observations": [{"image": "DSC_0634.JPG", "original_feature_id": i,
                                   "xy_edge_frame": [512., 341.]}],
                 "source_track": [["DSC_0634.JPG", i]]}
                for i in range(1, 259)]}
    candidates = {"schema": "pipes_anchor_recovery_candidates_v1", "points": [
        {"id": i, "xyz": [0., 0., 1.], "baseline_fallback": True,
         "source_anchor": {"image": "DSC_0634.JPG", "original_feature_id": i,
                           "xy_edge_frame": [512., 341.]},
         "source_track": [["DSC_0634.JPG", i]],
         "reason": "no_new_support"} for i in range(1, 259)]}
    return baseline, candidates, metadata


class RecoveryEvaluateTest(unittest.TestCase):
    def validate(self, candidate, baseline, metadata, raw_candidate=None, raw_baseline=None):
        raw_baseline = raw_baseline or [p["xyz"] for p in baseline["points"]]
        raw_candidate = raw_candidate or [p["xyz"] for p in candidate["points"]]
        return ev.validate_candidates(candidate, baseline, raw_baseline, raw_candidate, metadata,
                                      size=(1024, 682))

    def test_empty_recovery_is_exact_baseline_with_all_ids(self):
        baseline, candidate, metadata = fixture()
        queries, recovered, reasons = self.validate(candidate, baseline, metadata)
        self.assertEqual(len(queries), 258)
        self.assertEqual(recovered, [])
        self.assertEqual(reasons, {"no_new_support": 258})
        before = np.arange(258, dtype=np.float64) / 1000
        after = before.copy()
        self.assertEqual(ev.paired_changes(before, after),
                         {key: {"near_to_far": 0, "far_to_near": 0}
                          for key in ("0.01", "0.02", "0.05", "0.10")})

    def test_exact_fallback_coordinate_tokens_and_ids(self):
        baseline, candidate, metadata = fixture()
        raw_base = [p["xyz"] for p in baseline["points"]]
        raw_candidate = copy.deepcopy(raw_base)
        raw_base[0] = [("float", "0.0"), ("float", "0.0"), ("float", "1.0")]
        raw_candidate[0] = [("int", "0"), ("float", "0.0"), ("float", "1.0")]
        with self.assertRaisesRegex(ValueError, "fallback XYZ tokens"):
            self.validate(candidate, baseline, metadata, raw_candidate, raw_base)
        candidate["points"][4]["id"] = 6
        with self.assertRaisesRegex(ValueError, "IDs"):
            self.validate(candidate, baseline, metadata)

    def test_recovered_must_have_anchor_two_new_views_positive_depth_and_four_px(self):
        baseline, candidate, metadata = fixture()
        point = candidate["points"][0]
        point["baseline_fallback"] = False
        point["xyz"] = [0., 0., 2.]
        point["observations"] = [
            {"image": name, "original_feature_id": i,
             "xy_edge_frame": [512., 341.], "reprojection_px": 0.}
            for i, name in ((1, "DSC_0634.JPG"), (3, "DSC_0638.JPG"), (4, "DSC_0639.JPG"))]
        point["source_track"] = [[ob["image"], ob["original_feature_id"]]
                                 for ob in point["observations"]]
        queries, recovered, _ = self.validate(candidate, baseline, metadata)
        self.assertEqual(recovered, [1])
        self.assertEqual(queries[0], [0., 0., 2.])
        point["xyz"] = [0., 0., -2.]
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            self.validate(candidate, baseline, metadata)
        point["xyz"] = [0., 0., 2.]
        point["observations"][1]["xy_edge_frame"] = [517., 341.]
        with self.assertRaisesRegex(ValueError, "reprojection"):
            self.validate(candidate, baseline, metadata)
        point["observations"][1]["xy_edge_frame"] = [512., 341.]
        point["observations"][2]["image"] = "DSC_0635.JPG"
        with self.assertRaisesRegex(ValueError, "two distinct new views"):
            self.validate(candidate, baseline, metadata)

    def test_paired_near_far_counts(self):
        before = np.array([.009, .012, .019, .025, .049, .06, .11])
        after = np.array([.012, .009, .025, .019, .06, .049, .09])
        changes = ev.paired_changes(before, after)
        self.assertEqual(changes["0.01"], {"near_to_far": 1, "far_to_near": 1})
        self.assertEqual(changes["0.02"], {"near_to_far": 1, "far_to_near": 1})
        self.assertEqual(changes["0.05"], {"near_to_far": 1, "far_to_near": 1})
        self.assertEqual(changes["0.10"], {"near_to_far": 0, "far_to_near": 1})

    def test_decisions_must_cover_frozen_population(self):
        baseline, candidate, _ = fixture()
        decisions = {"schema": "pipes_anchor_recovery_decisions_v1", "decisions": [
            {"original_point_id": i, "source_anchor": candidate["points"][i-1]["source_anchor"],
             "original_xyz": [0., 0., 1.], "original_source_track": [["DSC_0634.JPG", i]],
             "status": "unresolved", "reason": "no_new_support"}
            for i in range(1, 259)]}
        ev.validate_decisions(decisions, candidate, baseline)
        decisions["decisions"][0]["original_point_id"] = 2
        with self.assertRaisesRegex(ValueError, "decision ID"):
            ev.validate_decisions(decisions, candidate, baseline)


if __name__ == "__main__":
    unittest.main()
