import math
import unittest

from scripts.sparse_verify.verify import project
from scripts.three_view_audit.run import (
    closest_ray_midpoint, graph_components, longest_baseline_index, orient, summary,
)


CAM = dict(model="SIMPLE_PINHOLE", width=768, height=512,
           params=[600.0, 384.0, 256.0])
R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def image(center):
    return dict(R=R, t=[-v for v in center], center=center, camera_id=1)


def xy(point, view):
    return list(project(CAM, [v+t for v, t in zip(point, view["t"])]))


class ThreeViewAuditTest(unittest.TestCase):
    def test_closest_ray_midpoint_known_geometry(self):
        point = [0.5, 0.2, 4]
        a, b = [0, 0, 0], [1, 0, 0]
        da = [x/math.dist(a, point) for x in point]
        db = [(x-c)/math.dist(b, point) for x, c in zip(point, b)]
        got, gap, sa, sb = closest_ray_midpoint(a, da, b, db)
        self.assertLess(math.dist(got, point), 1e-11)
        self.assertLess(gap, 1e-11)
        self.assertGreater(sa, 0)
        self.assertGreater(sb, 0)

    def test_all_orientations_exact_then_third_outlier(self):
        point = [0.5, 0.2, 4]
        views = {"a": image([0, 0, 0]), "b": image([1, 0, 0]),
                 "c": image([0, 1, 0])}
        features = {name: [xy(point, view)] for name, view in views.items()}
        cycle = (("a", 0), ("b", 0), ("c", 0))
        for orientation in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            result = orient(cycle, *orientation, views, {1: CAM}, features)
            self.assertEqual(result["status"], "scored")
            self.assertLess(result["third_reprojection_px"], 1e-8)
            self.assertLess(result["first_pair_reprojection_px"], 1e-8)
        features["c"][0][1] += 10
        bad = orient(cycle, 0, 1, 2, views, {1: CAM}, features)
        self.assertGreater(bad["third_reprojection_px"], 4)

    def test_low_parallax_is_explicitly_unavailable(self):
        point = [0, 0, 1000]
        views = {"a": image([0, 0, 0]), "b": image([1, 0, 0]),
                 "c": image([0, 1, 0])}
        features = {name: [xy(point, view)] for name, view in views.items()}
        result = orient((("a", 0), ("b", 0), ("c", 0)), 0, 1, 2,
                        views, {1: CAM}, features)
        self.assertEqual(result["status"], "low_parallax")
        self.assertIsNone(result["third_reprojection_px"])

    def test_graph_conflict_and_failure_cluster(self):
        a, b, c, clone = ("a", 0), ("b", 0), ("c", 0), ("a", 1)
        pairs = [(a, b), (b, c), (a, c), (clone, b)]
        cycle = dict(id=0, refs=[dict(image=n, feature_id=i) for n, i in (a, b, c)],
                     lexical=dict(status="scored", third_reprojection_px=8))
        result = graph_components(pairs, [cycle])
        self.assertEqual(result["component_count"], 1)
        self.assertEqual(result["components_with_same_image_feature_conflict"], 1)
        self.assertEqual(result["failed_cycles_in_conflicted_components"], 1)

    def test_camera_only_policy_tie_break(self):
        choices = [dict(images=["b", "c"], camera_center_distance=5),
                   dict(images=["a", "c"], camera_center_distance=5),
                   dict(images=["a", "b"], camera_center_distance=2)]
        self.assertEqual(longest_baseline_index(choices), 1)
        choices[0]["camera_center_distance"] = 6
        self.assertEqual(longest_baseline_index(choices), 0)

    def test_fixed_denominator_counts_missing_as_bad(self):
        low = dict(status="low_parallax", third_reprojection_px=None)
        good = dict(status="scored", third_reprojection_px=1)
        bad = dict(status="scored", third_reprojection_px=5)
        cycles = [dict(lexical=good, all_orientations=[good, low, bad],
                       longest_baseline=low, longest_baseline_orientation_index=1,
                       refs=[dict(image="a", feature_id=0)]),
                  dict(lexical=bad, all_orientations=[bad, good, low],
                       longest_baseline=bad, longest_baseline_orientation_index=0,
                       refs=[dict(image="b", feature_id=0)])]
        result = summary(cycles)["longest_baseline_on_fixed_lexical_eligible"]
        self.assertEqual(result["original_eligible_count"], 2)
        self.assertEqual(result["missing_inclusive_bad_over_4px_count"], 2)
        self.assertEqual(result["target_changed_count"], 1)


if __name__ == "__main__":
    unittest.main()
