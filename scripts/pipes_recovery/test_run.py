"""Small image-free checks for frozen-anchor recovery decisions."""

import unittest

from scripts.pipes_recovery import run


def point(pid=1):
    return {"id": pid, "xyz": [1., 2., 3.],
            "observations": [
                {"image": "DSC_0636.JPG", "original_feature_id": 8},
                {"image": "DSC_0634.JPG", "original_feature_id": 4},
                {"image": "DSC_0635.JPG", "original_feature_id": 5}],
            "source_track": [["DSC_0636.JPG", 8], ["DSC_0634.JPG", 4],
                             ["DSC_0635.JPG", 5]]}


def coordinates():
    xy = {name: [[float(i), float(i + 1)] for i in range(10)] for name in run.ALL}
    return xy


def matrix(supports, pair=True):
    matched = {(original, new): {} for original in run.ORIGINAL for new in run.NEW}
    matched.update({(a, b): {} for a_index, a in enumerate(run.NEW)
                    for b in run.NEW[a_index + 1:]})
    for new, fid in supports:
        matched[("DSC_0634.JPG", new)][(4, fid)] = .5
    if pair and len(supports) >= 2:
        a, b = supports[:2]
        matched[(a[0], b[0])][(a[1], b[1])] = .6
    return matched


class RecoveryTests(unittest.TestCase):
    def test_lexical_anchor_stable_and_nonanchor_ignored(self):
        self.assertEqual(run.anchor_for(point()), ("DSC_0634.JPG", 4))
        supports = [(run.NEW[0], 2), (run.NEW[1], 3)]
        matches = matrix(supports)
        # Matching another original observation cannot create new support.
        matches[("DSC_0635.JPG", run.NEW[2])][(5, 1)] = .1
        seen = []

        def reconstruct(track, *_):
            seen.append(track)
            return {"xyz": [4., 5., 6.], "source_track": [list(v) for v in track],
                    "observations": [{"image": n, "original_feature_id": fid,
                                      "reprojection_px": .2} for n, fid in track]}, None

        candidate, decision = run.decision_for(point(), matches, {}, {}, coordinates(), reconstruct)
        self.assertEqual(seen, [((run.ORIGINAL[0], 4), *supports)])
        self.assertEqual(candidate["xyz"], [4., 5., 6.])
        self.assertEqual(decision["new_view_support_count"], 2)
        self.assertEqual(len(candidate["observations"]), 3)

    def test_requires_two_distinct_new_views_and_immutable_fallback(self):
        frozen = point()
        candidate, decision = run.decision_for(frozen, matrix([(run.NEW[0], 2)]),
                                                {}, {}, coordinates())
        self.assertEqual(decision["reason"], "fewer_than_two_new_views")
        self.assertTrue(candidate["baseline_fallback"])
        self.assertEqual(candidate["xyz"], frozen["xyz"])
        candidate["xyz"][0] = 99
        self.assertEqual(frozen["xyz"], [1., 2., 3.])

    def test_all_new_view_pairs_must_agree(self):
        supports = [(run.NEW[0], 2), (run.NEW[1], 3), (run.NEW[2], 4)]
        called = []

        def forbidden(*_):
            called.append(True)
            raise AssertionError("must reject before reconstruction")

        _, decision = run.decision_for(point(), matrix(supports), {}, {},
                                       coordinates(), forbidden)
        self.assertEqual(decision["reason"], "new_view_pair_conflict")
        self.assertEqual(len(decision["new_view_pair_proofs"]), 3)
        self.assertEqual(called, [])

    def test_reconstructor_failure_falls_back_without_old_xyz_as_fit_input(self):
        supports = [(run.NEW[0], 2), (run.NEW[1], 3)]
        calls = []

        def failed(track, images, cameras, xy):
            calls.append((track, images, cameras, xy))
            return None, "high_reprojection"

        frozen = point()
        candidate, decision = run.decision_for(frozen, matrix(supports), {}, {},
                                                coordinates(), failed)
        self.assertEqual(decision["reason"], "high_reprojection")
        self.assertEqual(candidate["xyz"], frozen["xyz"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("DSC_0635.JPG", [name for name, _ in calls[0][0]])

    def test_shared_feature_rejects_both_admissible_points(self):
        candidates, decisions = [], []
        for pid in (1, 2):
            p = point(pid)
            c, d = run.decision_for(p, matrix([(run.NEW[0], 2), (run.NEW[1], 3)]),
                                    {}, {}, coordinates(), lambda track, *_: (
                                        {"xyz": [4., 5., 6.],
                                         "source_track": [list(v) for v in track],
                                         "observations": [{"image": n, "original_feature_id": f,
                                                           "reprojection_px": 0.}
                                                          for n, f in track]}, None))
            candidates.append(c)
            decisions.append(d)
        self.assertEqual(run.reject_shared_new_features(candidates, decisions), 2)
        self.assertEqual([d["reason"] for d in decisions],
                         ["shared_new_view_feature"] * 2)
        self.assertEqual([c["xyz"] for c in candidates], [[1., 2., 3.]] * 2)


if __name__ == "__main__":
    unittest.main()
