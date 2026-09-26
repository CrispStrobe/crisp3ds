"""Checks for frozen-point context verification without reading the live images."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from scripts.pipes_context import run


def observation(image, feature_id, xy=(10.5, 20.5)):
    return {"image": image, "original_feature_id": feature_id,
            "xy_edge_frame": list(xy)}


class ContextTests(unittest.TestCase):
    def test_strict_mutual_ratio_uses_all_descriptor_competitors(self):
        a = [np.array([0., 0.], dtype=np.float32),
             np.array([1., 0.], dtype=np.float32),
             np.array([10., 0.], dtype=np.float32)]
        b = [np.array([.45, 0.], dtype=np.float32),
             np.array([10., 0.], dtype=np.float32),
             np.array([30., 0.], dtype=np.float32)]
        self.assertEqual(run.context_matches(a, b), {(2, 1)})
        b[1] = None
        self.assertEqual(run.context_matches(a, b), set())

    def test_all_pair_rule_and_size_one_control(self):
        point = {"id": 1, "observations": [observation("a", 0),
                                             observation("b", 0), observation("c", 0)]}
        descriptors = {name: [np.ones(128, dtype=np.float32)] for name in "abc"}
        original = {("a", "b"): {(0, 0)}, ("a", "c"): {(0, 0)},
                    ("b", "c"): set()}
        candidate = {pair: {(0, 0)} for pair in original}
        verdict = run.point_verdict(point, candidate, descriptors, original)
        self.assertTrue(verdict["retained"])
        self.assertFalse(verdict["baseline_all_pairs_retained"])
        self.assertEqual(verdict["baseline_pairs_passed"], 2)
        candidate[("a", "c")] = set()
        verdict = run.point_verdict(point, candidate, descriptors, original)
        self.assertFalse(verdict["retained"])
        self.assertEqual(verdict["pairs_passed"], 2)
        descriptors["c"][0] = None
        verdict = run.point_verdict(point, candidate, descriptors, original)
        self.assertEqual(verdict["reason"], "missing_descriptor")
        self.assertEqual(verdict["pairs_missing_descriptor"], 2)

    def test_compute_tracks_ids_and_preserves_original_keypoints(self):
        originals = [cv2.KeyPoint(10., 20., 4., 45., .1, 3, 99),
                     cv2.KeyPoint(30., 40., 6., 90., .2, 4, 98)]
        before = [(k.pt, k.size, k.angle, k.octave, k.class_id) for k in originals]
        gray = np.zeros((60, 60), dtype=np.uint8)

        def compute(_, requested):
            self.assertEqual([k.class_id for k in requested], [0, 1])
            self.assertEqual([k.size for k in requested], [8., 12.])
            self.assertEqual([k.angle for k in requested], [45., 90.])
            self.assertEqual([k.octave for k in requested], [3, 4])
            return requested[::-1], np.stack([np.ones(128), np.zeros(128)]).astype(np.float32)

        result = run.enlarged_descriptors(Mock(compute=compute), gray, originals)
        self.assertEqual([k.class_id for k in originals], [99, 98])
        self.assertEqual(before, [(k.pt, k.size, k.angle, k.octave, k.class_id)
                                  for k in originals])
        self.assertEqual(result[0][0], 0.)
        self.assertEqual(result[1][0], 1.)

    def test_compute_rejects_duplicate_and_moved_ids(self):
        originals = [cv2.KeyPoint(10., 20., 4.), cv2.KeyPoint(30., 40., 6.)]
        gray = np.zeros((60, 60), dtype=np.uint8)
        duplicate = lambda _, requested: ([requested[0], requested[0]],
                                          np.ones((2, 128), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run.enlarged_descriptors(Mock(compute=duplicate), gray, originals)
        def moved(_, requested):
            requested[0].pt = (11., 20.)
            return [requested[0]], np.ones((1, 128), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "center changed"):
            run.enlarged_descriptors(Mock(compute=moved), gray, originals)

    def test_frozen_coordinates_and_inputs_fail_closed(self):
        points = [{"id": i, "observations": [observation("a", 0), observation("b", 0)],
                   "source_track": [["a", 0], ["b", 0]]} for i in range(1, 259)]
        xy = {"a": [[10.5, 20.5]], "b": [[10.5, 20.5]]}
        run.assert_frozen_observations(points, xy, {"a": 1, "b": 1})
        points[0]["observations"][0]["xy_edge_frame"][0] += 1e-4
        with self.assertRaisesRegex(ValueError, "coordinate mismatch"):
            run.assert_frozen_observations(points, xy, {"a": 1, "b": 1})
        with self.assertRaisesRegex(ValueError, "changed during run"):
            run.assert_unchanged({"file": "old"}, {"file": "new"})


if __name__ == "__main__":
    unittest.main()
