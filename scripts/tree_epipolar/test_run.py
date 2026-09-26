#!/usr/bin/env python3
"""Synthetic geometry and partition checks for the fixed diagnostic."""

import math
import unittest

import cv2
import numpy as np

from run import metrics, normalize_f, pair_result, sampson, supplied_f


def camera(view_id, shift):
    return dict(id=view_id, K=[[500., 0., 384.], [0., 500., 256.], [0., 0., 1.]],
                R=np.eye(3).tolist(), t=[shift, 0., 0.])


class DummyKeypoint:
    def __init__(self, pt):
        self.pt = pt


class GeometryTests(unittest.TestCase):
    def test_known_camera_outliers_and_split(self):
        rng = np.random.default_rng(17)
        a, b = camera(1, 0), camera(4, -.5)
        ka = np.asarray(a["K"])
        points = rng.uniform([-1, -.7, 3.], [1., .7, 7.], size=(80, 3))
        xy_a = (ka @ points.T).T
        xy_a = (xy_a[:, :2] / xy_a[:, 2:]).astype(np.float32)
        moved = points + np.array(b["t"])
        xy_b = (ka @ moved.T).T
        xy_b = (xy_b[:, :2] / xy_b[:, 2:]).astype(np.float32)
        xy_a += rng.normal(0, .1, xy_a.shape).astype(np.float32)
        xy_b += rng.normal(0, .1, xy_b.shape).astype(np.float32)
        xy_b[[3, 10, 27, 49]] += np.array([0, 40], dtype=np.float32)
        # Unique descriptors ensure that mutual matching retains all points.
        descriptors = np.eye(80, dtype=np.float32)
        features = {1:([DummyKeypoint(tuple(p)) for p in xy_a], descriptors),
                    4:([DummyKeypoint(tuple(p)) for p in xy_b], descriptors.copy())}
        outcome, rows = pair_result(a, b, features)
        self.assertEqual(outcome["status"], "available")
        self.assertEqual(len(rows), 80)
        self.assertEqual(outcome["heldout_count"], 16)
        self.assertEqual([r["order"] for r in rows if r["split"] == "holdout"], list(range(4, 80, 5)))
        self.assertLess(outcome["supplied_heldout"]["median_px"], .5)
        self.assertLess(outcome["fitted_heldout"]["median_px"], 1.)
        # Changing held-out coordinates leaves the train-only fitted model fixed.
        features[4][0][4].pt = (100., 100.)
        changed, _ = pair_result(a, b, features)
        self.assertTrue(np.allclose(outcome["fitted_F"], changed["fitted_F"]))

    def test_degenerate_identical_camera(self):
        self.assertIsNone(supplied_f(camera(1, 0), camera(4, 0)))

    def test_f_scale_and_sign_invariance(self):
        f = supplied_f(camera(1, 0), camera(4, -.5))
        self.assertIsNotNone(f)
        pa, pb = [350., 240.], [300., 241.]
        expected = sampson(f, pa, pb)
        self.assertAlmostEqual(expected, sampson(normalize_f(f*73), pa, pb))
        self.assertAlmostEqual(expected, sampson(normalize_f(-f*1e-7), pa, pb))
        self.assertEqual(metrics([1., None, 3.])["within_1px_fraction"], 1/3)


if __name__ == "__main__":
    unittest.main()
