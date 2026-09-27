"""Synthetic-only checkerboard detector, planar PnP, temporal and seal checks."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from scripts.object_motion import checkerboard_pose as diagnostic


def camera_from_board(center):
    center = np.asarray(center, dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack((right, down, forward)).T
    return rotation, -rotation @ center


def synthetic_corners(center, noise=0.0, seed=0):
    rotation, translation = camera_from_board(center)
    rvec = cv2.Rodrigues(rotation)[0]
    points = cv2.projectPoints(diagnostic.OBJECT_CORNERS, rvec, translation,
                               diagnostic.K, None)[0].reshape(8, 9, 2)
    if noise:
        points += np.random.default_rng(seed).normal(0, noise, points.shape)
    return points


class CheckerboardPoseTests(TestCase):
    def test_full_9_by_8_corner_detector_on_rendered_board(self):
        gray = np.full((1024, 1280), 180, np.uint8)
        left, top, square = 420, 320, 36
        for row in range(9):
            for column in range(10):
                color = 0 if (row + column) % 2 else 255
                cv2.rectangle(gray, (left + square * column, top + square * row),
                              (left + square * (column + 1) - 1,
                               top + square * (row + 1) - 1), color, -1)
        result = diagnostic.detect_frame(gray, "synthetic.jpg", 0)
        self.assertTrue(result["detected"])
        self.assertEqual(len(result["candidates"]), 4)
        self.assertGreater(result["hull_fraction"], diagnostic.MIN_HULL_FRACTION)

    def test_planar_branches_and_180_label_gauge(self):
        points = synthetic_corners([12, 3, 30], noise=0.08, seed=4)
        candidates = diagnostic.pose_candidates(points)
        self.assertEqual(len(candidates), 4)
        self.assertEqual({candidate["label_flip_180"] for candidate in candidates}, {0, 1})
        for branch in (0, 1):
            a, b = [candidate for candidate in candidates
                    if candidate["planar_branch"] == branch]
            self.assertAlmostEqual(a["rms_px"], b["rms_px"], places=8)
        best = min((candidate for candidate in candidates if candidate["label_flip_180"] == 0),
                   key=lambda candidate: candidate["rms_px"])
        self.assertLess(best["rms_px"], 0.2)
        self.assertLess(np.linalg.norm(best["center"] - [12, 3, 30]), 0.2)

    def test_cyclic_selection_recovers_relative_reversals(self):
        frames = []
        flipped = {3, 4, 8}
        for index in range(12):
            angle = 2 * math.pi * index / 12
            center = np.array([12 * math.cos(angle), 12 * math.sin(angle), 30.0])
            points = synthetic_corners(center, noise=0.04, seed=index)
            if index in flipped:
                points = points[::-1, ::-1].copy()
            frames.append({"name": f"frame{index}", "slot_index": index * 5,
                           "candidates": diagnostic.pose_candidates(points)})
        selected = diagnostic.select_cycle(frames, 60)
        self.assertIsNotNone(selected)
        self.assertEqual({i for i, item in enumerate(selected["chosen"])
                          if item["label_flip_180"]}, flipped)
        self.assertTrue(all(item["rms_px"] < 0.2 for item in selected["chosen"]))
        self.assertIn("global 180-degree", selected["gauge"])

    def test_sealed_train_preflight_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            slots = [f"frame_{i:02d}.jpg" for i in range(60)]
            profile = root / "profile.json"
            profile.write_text(json.dumps({"schema": "turntable_full_turn_profile_v1",
                                           "full_turn": True, "uniform_slots": True, "slots": slots}))
            records = []
            for name in slots[:48]:
                data = name.encode()
                path = images / name
                path.write_bytes(data)
                records.append({"name": name, "source": str(path), "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()})
            manifest = root / "inputs.json"
            manifest.write_text(json.dumps(records))
            with patch.multiple(diagnostic, MANIFEST=manifest, MANIFEST_SHA256=diagnostic.digest(manifest),
                                TRAIN_IMAGES=images, PROFILE=profile,
                                PROFILE_SHA256=diagnostic.digest(profile), EXTERNAL=root):
                result, _ = diagnostic.preflight()
                self.assertEqual(len(result["images"]), 48)
                self.assertFalse(result["writes"])
                (images / slots[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    diagnostic.preflight()
