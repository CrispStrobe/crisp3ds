"""Analytic, model-free fixtures fixed before real camera evaluation."""

import math
import json
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from scripts.object_motion.orbit_plausibility import evaluate, preflight, MODEL_FILES


PROFILE = {"schema": "turntable_full_turn_profile_v1", "full_turn": True,
           "uniform_slots": True, "slots": [f"frame_{i:02d}" for i in range(24)]}


def ring(*, phase=None, center_change=None, orientation_change=None, missing=()):
    rows = []
    for i, name in enumerate(PROFILE["slots"]):
        if i in missing:
            continue
        theta = 2 * math.pi * (phase(i) if phase else i) / 24
        center = np.array([math.cos(theta), math.sin(theta), 0.0])
        if center_change and i in center_change:
            center = center + np.asarray(center_change[i])
        forward = -np.array([math.cos(theta), math.sin(theta), 0.0])
        if orientation_change and i in orientation_change:
            forward = np.asarray(orientation_change[i], dtype=float)
        up = np.array([0., 0., 1.])
        right = np.cross(up, forward)
        rotation = np.column_stack((right, up, forward))
        rows.append({"name": name, "center": center, "camera_to_world_rotation": rotation})
    return rows


class OrbitPlausibilityTests(TestCase):
    def test_clean_ring_passes_under_similarity_and_rigid_rotation(self):
        rows = ring(missing=(2, 8, 14, 20))
        result = evaluate(rows, PROFILE)
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["winding_degrees"], 360)
        turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved = [{"name": row["name"], "center": 9 * turn @ row["center"] + [4, -3, 2],
                  "camera_to_world_rotation": turn @ row["camera_to_world_rotation"]} for row in rows]
        self.assertEqual(evaluate(moved, PROFILE)["status"], "pass")

    def test_adjacent_jump_and_orientation_are_separate_evidence(self):
        result = evaluate(ring(center_change={4: [3., 0., 0.]}), PROFILE)
        self.assertIn("adjacent_jump", result["failures"])
        result = evaluate(ring(orientation_change={4: [1., 0., 0.]}), PROFILE)
        self.assertIn("adjacent_jump", result["failures"])

    def test_fold_and_reversal_fail(self):
        fold = evaluate(ring(phase=lambda i: 2 * i), PROFILE)
        self.assertIn("opposing_view_collapse", fold["failures"])
        self.assertIn("angular_winding_or_reversal", fold["failures"])
        reverse = evaluate(ring(phase=lambda i: i if i <= 15 else 30 - i), PROFILE)
        self.assertIn("angular_winding_or_reversal", reverse["failures"])

    def test_orientation_facing_outward_fails(self):
        result = evaluate(ring(orientation_change={i: [math.cos(i * math.pi / 12),
                                                     math.sin(i * math.pi / 12), 0.] for i in range(24)}), PROFILE)
        self.assertIn("camera_orientation", result["failures"])

    def test_unavailable_and_explicit_profile(self):
        self.assertEqual(evaluate(ring(missing=tuple(range(8))), PROFILE)["status"], "unavailable")
        line = ring()
        for i, row in enumerate(line):
            row["center"] = [float(i), 0., 0.]
        self.assertEqual(evaluate(line, PROFILE)["status"], "unavailable")
        with self.assertRaisesRegex(ValueError, "explicit uniformly sampled"):
            evaluate(ring(), {**PROFILE, "full_turn": False})

    def test_preflight_reads_only_and_binds_input_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            for name in MODEL_FILES:
                (model / name).write_bytes(name.encode())
            profile = root / "profile.json"
            profile.write_text(json.dumps(PROFILE))
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            result = preflight(model, profile, root)
            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["writes"])
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))
            (model / "images.bin").write_bytes(b"changed")
            self.assertNotEqual(result["model_files_sha256"]["images.bin"],
                                preflight(model, profile, root)["model_files_sha256"]["images.bin"])
