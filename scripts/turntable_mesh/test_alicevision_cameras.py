import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.turntable_mesh.alicevision_cameras import (
    audit_scene,
    pixel_intrinsic,
    radial_field_check,
)


def scene():
    intrinsic = {
        "intrinsicId": "1",
        "type": "pinhole",
        "distortionType": "radialk3",
        "width": "100",
        "height": "80",
        "sensorWidth": "36",
        "focalLength": "36",
        "pixelRatio": "1.1",
        "principalPoint": ["2", "-1"],
        "distortionParams": ["0", "0", "0"],
        "locked": "true",
    }
    _, _, p, _ = pixel_intrinsic(intrinsic)
    views, poses = [], []
    for i in range(3):
        views.append(
            {
                "viewId": str(i),
                "poseId": str(i),
                "intrinsicId": "1",
                "path": f"frame{i}.png",
            }
        )
        poses.append(
            {
                "poseId": str(i),
                "pose": {
                    "transform": {
                        "rotation": np.eye(3).reshape(-1, order="F").tolist(),
                        "center": [i * 0.1, 0, 0],
                    }
                },
            }
        )
    points = []
    for i in range(25):
        X = [i * 0.004, 0.01 * (i % 5), 2.0]
        obs = []
        for v in range(3):
            xy = ((np.array(X[:2]) - [v * 0.1, 0]) / X[2]) * p[:2] + p[2:]
            obs.append({"observationId": str(v), "x": xy.tolist()})
        points.append({"landmarkId": str(i), "X": X, "observations": obs})
    return {
        "version": ["1", "2", "14"],
        "views": views,
        "poses": poses,
        "intrinsics": [intrinsic],
        "structure": points,
    }


class AliceVisionCameraTests(unittest.TestCase):
    def audit(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scene.sfm"
            p.write_text(json.dumps(data))
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            out = audit_scene(p, **kwargs)
            self.assertEqual(before, hashlib.sha256(p.read_bytes()).hexdigest())
            return out

    def test_serialized_calibration_and_full_valid_scene(self):
        data = scene()
        w, h, p, k = pixel_intrinsic(data["intrinsics"][0])
        np.testing.assert_allclose(p, [100 / 1.1, 100, 52, 39])
        out = self.audit(
            data,
            expected_names=["frame0.png", "frame1.png", "frame2.png"],
            expected_calibration={"pixels": p.tolist(), "k": k.tolist()},
        )
        self.assertTrue(out["passed"])
        self.assertEqual(out["observations"], 75)
        self.assertEqual(out["positive_depth_fraction"], 1)
        self.assertLess(out["reprojection_pixels"]["max"], 1e-10)
        self.assertFalse(out["shape_accuracy_claim"])

    def test_analytic_fold_inside_field_rejected_not_missed_by_sparse_observations(
        self,
    ):
        # g'(r)=1-3r²: positive at all central observations, but first
        # fold cannot cover a large sensor's corners. Vertex/ray samples alone
        # would miss it; the analytic branch bound rejects the whole field.
        out = radial_field_check(1000, 1000, [500, 500, 500, 500], [-1, 0, 0])
        self.assertFalse(out["passed"])
        self.assertAlmostEqual(out["first_radial_derivative_zero"], 1 / np.sqrt(3))
        self.assertTrue(
            radial_field_check(100, 100, [500, 500, 50, 50], [-1, 0, 0])["passed"]
        )
        self.assertTrue(
            radial_field_check(100, 100, [100, 100, 50, 50], [0, 0.1, 0])["passed"]
        )

    def test_changed_fixed_lens_coverage_and_per_view_error_fail(self):
        data = scene()
        _, _, p, k = pixel_intrinsic(data["intrinsics"][0])
        out = self.audit(
            data,
            expected_calibration={
                "pixels": (p + [1, 0, 0, 0]).tolist(),
                "k": k.tolist(),
            },
        )
        self.assertFalse(out["passed"])
        missing = copy.deepcopy(data)
        missing["poses"] = missing["poses"][:1]
        missing["structure"] = []
        self.assertIn(
            "insufficient registered camera coverage", self.audit(missing)["reasons"]
        )
        noisy = copy.deepcopy(data)
        for point in noisy["structure"]:
            point["observations"][2]["x"][0] += 5
        self.assertIn(
            "per-view reprojection exceeds declared bound", self.audit(noisy)["reasons"]
        )

    def test_reflection_nonfinite_and_behind_camera_fail_closed(self):
        data = scene()
        reflection = copy.deepcopy(data)
        reflection["poses"][0]["pose"]["transform"]["rotation"][0] = -1
        with self.assertRaisesRegex(ValueError, "proper"):
            self.audit(reflection)
        bad = copy.deepcopy(data)
        bad["intrinsics"][0]["focalLength"] = "nan"
        with self.assertRaises(ValueError):
            self.audit(bad)
        bad = copy.deepcopy(data)
        bad["structure"][0]["X"][2] = -1
        self.assertIn("landmark behind camera", self.audit(bad)["reasons"])
        bad = copy.deepcopy(data)
        bad["structure"][0]["observations"].append(
            bad["structure"][0]["observations"][0]
        )
        with self.assertRaisesRegex(ValueError, "observations"):
            self.audit(bad)

    def test_duplicate_landmarks_cannot_inflate_support(self):
        bad = scene()
        bad["structure"].append(copy.deepcopy(bad["structure"][0]))
        with self.assertRaisesRegex(ValueError, "landmark"):
            self.audit(bad)
        bad = scene()
        bad["structure"][0]["observations"] = bad["structure"][0]["observations"][:1]
        with self.assertRaisesRegex(ValueError, "landmark"):
            self.audit(bad)

    def test_policy_schema_and_inventory_rejections(self):
        for kwargs in [
            {"minimum_coverage": True},
            {"minimum_coverage": 0},
            {"maximum_reprojection_p95": float("inf")},
            {"minimum_observations_per_view": True},
        ]:
            with self.assertRaises(ValueError):
                self.audit(scene(), **kwargs)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.audit(scene(), expected_names=["other.png"])
        bad = scene()
        bad["version"] = ["1", "2", "3"]
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.audit(bad)
        for field, value in [
            ("width", True),
            ("height", "0"),
            ("pixelRatio", "0"),
            ("type", "equidistant"),
        ]:
            row = scene()["intrinsics"][0]
            row[field] = value
            with self.assertRaises(ValueError):
                pixel_intrinsic(row)
