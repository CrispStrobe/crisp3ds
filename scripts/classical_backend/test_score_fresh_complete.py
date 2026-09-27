"""Fail-closed checks for post hoc fresh mesh scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts.classical_backend import score_fresh_complete as target


class FreshScoreTests(unittest.TestCase):
    def test_incomplete_producer_rejected_before_oracles(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "result.json").write_text(json.dumps({
                "schema": target.producer.SCHEMA, "status": "failed",
                "sources_unchanged": True, "stages": []}))
            with self.assertRaisesRegex(ValueError, "incomplete or unsealed"):
                target.checked_run(run)

    def test_completed_report_requires_gate_and_exact_final_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            source = {key: str(run / key) for key in
                      ("model", "images", "pose_masks", "manifest", "producer", "camera_gate")}
            source.update({"producer_sha256": "a" * 64, "camera_gate_sha256": "b" * 64,
                           "model_sha256": {name: "c" * 64 for name in target.MODEL_FILES}})
            report = {"schema": target.producer.SCHEMA, "status": "complete",
                      "sources_unchanged": True, "quality_accepted": False,
                      "metric_scale_verified": False, "source": source,
                      "stages": [{"name": name, "status": "complete"} for name in target.STAGES],
                      "artifact_sha256": {name: "d" * 64 for name in
                                          ("refined.ply", "mesh.ply", "textured.obj")}}
            (run / "result.json").write_text(json.dumps(report))
            with patch.object(target.producer, "input_binding", side_effect=ValueError("camera gate failed")):
                with self.assertRaisesRegex(ValueError, "camera gate failed"):
                    target.checked_run(run)
            with patch.object(target.producer, "input_binding", return_value={}), \
                    patch.object(target, "model_hashes", return_value=source["model_sha256"]):
                with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                    target.checked_run(run)

    def test_camera_only_transport_recovers_proper_similarity(self):
        names = sorted(target.producer.EXPECTED_NAMES)
        theta = np.deg2rad(31)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        candidate, parent = {}, {}
        for index, name in enumerate(names):
            angle = 2 * np.pi * index / len(names)
            center = np.array([np.cos(angle), np.sin(angle), .1 * np.sin(3 * angle)])
            candidate[name] = (center, np.eye(3))
            parent[name] = (1.7 * rotation @ center + [2, -3, .5], rotation.T)
        with patch.object(target.colmap_camera_compare, "read_candidate", side_effect=[
            (candidate, {"candidate": "hash"}, {}), (parent, target.PARENT_MODEL_SHA256, {})]):
            result = target.transported_gauge(Path("new"), Path("006"), {"candidate": "hash"})
        self.assertEqual(result["status"], "available")
        matrix = np.asarray(result["candidate_to_parent_matrix"])
        np.testing.assert_allclose(matrix[:3, :3], 1.7 * rotation, atol=1e-12)
        np.testing.assert_allclose(matrix[:3, 3], [2, -3, .5], atol=1e-12)
        self.assertLess(result["camera_comparison"]["center_rms_over_reference_radius"], 1e-12)

    def test_camera_transport_unavailable_for_missing_or_bad_cameras(self):
        names = sorted(target.producer.EXPECTED_NAMES)
        candidate = {name: (np.array([np.cos(i), np.sin(i), 0.]), np.eye(3))
                     for i, name in enumerate(names)}
        parent = {name: (value[0].copy(), np.eye(3)) for name, value in candidate.items()}
        with patch.object(target.colmap_camera_compare, "read_candidate", side_effect=[
            (dict(list(candidate.items())[1:]), {"candidate": "hash"}, {}),
            (parent, target.PARENT_MODEL_SHA256, {})]):
            result = target.transported_gauge(Path("new"), Path("006"), {"candidate": "hash"})
        self.assertEqual(result["status"], "unavailable")
        candidate[names[0]] = (candidate[names[0]][0] + [1, 0, 0], np.eye(3))
        with patch.object(target.colmap_camera_compare, "read_candidate", side_effect=[
            (candidate, {"candidate": "hash"}, {}), (parent, target.PARENT_MODEL_SHA256, {})]):
            result = target.transported_gauge(Path("new"), Path("006"), {"candidate": "hash"})
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("5% radius", result["reason"])

    def test_untextured_three_view_preview_is_deterministic_and_hashed(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        geometry = (vertices, faces)
        original = vertices.copy()
        with tempfile.TemporaryDirectory() as temp:
            first = target.save_geometry_preview(geometry, geometry, np.eye(4),
                                                 Path(temp) / "one.png", count=64)
            second = target.save_geometry_preview(geometry, geometry, np.eye(4),
                                                  Path(temp) / "two.png", count=64)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["sha256"], target.digest(Path(first["path"])))
            self.assertEqual(first["views"], ["XY", "XZ", "YZ"])
            np.testing.assert_array_equal(vertices, original)


if __name__ == "__main__":
    unittest.main()
