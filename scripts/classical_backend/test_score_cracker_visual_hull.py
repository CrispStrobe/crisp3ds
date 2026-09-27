"""Synthetic-only checks of frozen score protocol and occupancy reconstruction."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.classical_backend import score_cracker_visual_hull as score
from scripts.classical_backend.visual_hull_core import exposed_cube_mesh


class HullScoreSyntheticTest(unittest.TestCase):
    def test_baseline_replay_fails_closed(self):
        rows = [{"threshold": 1.0, "precision": 0.25, "recall": 0.5, "f_score": 1 / 3},
                {"threshold": 2.0, "precision": 0.5, "recall": 0.5, "f_score": 0.5},
                {"threshold": 4.0, "precision": 1.0, "recall": 0.75, "f_score": 6 / 7}]
        score.assert_baseline_replay({"threshold_scores": rows}, {"threshold_scores": rows}, "rough")
        changed = [dict(row) for row in rows]
        changed[1]["f_score"] += 0.001
        with self.assertRaisesRegex(ValueError, "baseline failed deterministic replay"):
            score.assert_baseline_replay({"threshold_scores": changed}, {"threshold_scores": rows}, "rough")

    def test_exact_voxel_reconstruction_with_enclosed_void(self):
        occupancy = np.zeros((7, 7, 7), bool)
        occupancy[1:6, 1:6, 1:6] = True
        occupancy[3, 3, 3] = False
        vertices, faces = exposed_cube_mesh(occupancy, np.zeros(3), 7)
        recovered = score.reconstruct_occupancy_from_exposed_faces(
            vertices, faces, np.zeros(3), 7, 7, int(occupancy.sum()))
        np.testing.assert_array_equal(recovered, occupancy)
        self.assertAlmostEqual(score.signed_mesh_volume(vertices, faces), float(occupancy.sum()))
        reversed_faces = faces[:, ::-1]
        np.testing.assert_array_equal(score.reconstruct_occupancy_from_exposed_faces(
            vertices, reversed_faces, np.zeros(3), 7, 7, int(occupancy.sum())), occupancy)
        self.assertAlmostEqual(score.signed_mesh_volume(vertices, reversed_faces), float(occupancy.sum()))
        with self.assertRaisesRegex(ValueError, "sealed count"):
            score.reconstruct_occupancy_from_exposed_faces(
                vertices, faces, np.zeros(3), 7, 7, int(occupancy.sum()) + 1)

    def test_frozen_reference_diagonal_thresholds(self):
        vertices = np.asarray([[0, 0, 0], [3, 4, 0]], dtype=float)
        prior = {"shared_gauge": {"comparison": {"reference_bbox_diagonal": 5.0}}}
        stage = {"thresholds": [0.025, 0.05, 0.1]}
        self.assertEqual(score.thresholds_from_reference(vertices, prior, stage), stage["thresholds"])
        with self.assertRaisesRegex(ValueError, "thresholds differ"):
            score.thresholds_from_reference(vertices, prior, {"thresholds": [0.025, 0.05, 0.11]})

    def test_bad_sealed_source_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            with patch.object(score, "OUTPUT", output), patch.object(score, "DATA", output.parent), \
                    patch.object(score, "disk_floor", return_value={}), \
                    patch.object(score, "verify_inputs", side_effect=ValueError("sealed input differs")), \
                    patch.object(sys, "argv", ["score", "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "sealed input differs"):
                    score.main()
            self.assertFalse(output.exists())

    def test_exact_sealed_preflight_and_mutated_mesh_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hull, producer = root / "hull", root / "producer"
            hull.mkdir()
            producer.mkdir()
            reference = root / "reference.ply"
            old_score_path, old_stage_path = root / "old-score.json", root / "old-stage.json"
            mesh, rough, refined = hull / "hull.ply", producer / "mesh.ply", producer / "refined.ply"
            for path, payload in ((mesh, b"candidate"), (rough, b"rough"),
                                  (refined, b"refined"), (reference, b"reference")):
                path.write_bytes(payload)
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            old_score = {"shared_gauge": {"status": "available",
                        "matrix_used_output_to_reference": np.eye(4).tolist()},
                         "threshold_fractions_of_reference_diagonal": list(score.FRACTIONS),
                         "reference_sha256": sha(reference)}
            old_score_path.write_text(json.dumps(old_score))
            old_stage_path.write_text(json.dumps({"source_sha256": {"score": sha(old_score_path)}}))
            hull_report = {"status": "complete", "neutral_mesh": {"sha256": sha(mesh)},
                           "boundary_touched": False, "occupied_voxels": 55815}
            report_path = hull / "hull-report.json"
            report_path.write_text(json.dumps(hull_report))
            result_path = hull / "result.json"
            result_path.write_text(json.dumps({"status": "complete", "hull_report_sha256": sha(report_path)}))
            hashes = {"hull_result": sha(result_path), "hull_report": sha(report_path),
                      "hull_mesh": sha(mesh), "old_score": sha(old_score_path),
                      "old_stage": sha(old_stage_path), "rough_native": sha(rough),
                      "refined_native": sha(refined), "reference": sha(reference)}
            with patch.object(score, "HULL", hull), patch.object(score, "PRODUCER", producer), \
                    patch.object(score, "REFERENCE", reference), patch.object(score, "OLD_SCORE", old_score_path), \
                    patch.object(score, "OLD_STAGE", old_stage_path), patch.object(score, "HASHES", hashes):
                self.assertEqual(score.verify_inputs()[0]["occupied_voxels"], 55815)
                mesh.write_bytes(b"mutated candidate")
                with self.assertRaisesRegex(ValueError, "sealed geometry differs"):
                    score.verify_inputs()


if __name__ == "__main__":
    unittest.main()
