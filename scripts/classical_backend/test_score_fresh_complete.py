"""Fail-closed checks for post hoc fresh mesh scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
