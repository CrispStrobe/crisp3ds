import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import fresh_masked_complete as target


class FreshMaskedCompleteTests(unittest.TestCase):
    def test_full_resolution_two_generation_budget(self):
        sizes = {name: (1282, 1024) for name in target.EXPECTED_NAMES}
        profile = target.depth_preflight(sizes, 120 << 20)
        self.assertEqual(len(profile["images"]), 60)
        self.assertEqual({row["actual_level"] for row in profile["images"].values()}, {0})
        self.assertLess(profile["predicted_two_generation_peak_bytes"], target.CAP)
        with self.assertRaisesRegex(ValueError, "peak exceeds"):
            target.depth_preflight(sizes, 2 << 30)

    def test_sparse_and_camera_reports_are_required_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ("model", "images", "masks"):
                (folder / name).mkdir()
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"images": []}))
            producer = folder / "turntable-profile.json"
            gate = folder / "rig-report.json"
            producer.write_text(json.dumps({"schema": "turntable_sparse_profile_v1",
                                            "status": {"status": "sparse_qa_failed"}}))
            gate.write_text(json.dumps({"schema": "ycb_stock_berkeley_camera_diagnostic_v1"}))
            with self.assertRaisesRegex(ValueError, "fresh sparse profile"):
                target.input_binding(folder / "model", folder / "images", folder / "masks",
                                     manifest, producer, gate, target.digest(producer),
                                     target.digest(gate), {})
            producer.write_text(json.dumps({"schema": "turntable_sparse_profile_v1",
                                            "profile": "prior_foreground_60_jpeg_replay",
                                            "independent_camera_gate_passed": None,
                                            "status": {"status": "completed", "registered": 60,
                                                       "inputs_unchanged": True,
                                                       "sparse_coverage_passed": True,
                                                       "model_hashes_verified": True}}))
            gate.write_text(json.dumps({"schema": "unknown"}))
            with self.assertRaisesRegex(ValueError, "independent rig"):
                target.input_binding(folder / "model", folder / "images", folder / "masks",
                                     manifest, producer, gate, target.digest(producer),
                                     target.digest(gate), {})

    def test_failing_rig_camera_gate_blocks_before_undistortion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "sparse-run"
            model = source / "models" / "0"
            model.mkdir(parents=True)
            for name in ("images", "masks"):
                (root / name).mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            expected = {name: "a" * 64 for name in target.MODEL_FILES}
            summary = source / "summary.json"
            summary.write_text(json.dumps({"model_dir": str(model),
                                           "model_files_sha256": expected}))
            producer = source / "turntable-profile.json"
            producer.write_text(json.dumps({
                "schema": "turntable_sparse_profile_v1",
                "profile": "prior_foreground_60_jpeg_replay",
                "independent_camera_gate_passed": None,
                "status": {"status": "completed", "inputs_unchanged": True,
                           "registered": 60, "sparse_coverage_passed": True,
                           "model_hashes_verified": True,
                           "model_files_sha256": expected,
                           "summary_sha256": target.digest(summary)}}))
            gate = root / "camera-gate.json"
            gate.write_text(json.dumps({
                "schema": "ycb_stock_berkeley_camera_diagnostic_v1",
                "candidate_model_dir": str(model), "candidate_model_sha256": expected,
                "coverage": {"registered": 60, "selected": 60},
                "comparison": {"center_rms_over_reference_radius": 0.06,
                               "orientation_p95_degrees": 9.0}}))
            status = json.loads(producer.read_text())["status"]
            gate_data = json.loads(gate.read_text())
            self.assertEqual(producer.parent, summary.parent)
            self.assertEqual(status["summary_sha256"], target.digest(summary))
            self.assertEqual(status["model_files_sha256"], expected)
            self.assertEqual(json.loads(summary.read_text())["model_files_sha256"], expected)
            self.assertEqual(Path(gate_data["candidate_model_dir"]).resolve(), model)
            self.assertEqual(gate_data["candidate_model_sha256"], expected)
            with patch.object(target, "model_hashes", return_value=expected):
                with self.assertRaisesRegex(ValueError, "5%/10 degree gate"):
                    target.input_binding(model, root / "images", root / "masks", manifest,
                                         producer, gate, target.digest(producer),
                                         target.digest(gate), expected)


if __name__ == "__main__":
    unittest.main()
