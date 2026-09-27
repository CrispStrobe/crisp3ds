import json
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import openmvg_mustard_fixed_pose_diagnostic as check
from scripts.classical_backend import openmvg_mustard_json_pose_validator as prior
from scripts.classical_backend import openmvg_sceaux_postcheck as shared
from scripts.classical_backend.test_openmvg_mustard_json_pose_validator import fixture


class FixedPoseDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.json_path = self.root / "sfm_camera_poses.json"
        self.receipt_path = self.root / "receipt.json"
        data = fixture()
        data["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["disto_k1"] = [0.0]
        self.json_path.write_text(json.dumps(data))
        self.write_receipt()

    def write_receipt(self, *, model_sha=check.MODEL_SHA256):
        missing = ["NP3_042.jpg", "NP3_048.jpg"]
        receipt = {"schema": "openmvg_mustard_fixed_pose_export_v1",
                   "status": "exported_fixed_model_diagnostic_only",
                   "output": str(self.root), "returncode": 0,
                   "json_sha256": shared.sha256(self.json_path),
                   "json_bytes": self.json_path.stat().st_size,
                   "command": [str(check.CONVERTER), "-i", str(check.SOURCE_MODEL),
                               "-o", str(self.json_path), "-V", "-I", "-E"],
                   "converter": {"binary_sha256": prior.CONVERTER_BINARY_SHA256},
                   "fixed_model": {"model_sha256": model_sha,
                                   "receipt_sha256": check.SOURCE_RECEIPT_SHA256,
                                   "sfm_log_sha256": check.SFM_LOG_SHA256,
                                   "status": "failed_registration_or_sparse_gate",
                                   "sfm_report": check.REPORT_COUNTS,
                                   "names": list(prior.TRAIN_NAMES)},
                   "exported": {"views": 48, "poses": 46, "intrinsics": 1,
                                "structure": 0, "control_points": 0,
                                "missing_names": missing,
                                "posed_names": sorted(set(prior.TRAIN_NAMES) - set(missing))}}
        self.receipt_path.write_text(json.dumps(receipt))

    def diagnose(self):
        return check.diagnose(self.root, expected_root=self.root,
                              receipt_pin=shared.sha256(self.receipt_path),
                              json_pin=shared.sha256(self.json_path))

    def test_sealed_fixed_intrinsic_named_orbit(self):
        result = self.diagnose()
        self.assertEqual(result["status"], "failed_48_of_48_gate")
        self.assertEqual(result["pose_count"], 46)
        self.assertEqual(result["missing_pose_names"], ["NP3_042.jpg", "NP3_048.jpg"])
        self.assertEqual(result["intrinsics"][0]["focal_length"], 1536.0)
        self.assertEqual(result["orbit"]["status"], "mapped_by_cereal_view_pose_ids")

    def test_sha_and_source_model_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed SHA-256"):
            check.diagnose(self.root, expected_root=self.root,
                           receipt_pin="0" * 64, json_pin=shared.sha256(self.json_path))
        self.write_receipt(model_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "sealed fixed model"):
            self.diagnose()

    def test_signed_pose_and_intrinsic_tampering_rejected(self):
        value = json.loads(self.json_path.read_text())
        value["extrinsics"][0]["value"]["center"][0] = float("nan")
        self.json_path.write_text(json.dumps(value))
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.diagnose()
        value = fixture()
        value["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["disto_k1"] = [0.0]
        value["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = 1535.0
        self.json_path.write_text(json.dumps(value))
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "frozen image-only heuristic"):
            self.diagnose()

    def test_signed_view_pose_link_tampering_rejected(self):
        value = json.loads(self.json_path.read_text())
        value["views"][0]["value"]["ptr_wrapper"]["data"]["id_pose"] = 999
        self.json_path.write_text(json.dumps(value))
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "pose links"):
            self.diagnose()


if __name__ == "__main__":
    unittest.main()
