import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import openmvg_mustard_global_pose_diagnostic as check
from scripts.classical_backend import openmvg_mustard_json_pose_validator as prior
from scripts.classical_backend import openmvg_sceaux_postcheck as shared
from scripts.classical_backend.test_openmvg_mustard_json_pose_validator import fixture


MISSING = {"NP3_252.jpg", "NP3_276.jpg"}


def global_fixture():
    data = fixture()
    name_by_id = {row["key"]: row["value"]["ptr_wrapper"]["data"]["filename"]
                  for row in data["views"]}
    data["extrinsics"] = [row for row in data["extrinsics"]
                          if name_by_id[row["key"]] not in MISSING]
    for view_id, name in name_by_id.items():
        if name in ("NP3_042.jpg", "NP3_048.jpg"):
            angle = math.radians(int(name[4:7]))
            data["extrinsics"].append({"key": view_id, "value": {
                "rotation": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                "center": [math.cos(angle), math.sin(angle), 0.2]}})
    intrinsic = data["intrinsics"][0]["value"]["ptr_wrapper"]["data"]
    intrinsic.update(focal_length=1589.1184933617688,
                     principal_point=[705.9022666974952, 538.8405298944786],
                     disto_k1=[-0.7807688306637307])
    return data


class GlobalDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.json_path = self.root / "sfm_camera_poses.json"
        self.receipt_path = self.root / "receipt.json"
        self.json_path.write_text(json.dumps(global_fixture()))
        self.write_receipt()

    def write_receipt(self, *, model_sha=check.MODEL_SHA256):
        receipt = {"schema": "openmvg_mustard_global_pose_export_v1",
                   "status": "exported_global_model_diagnostic_only",
                   "output": str(self.root), "returncode": 0,
                   "json_sha256": shared.sha256(self.json_path),
                   "json_bytes": self.json_path.stat().st_size,
                   "command": [str(check.CONVERTER), "-i", str(check.SOURCE_MODEL),
                               "-o", str(self.json_path), "-V", "-I", "-E"],
                   "converter": {"binary_sha256": prior.CONVERTER_BINARY_SHA256},
                   "global_model": {"model_sha256": model_sha,
                                    "receipt_sha256": check.SOURCE_RECEIPT_SHA256,
                                    "sfm_log_sha256": check.SFM_LOG_SHA256,
                                    "source_receipt_sha256": prior.SOURCE_RECEIPT_SHA256,
                                    "staged_match_count": 105,
                                    "status": "failed_registration_or_sparse_gate",
                                    "sfm_report": check.REPORT_COUNTS,
                                    "names": list(prior.TRAIN_NAMES)},
                   "exported": {"views": 48, "poses": 46, "intrinsics": 1,
                                "structure": 0, "control_points": 0,
                                "missing_names": sorted(MISSING),
                                "posed_names": sorted(set(prior.TRAIN_NAMES) - MISSING)}}
        self.receipt_path.write_text(json.dumps(receipt))

    def diagnose(self):
        return check.diagnose(self.root, expected_root=self.root,
                              receipt_pin=shared.sha256(self.receipt_path),
                              json_pin=shared.sha256(self.json_path))

    def test_sealed_global_named_orbit(self):
        result = self.diagnose()
        self.assertEqual(result["status"], "failed_48_of_48_gate")
        self.assertEqual(result["missing_pose_names"], sorted(MISSING))
        self.assertEqual(result["orbit"]["status"], "mapped_by_cereal_view_pose_ids")
        self.assertAlmostEqual(result["orbit"]["six_degree_chord_over_opposing"],
                               math.sin(math.radians(3)), places=6)
        self.assertEqual(result["intrinsics"][0]["distortion"]["disto_k1"],
                         (-0.7807688306637307,))

    def test_sha_or_source_model_tamper_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed SHA-256"):
            check.diagnose(self.root, expected_root=self.root,
                           receipt_pin="0" * 64, json_pin=shared.sha256(self.json_path))
        self.write_receipt(model_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "sealed model"):
            self.diagnose()

    def test_signed_named_pose_and_rotation_tamper_rejected(self):
        value = json.loads(self.json_path.read_text())
        value["views"][0]["value"]["ptr_wrapper"]["data"]["id_pose"] = 999
        self.json_path.write_text(json.dumps(value))
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "pose links"):
            self.diagnose()
        value = global_fixture()
        value["extrinsics"][0]["value"]["rotation"][0] = [2., 0., 0.]
        self.json_path.write_text(json.dumps(value))
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.diagnose()

    def test_signed_missing_name_inventory_tamper_rejected(self):
        receipt = json.loads(self.receipt_path.read_text())
        receipt["exported"]["missing_names"] = ["NP3_042.jpg", "NP3_048.jpg"]
        self.receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "named-pose inventory"):
            self.diagnose()


if __name__ == "__main__":
    unittest.main()
