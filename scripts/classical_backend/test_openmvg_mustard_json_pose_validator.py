import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import openmvg_mustard_json_pose_validator as validator
from scripts.classical_backend import openmvg_sceaux_postcheck as shared


NAMES = tuple(f"NP3_{angle:03d}.jpg" for angle in range(0, 360, 6)
              if angle % 30 != 24)
MISSING = {"NP3_042.jpg", "NP3_048.jpg"}
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def fixture():
    views, extrinsics = [], []
    for view_id, name in enumerate(reversed(NAMES)):
        angle = math.radians(int(name[4:7]))
        views.append({"key": view_id, "value": {"ptr_wrapper": {"data": {
            "local_path": "", "filename": name, "width": 1280, "height": 1024,
            "id_view": view_id, "id_intrinsic": 0, "id_pose": view_id}}}})
        if name not in MISSING:
            extrinsics.append({"key": view_id, "value": {
                "rotation": copy.deepcopy(IDENTITY),
                "center": [math.cos(angle), math.sin(angle), 0.2]}})
    return {"sfm_data_version": "0.3", "root_path": "./images", "views": views,
            "intrinsics": [{"key": 0, "value": {"polymorphic_name": "pinhole_radial_k1",
                "ptr_wrapper": {"data": {"width": 1280, "height": 1024,
                                         "focal_length": 1536.0,
                                         "principal_point": [640.0, 512.0],
                                         "disto_k1": 0.01}}}}],
            "extrinsics": extrinsics, "structure": [], "control_points": []}


class PayloadTests(unittest.TestCase):
    def test_exact_named_46_poses_and_orbit(self):
        result = validator.parse_export(fixture(), NAMES)
        self.assertEqual(result["status"], "failed_48_of_48_gate")
        self.assertEqual(result["missing_pose_names"], sorted(MISSING))
        self.assertEqual(result["orbit"]["status"], "mapped_by_cereal_view_pose_ids")
        self.assertAlmostEqual(result["orbit"]["six_degree_chord_over_opposing"],
                               math.sin(math.radians(3)), places=6)
        self.assertIn("explicit cereal", result["orbit"]["mapping"])

    def test_duplicate_name_and_broken_pose_link_rejected(self):
        value = fixture()
        value["views"][0]["value"]["ptr_wrapper"]["data"]["filename"] = value["views"][1]["value"]["ptr_wrapper"]["data"]["filename"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validator.parse_export(value, NAMES)
        value = fixture()
        value["extrinsics"][0]["key"] = 999
        with self.assertRaisesRegex(ValueError, "pose links"):
            validator.parse_export(value, NAMES)

    def test_nonfinite_center_or_intrinsic_and_bad_rotation_rejected(self):
        value = fixture()
        value["extrinsics"][0]["value"]["center"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validator.parse_export(value, NAMES)
        value = fixture()
        value["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = math.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validator.parse_export(value, NAMES)
        value = fixture()
        value["extrinsics"][0]["value"]["rotation"][0] = [2., 0., 0.]
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validator.parse_export(value, NAMES)
        value = fixture()
        value["extrinsics"][0]["value"]["rotation"][2] = [0., 0., -1.]
        with self.assertRaisesRegex(ValueError, "determinant"):
            validator.parse_export(value, NAMES)

    def test_wrong_train_inventory_and_export_scope_rejected(self):
        with self.assertRaisesRegex(ValueError, "TRAIN name inventory"):
            validator.parse_export(fixture(), NAMES[:-1] + ("NP3_024.jpg",))
        value = fixture()
        value["views"][0]["value"]["ptr_wrapper"]["data"]["filename"] = "NP5_000.jpg"
        with self.assertRaisesRegex(ValueError, "unsealed view"):
            validator.parse_export(value, NAMES)
        value = fixture()
        value["structure"] = [{"key": 1}]
        with self.assertRaisesRegex(ValueError, "only -V -I -E"):
            validator.parse_export(value, NAMES)


class SealTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.export = self.root / "sfm_camera_poses.json"
        self.receipt = self.root / "receipt.json"
        self.export.write_text(json.dumps(fixture()))
        self.write_receipt()

    def write_receipt(self):
        command = [str(validator.EXPORT_ROOT.parent / "openmvg-v21-ligt-off-oracle-001" /
                       "build/Darwin-arm64-Release/openMVG_main_ConvertSfM_DataFormat"),
                   "-i", str(validator.EXPORT_ROOT.parent /
                              "openmvg-mustard-photo-sfm-001/sparse/sfm_data.bin"),
                   "-o", str(self.export), "-V", "-I", "-E"]
        self.receipt.write_text(json.dumps({
            "schema": "synthetic_export_v1", "status": "exported_failed_model_diagnostic_only",
            "output": str(self.root), "returncode": 0, "json_bytes": self.export.stat().st_size,
            "json_sha256": shared.sha256(self.export), "command": command,
            "converter": {"binary_sha256": validator.CONVERTER_BINARY_SHA256},
            "failed_model": {"sfm_data_bin_sha256": validator.SOURCE_MODEL_SHA256,
                             "receipt_sha256": validator.SOURCE_RECEIPT_SHA256,
                             "status": "failed_registration_or_sparse_gate",
                             "sfm_report": {"views": 48, "poses": 46, "intrinsics": 1,
                                            "tracks": 222, "residuals": 1497}},
            "exported": {"views": 48, "poses": 46, "intrinsics": 1,
                         "structure": 0, "control_points": 0,
                         "missing_names": sorted(MISSING),
                         "posed_names": sorted(set(NAMES) - MISSING)}}))

    def test_unset_pin_abstains_without_reading_export(self):
        result = validator.audit(self.root / "absent.json", self.root / "absent-receipt.json",
                                 receipt_pin=None, receipt_schema=None)
        self.assertEqual(result["status"], "abstained_unpinned_export")

    def test_pinned_synthetic_receipt_binds_payload(self):
        digest = shared.sha256(self.receipt)
        result = validator.audit(self.export, self.receipt, receipt_pin=digest,
                                 receipt_schema="synthetic_export_v1",
                                 expected_json_hash=shared.sha256(self.export),
                                 expected_json=self.export, expected_receipt=self.receipt)
        self.assertEqual(result["pose_count"], 46)
        self.assertEqual(result["export_receipt_sha256"], digest)
        self.export.write_text(self.export.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validator.audit(self.export, self.receipt, receipt_pin=digest,
                            receipt_schema="synthetic_export_v1",
                            expected_json_hash=shared.sha256(self.export)[:-1] + "0",
                            expected_json=self.export, expected_receipt=self.receipt)

    def test_duplicate_json_key_rejected(self):
        self.export.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            validator.load_json(self.export)


if __name__ == "__main__":
    unittest.main()
