"""Synthetic sealed-lineage tests; no ignored datasets or native mapper required."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.object_dataset import sensor_depth as sensor
from scripts.object_motion import recovered_camera_reference as recovered


class RecoveredCameraReferenceTests(unittest.TestCase):
    def setUp(self):
        scratch = recovered.ROOT / ".local-tools/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.four_path = self.root / "004/report.json"
        self.five_path = self.root / "005/report.json"
        self.prepared_path = self.root / "prepare.json"
        self.photo_path = self.root / "photos.json"
        self.oracle_path = self.root / "oracle.json"
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        self.four_model = self.four_path.parent / "models/0"
        self.copy_model = self.five_path.parent / "input_model_copy"
        self.five_model = self.five_path.parent / "refined_model"
        hashes = {}
        for folder in (self.four_model, self.copy_model, self.five_model):
            folder.mkdir(parents=True)
            hashes[folder] = {}
            for name in recovered.MODEL_NAMES:
                path = folder / name
                path.write_bytes((str(folder == self.five_model) + name).encode())
                hashes[folder][name] = recovered.base.sha256(path)
        photos = {f"NP3_{angle:03}.jpg": f"{angle:064x}" for angle in recovered.base.ANGLES}
        selected = {"sources": {"berkeley_rgbd": {"sha256": recovered.base.ARCHIVE_SHA256}},
                    "photos": [{"path": f"photos/{name}", "sha256": digest}
                               for name, digest in photos.items()]}
        self.photo_path.write_text(json.dumps(selected))
        self.prepared_path.write_text(json.dumps({
            "source_manifest_sha256": recovered.base.sha256(self.photo_path),
            "images": [{"name": name, "sha256": digest} for name, digest in photos.items()]}))
        members = {}
        for member in recovered.base.MEMBERS:
            path = self.metadata / member.removeprefix(recovered.base.PREFIX)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(member.encode())
            members[member] = {"sha256": recovered.base.sha256(path)}
        self.oracle = {"schema": "ycb_berkeley_camera_oracle_v1",
                       "source_archive_sha256": recovered.base.ARCHIVE_SHA256,
                       "metadata": {"members": members}}
        self.oracle_path.write_text(json.dumps(self.oracle))
        self.four = {"schema": "ycb_image_only_fixed_initial_intrinsics_v1", "status": "complete",
                     "registered": 60, "model_dir": str(self.four_model),
                     "model_files_sha256": hashes[self.four_model],
                     "source_database_sha256": "d" * 64,
                     "cache_mutation_audit_sha256": "a" * 64,
                     "seed_pair": ["NP3_018.jpg", "NP3_030.jpg"],
                     "source_image_hashes": photos,
                     "source_manifest_sha256": recovered.base.sha256(self.prepared_path)}
        self.four_path.write_text(json.dumps(self.four))
        self.five = {"schema": "ycb_image_only_delayed_self_calibration_v1", "status": "complete",
                     "registered": 60, "source_producer_report_sha256": recovered.base.sha256(self.four_path),
                     "source_model_files_sha256": hashes[self.four_model],
                     "input_model_copy": str(self.copy_model),
                     "input_model_copy_sha256": hashes[self.copy_model],
                     "refined_model_dir": str(self.five_model),
                     "refined_model_files_sha256": hashes[self.five_model],
                     "source_database_sha256": self.four["source_database_sha256"],
                     "source_cache_mutation_audit_sha256": self.four["cache_mutation_audit_sha256"],
                     "seed_pair": self.four["seed_pair"], "after_camera_params": [1104, 640, 512, -.05],
                     "supplied_intrinsics_or_poses_used": False, "reference_mesh_used": False}
        self.five_path.write_text(json.dumps(self.five))

    def _patch_paths(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, value in (("PRODUCER_005", self.five_path), ("PRODUCER_004", self.four_path),
                              ("PRODUCER_005_SHA256", recovered.base.sha256(self.five_path)),
                              ("PRODUCER_004_SHA256", recovered.base.sha256(self.four_path)),
                              ("PREPARED", self.prepared_path), ("ORACLE", self.oracle_path),
                              ("METADATA", self.metadata)):
            stack.enter_context(patch.object(recovered, target, value))
        stack.enter_context(patch.object(recovered.base, "PHOTO_MANIFEST", self.photo_path))
        stack.enter_context(patch.object(sensor, "REFERENCE_003", self.oracle_path))
        stack.enter_context(patch.object(sensor, "CALIBRATION", self.metadata / "calibration.h5"))
        return stack

    def test_exact_chain_and_tamper_rejection(self):
        self._patch_paths()
        five, four, oracle, photos = recovered.verified_inputs(
            self.five_path, self.four_path, self.oracle_path, self.metadata,
            self.photo_path, self.prepared_path)
        self.assertEqual(len(photos), 60)
        self.assertEqual(five["source_producer_report_sha256"], recovered.PRODUCER_004_SHA256)
        model_file = self.five_model / "images.bin"
        model_file.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "model changed"):
            recovered.verified_inputs(self.five_path, self.four_path, self.oracle_path,
                                      self.metadata, self.photo_path, self.prepared_path)

    def test_report_sensor_verifier_and_photo_lineage(self):
        self._patch_paths()
        centers = {f"NP3_{angle:03}.jpg": np.array([np.cos(np.deg2rad(angle)),
                 np.sin(np.deg2rad(angle)), angle / 360]) for angle in recovered.base.ANGLES}
        rotations = {name: np.eye(3) for name in centers}
        camera = type("Camera", (), {"model": type("Model", (), {"name": "SIMPLE_RADIAL"})(),
                                      "width": 1280, "height": 1024,
                                      "params": [1104, 640, 512, -.05]})()
        output = self.root / "camera_report"
        with patch.object(recovered.base, "estimated_cameras", return_value=(centers, rotations, camera)), \
             patch.object(recovered.base, "reference_cameras", return_value=(
                 centers, rotations, np.eye(3), np.zeros(5))):
            report = recovered.evaluate(output, self.five_path, self.four_path,
                                        self.oracle_path, self.metadata, self.photo_path,
                                        self.prepared_path)
        self.assertLess(report["camera_centers"]["fit_rms"], 1e-12)
        checked, matrix = sensor._verified_camera_report(output / "report.json",
                                                         self.five_model, self.oracle)
        np.testing.assert_allclose(matrix, np.eye(4), atol=1e-12)
        self.assertEqual(checked["producer_005_report_sha256"], recovered.PRODUCER_005_SHA256)
        dense = self.root / "dense/sparse"
        dense.mkdir(parents=True)
        dense_hashes = {}
        for name in recovered.MODEL_NAMES:
            path = dense / name
            path.write_bytes(("undistorted-" + name).encode())
            dense_hashes[name] = recovered.base.sha256(path)
        native = self.root / "result.json"
        native.write_text(json.dumps({
            "schema": "classical_recovered_dense_v1", "status": "complete",
            "source_model": str(self.five_model),
            "source_model_sha256": checked["producer_model_files_sha256"],
            "source_reports": {"producer": {"sha256": recovered.PRODUCER_005_SHA256},
                               "source": {"sha256": recovered.PRODUCER_004_SHA256},
                               "manifest": {"sha256": checked["prepared_manifest_sha256"]}},
            "stages": [{"name": "undistort", "status": "complete",
                        "artifact": {"model_sha256": dense_hashes}}]}))
        self.assertEqual(sensor.verify_recovered_dense_lineage(
            native, self.five_model, dense, checked)["undistorted_model_sha256"], dense_hashes)
        (dense / "cameras.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "sparse model changed"):
            sensor.verify_recovered_dense_lineage(native, self.five_model, dense, checked)
        checked["source_image_hashes"]["NP3_000.jpg"] = "f" * 64
        (output / "report.json").write_text(json.dumps(checked))
        with self.assertRaisesRegex(ValueError, "lineage"):
            sensor._verified_camera_report(output / "report.json", self.five_model, self.oracle)


if __name__ == "__main__":
    unittest.main()
