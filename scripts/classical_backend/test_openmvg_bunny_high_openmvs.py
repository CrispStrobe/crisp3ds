"""Synthetic gate and bounded command checks; no native reconstruction runs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import openmvg_bunny_high_openmvs as continuation


class OpenMVGBunnyHighOpenMVSTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        (self.source / "sparse").mkdir()
        (self.source / "images").mkdir()
        model = self.source / "sparse" / "sfm_data.bin"
        model.write_bytes(b"synthetic sparse model")
        model_hash = continuation.sha256(model)
        self.images = self.base / "images"
        self.images.mkdir()
        mapping = {}
        photo_hashes = {}
        rows = []
        for i in range(73):
            name = f"frame_{i:04}.png"
            original = f"bunny_{(i + 7) % 73}_rgb.png"
            (self.images / name).write_bytes(f"synthetic photo {i}".encode())
            (self.source / "images" / name).write_bytes(f"synthetic photo {i}".encode())
            photo_hashes[name] = continuation.sha256(self.images / name)
            mapping[name] = original
            rows.append({"output": name, "source": original})
        manifest = self.images / "prepare-manifest.json"
        manifest.write_text(json.dumps({"images": rows}))
        self.manifest_hash = continuation.sha256(manifest)
        producer = {
            "schema": continuation.SOURCE_SCHEMA,
            "status": "completed_pending_geometry_review",
            "source_images": str(self.images),
            "photos": {"photo_sha256": photo_hashes,
                       "prepare_manifest_sha256": self.manifest_hash,
                       "source_frame_by_photo": mapping},
            "staged_photo_sha256": photo_hashes,
            "stages": [{"name": "sfm", "status": "completed", "artifact_sha256": model_hash}],
            "output_inventory": {"sparse/sfm_data.bin": {"sha256": model_hash}},
        }
        self.producer_path = self.source / "receipt.json"
        self.producer_path.write_text(json.dumps(producer))
        self.gate = {
            "schema": continuation.GATE_SCHEMA,
            "decision": "approved_for_openmvs", "camera_geometry_reviewed": True,
            "object_region_consistent": True, "reviewer": "independent synthetic reviewer",
            "source_receipt_sha256": continuation.sha256(self.producer_path),
            "sparse_model_sha256": model_hash,
            "prepare_manifest_sha256": self.manifest_hash,
        }
        self.gate_path = self.base / "gate.json"
        self.gate_path.write_text(json.dumps(self.gate))
        self.bin_dir = self.base / "bin"
        self.bin_dir.mkdir()
        for name in continuation.TOOLS:
            binary = self.bin_dir / name
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
        self.args = argparse.Namespace(
            source_receipt=self.producer_path, camera_gate=self.gate_path,
            camera_gate_sha256=continuation.sha256(self.gate_path),
            expected_images=self.images,
            openmvg_binary=self.bin_dir / continuation.TOOLS[0],
            openmvs_binary_dir=self.bin_dir, output=self.base / "out")

    def test_reviewed_exact_sparse_plan(self):
        bound = continuation.validate(self.args)
        self.assertEqual(bound["source_frame_by_photo"]["frame_0000.png"], "bunny_7_rgb.png")
        stages = continuation.commands(self.args.output, bound)
        self.assertEqual([row[0] for row in stages], ["convert", "densify", "mesh", "refine", "texture"])
        self.assertIn("--max-threads", stages[1][1])
        self.assertIn("2", stages[1][1])
        self.assertFalse(self.args.output.exists())

    def test_gate_hash_mismatch_abstains_before_output(self):
        self.args.camera_gate_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            continuation.validate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_registration_alone_does_not_authorize_dense(self):
        self.gate["decision"] = "pending"
        self.gate_path.write_text(json.dumps(self.gate))
        self.args.camera_gate_sha256 = continuation.sha256(self.gate_path)
        with self.assertRaisesRegex(ValueError, "does not approve"):
            continuation.validate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_changed_numeric_source_mapping_abstains(self):
        producer = json.loads(self.producer_path.read_text())
        producer["photos"]["source_frame_by_photo"]["frame_0000.png"] = "bunny_0_rgb.png"
        self.producer_path.write_text(json.dumps(producer))
        with self.assertRaisesRegex(ValueError, "mapping"):
            continuation.validate(self.args)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
