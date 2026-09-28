"""Synthetic, no-native-work checks for the gated HIGH bunny finish handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_bunny_high_finish as finish
from scripts.classical_backend.run import digest


class FinishPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "rough"
        images = self.source / "images"
        images.mkdir(parents=True)
        inventory = {}
        for name in sorted(finish.NAMES):
            path = images / name
            path.write_bytes(b"synthetic photo " + name.encode())
            inventory[f"images/{name}"] = {"bytes": path.stat().st_size,
                                            "sha256": digest(path)}
        artifacts = {}
        for name in ("dense.mvs", "dense.ply", "mesh.mvs", "mesh.ply"):
            path = self.source / name
            path.write_bytes(("synthetic " + name).encode())
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
            inventory[name] = artifacts[name].copy()
        self.receipt = {
            "schema": finish.ROUGH_SCHEMA,
            "status": "rough_complete_pending_quality_review",
            "source_unchanged": True, "quality_accepted": False,
            "stages": [{"name": stage, "status": "complete", "returncode": 0,
                        "cache_fusion_verified": True if stage == "densify" else None,
                        "command": ["DensifyPointCloud", "--geometric-iters", "0",
                                    "--resolution-level", "3"] if stage == "densify" else ["ReconstructMesh"],
                        "artifact": {"path": str((self.source / name).resolve()),
                                     **artifacts[name]}}
                       for stage, name in (("densify", "dense.ply"), ("mesh", "mesh.ply"))],
            "artifacts": artifacts, "output_inventory": inventory,
        }
        self.receipt_path = self.source / "result.json"
        self._save_receipt()
        self.review_path = self.base / "review.json"
        self.review = {
            "schema": finish.REVIEW_SCHEMA,
            "decision": "approved_for_refine_texture", "native_mesh_valid": True,
            "visual_object_shape_reviewed": True, "reviewer": "synthetic fixture",
            "rough_receipt_sha256": digest(self.receipt_path),
            "dense_mvs_sha256": artifacts["dense.mvs"]["sha256"],
            "mesh_ply_sha256": artifacts["mesh.ply"]["sha256"],
        }
        self._save_review()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        for name in finish.TOOLS:
            path = self.bin / name
            path.write_bytes(b"#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        self.args = argparse.Namespace(source=self.source,
            source_receipt_sha256=digest(self.receipt_path),
            review_receipt=self.review_path, review_receipt_sha256=digest(self.review_path),
            output=self.base / "fresh", binary_dir=self.bin)
        self.ply_patch = patch.object(finish, "checked_ply", return_value=1)
        self.disk_patch = patch.object(finish.shutil, "disk_usage",
                                       return_value=type("Disk", (), {"free": 100 << 30})())
        self.ply_patch.start()
        self.disk_patch.start()
        self.addCleanup(self.ply_patch.stop)
        self.addCleanup(self.disk_patch.stop)

    def _save_receipt(self) -> None:
        self.receipt_path.write_text(json.dumps(self.receipt))

    def _save_review(self) -> None:
        self.review_path.write_text(json.dumps(self.review))

    def test_positive_preflight_is_read_only_and_frozen_commands(self) -> None:
        bound = finish.validate(self.args)
        self.assertFalse(self.args.output.exists())
        commands = finish.commands(self.args.output, bound)
        self.assertEqual([name for name, _, _ in commands], ["refine", "texture"])
        self.assertIn("--resolution-level", commands[0][1])
        self.assertIn("--scales", commands[0][1])
        self.assertIn("--max-threads", commands[1][1])
        self.assertIn(str(self.args.output / "refined.ply"), commands[1][1])

    def test_no_review_approval_abstains(self) -> None:
        self.review["visual_object_shape_reviewed"] = False
        self._save_review()
        self.args.review_receipt_sha256 = digest(self.review_path)
        with self.assertRaisesRegex(ValueError, "native/visual review"):
            finish.validate(self.args)
        self.assertFalse(self.args.output.exists())

    def test_modified_rough_artifact_abstains(self) -> None:
        (self.source / "mesh.ply").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "native artifact differs"):
            finish.validate(self.args)

    def test_failed_cache_resume_schema_is_not_accepted(self) -> None:
        self.receipt["schema"] = "openmvg_bunny_high_cache_rough_v1"
        self._save_receipt()
        self.args.source_receipt_sha256 = digest(self.receipt_path)
        self.review["rough_receipt_sha256"] = self.args.source_receipt_sha256
        self._save_review()
        self.args.review_receipt_sha256 = digest(self.review_path)
        with self.assertRaisesRegex(ValueError, "rough source has not completed"):
            finish.validate(self.args)

    def test_incomplete_stage_abstains(self) -> None:
        self.receipt["stages"][1]["status"] = "failed"
        self._save_receipt()
        self.args.source_receipt_sha256 = digest(self.receipt_path)
        self.review["rough_receipt_sha256"] = self.args.source_receipt_sha256
        self._save_review()
        self.args.review_receipt_sha256 = digest(self.review_path)
        with self.assertRaisesRegex(ValueError, "stage not sealed complete"):
            finish.validate(self.args)

    def test_stage_path_mismatch_abstains(self) -> None:
        self.receipt["stages"][1]["artifact"]["path"] = "/unrelated/mesh.ply"
        self._save_receipt()
        self.args.source_receipt_sha256 = digest(self.receipt_path)
        self.review["rough_receipt_sha256"] = self.args.source_receipt_sha256
        self._save_review()
        self.args.review_receipt_sha256 = digest(self.review_path)
        with self.assertRaisesRegex(ValueError, "does not seal"):
            finish.validate(self.args)

    def test_unverified_or_geometric_fusion_abstains(self) -> None:
        self.receipt["stages"][0]["command"][2] = "2"
        self._save_receipt()
        self.args.source_receipt_sha256 = digest(self.receipt_path)
        self.review["rough_receipt_sha256"] = self.args.source_receipt_sha256
        self._save_review()
        self.args.review_receipt_sha256 = digest(self.review_path)
        with self.assertRaisesRegex(ValueError, "frozen geom0"):
            finish.validate(self.args)

    def test_insufficient_new_output_headroom_abstains(self) -> None:
        with patch.object(finish, "MIN_NEW_OUTPUT_HEADROOM", finish.CAP):
            with self.assertRaisesRegex(ValueError, "less than 96 MiB"):
                finish.validate(self.args)


if __name__ == "__main__":
    unittest.main()
