from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.apple_object_capture import launch as target
from scripts.classical_backend.run import RESERVE


class AppleLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "probe"
        self.binary.write_bytes(b"fake compiled probe")
        self.run_dir = self.root / "fresh"
        self.platform_patch = patch.object(target.sys, "platform", "darwin")
        self.platform_patch.start()
        self.addCleanup(self.platform_patch.stop)
        self.disk_patch = patch.object(target.shutil, "disk_usage",
                                       return_value=types.SimpleNamespace(free=RESERVE + (2 << 30)))
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def test_positive_support_report(self):
        def support_stage(run, *args):
            (run / "photogrammetry.log").write_text(
                '{"schema":"apple_photogrammetry_support_v1","supported":true}\n')
            return {"name": "photogrammetry", "status": "complete", "seconds": 0.1}
        with patch.object(target, "stage", side_effect=support_stage):
            report = target.launch(self.binary, self.run_dir, max_output_mib=10)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(json.loads((self.run_dir / "result.json").read_text())["support"]["supported"])
        self.assertFalse(report["cross_platform_backend"])

    def test_unsupported_or_missing_support_fails(self):
        def bad_stage(run, *args):
            (run / "photogrammetry.log").write_text(
                '{"schema":"apple_photogrammetry_support_v1","supported":false}\n')
            return {"name": "photogrammetry", "status": "complete", "seconds": 0.1}
        with patch.object(target, "stage", side_effect=bad_stage):
            report = target.launch(self.binary, self.run_dir, max_output_mib=10)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(self.run_dir.joinpath("model.usdz").exists())

    def test_malformed_support_log_fails(self):
        def malformed_stage(run, *args):
            (run / "photogrammetry.log").write_text("not a support response\n")
            return {"name": "photogrammetry", "status": "complete", "seconds": 0.1}
        with patch.object(target, "stage", side_effect=malformed_stage):
            report = target.launch(self.binary, self.run_dir)
        self.assertEqual(report["status"], "failed")
        self.assertIn("no positive hardware result", report["failure"])
        self.assertEqual(report["limits"]["max_output_bytes"], 100 << 20)

    def test_rgb_mode_is_opt_in_and_hash_bound(self):
        images = self.root / "images"
        images.mkdir()
        for index in range(3):
            (images / f"photo{index}.jpg").write_bytes(bytes([index + 1]))
        def model_stage(run, *args):
            (run / "model.usdz").write_bytes(b"generated USDZ fixture")
            (run / "photogrammetry.log").write_text("PROCESSING_COMPLETE\n")
            return {"name": "photogrammetry", "status": "complete", "seconds": 1.0}
        with patch.object(target, "stage", side_effect=model_stage) as native:
            report = target.launch(self.binary, self.run_dir, images=images, max_output_mib=10)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["inputs"]), 3)
        self.assertEqual(report["artifact"]["bytes"], len(b"generated USDZ fixture"))
        self.assertEqual(native.call_args.args[2][1:5],
                         ["--images", str(images.resolve()), "--output",
                          str((self.run_dir / "model.usdz").resolve())])

    def test_overwrite_disk_and_platform_gates(self):
        self.run_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            target.launch(self.binary, self.run_dir)
        self.run_dir.rmdir()
        with patch.object(target.shutil, "disk_usage", return_value=types.SimpleNamespace(free=RESERVE)):
            with self.assertRaisesRegex(ValueError, "insufficient free space"):
                target.launch(self.binary, self.run_dir)
        with patch.object(target.sys, "platform", "linux"):
            with self.assertRaisesRegex(ValueError, "macOS"):
                target.launch(self.binary, self.run_dir)

    def test_bad_caps_and_symlink_photo_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid output"):
            target.launch(self.binary, self.run_dir, max_output_mib=0)
        images = self.root / "images"
        images.mkdir()
        for index in range(3):
            (images / f"photo{index}.jpg").write_bytes(b"image")
        (images / "photo2.jpg").unlink()
        (images / "photo2.jpg").symlink_to(images / "photo1.jpg")
        with self.assertRaisesRegex(ValueError, "linked"):
            target.launch(self.binary, self.run_dir, images=images)

    def test_rejects_unselected_input_entries(self):
        images = self.root / "images"
        images.mkdir()
        for index in range(3):
            (images / f"photo{index}.jpg").write_bytes(b"image")
        (images / "metadata.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "only supported"):
            target.launch(self.binary, self.run_dir, images=images)

    def test_failure_still_checks_source_hashes(self):
        images = self.root / "images"
        images.mkdir()
        for index in range(3):
            (images / f"photo{index}.jpg").write_bytes(b"image")
        def failing_stage(run, *args):
            (images / "photo1.jpg").write_bytes(b"changed")
            raise RuntimeError("native failed")
        with patch.object(target, "stage", side_effect=failing_stage):
            report = target.launch(self.binary, self.run_dir, images=images, max_output_mib=10)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_images_unchanged"])
        self.assertEqual(report["provenance_failure"], "binary or source images changed")


if __name__ == "__main__":
    unittest.main()
