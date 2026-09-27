"""Synthetic-only seals for the GLOBAL SfM camera-pose exporter."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_global_pose_export as export


class GlobalPoseExportTest(unittest.TestCase):
    def test_converter_command_is_pose_only_and_distinct_output(self):
        output = Path("/synthetic/global-export")
        argv = export.command(output)
        self.assertEqual(argv[argv.index("-i") + 1], str(export.MODEL))
        self.assertEqual(argv[argv.index("-o") + 1], str(output / "sfm_camera_poses.json"))
        self.assertEqual(argv[-3:], ["-V", "-I", "-E"])
        self.assertNotIn("-S", argv)
        self.assertNotIn("-C", argv)

    def test_sealed_global_receipt_model_log_full_inventory_and_staged_seals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sparse").mkdir()
            (root / "logs").mkdir()
            (root / "matches").mkdir()
            model = root / "sparse/sfm_data.bin"
            log = root / "logs/01-sfm-global.log"
            model.write_bytes(b"global model")
            log.write_bytes(b"global log")
            names = [f"NP3_{index:03}.jpg" for index in range(48)]
            staged = {f"matches/file_{index:03}.bin": str(index) for index in range(105)}
            inputs = {"source_receipt_sha256": export.prior.FAIL_RECEIPT_SHA,
                      "names": names, "reused_file_sha256": staged}
            binaries = {"binary_sha256": {"openMVG_main_SfM": "sealed"}}
            source = {"reviewed": "source hash"}
            model_sha, log_sha = export.prior.sha(model), export.prior.sha(log)
            inventory = export.core.output_inventory(root)
            receipt = {"schema": "openmvg_mustard_global_sfm_control_v1",
                       "status": "failed_registration_or_sparse_gate",
                       "sfm_report": {"views": 48, "poses": 46, "intrinsics": 1,
                                      "tracks": 69, "residuals": 1375},
                       "stage": {"name": "sfm_global", "status": "completed", "returncode": 0,
                                 "model_sha256": model_sha, "log_sha256": log_sha,
                                 "command": export.global_sfm.command(root)},
                       "inputs": inputs, "binaries": binaries,
                       "reviewed_global_source_sha256": source,
                       "staged_match_sha256": staged, "output_inventory": inventory}
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            with patch.object(export, "RECEIPT_SHA", export.prior.sha(receipt_path)), \
                 patch.object(export, "MODEL_SHA", model_sha), \
                 patch.object(export, "SFM_LOG_SHA", log_sha), \
                 patch.object(export.global_sfm.shared, "verify_inputs", return_value=inputs), \
                 patch.object(export.core, "verify_binaries", return_value=binaries), \
                 patch.object(export.global_sfm, "verify_source", return_value=source), \
                 patch.object(export.global_sfm.shared, "verify_staged_matches") as stage_check:
                checked = export.verify_global_model(root)
                self.assertEqual(checked["staged_match_count"], 105)
                stage_check.assert_called_once_with(root, staged)
                (root / "sparse/extra.bin").write_bytes(b"unexpected")
                with self.assertRaisesRegex(RuntimeError, "run/input contract differs"):
                    export.verify_global_model(root)

    def test_fresh_output_and_11gib_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "already"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                export.preflight(existing)
            with patch.object(export.prior.build.fifth.four.fork.base, "tree_bytes", return_value=0), \
                 patch.object(export.prior.build.fifth.four.fork.base, "free_bytes",
                              side_effect=[export.prior.FLOOR, export.prior.FLOOR]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    export.prior.capacity(root / "new", prospective=True)

    def test_non_macos_native_conversion_fails_closed(self):
        with patch.object(export.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "requires macOS"):
                export.run(Path("/synthetic/uncreated"))


if __name__ == "__main__":
    unittest.main()
