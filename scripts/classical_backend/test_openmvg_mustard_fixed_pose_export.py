"""Synthetic-only tests for the fixed-intrinsic model export gate."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_fixed_pose_export as fixed


class FixedPoseExportTest(unittest.TestCase):
    def test_command_is_pose_only_and_uses_distinct_model(self):
        argv = fixed.command(Path("/synthetic/new"))
        self.assertEqual(argv[argv.index("-i") + 1], str(fixed.MODEL))
        self.assertEqual(argv[argv.index("-o") + 1], "/synthetic/new/sfm_camera_poses.json")
        self.assertEqual(argv[-3:], ["-V", "-I", "-E"])
        self.assertNotIn("-S", argv)
        self.assertNotIn("-C", argv)

    def test_sealed_fixed_receipt_model_log_and_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sparse").mkdir()
            (root / "logs").mkdir()
            model = root / "sparse/sfm_data.bin"
            log = root / "logs/01-sfm.log"
            model.write_bytes(b"fixed model")
            log.write_bytes(b"sealed log")
            names = [f"NP3_{index:03}.jpg" for index in range(48)]
            model_sha, log_sha = fixed.prior.sha(model), fixed.prior.sha(log)
            inventory = fixed.core.output_inventory(root)
            receipt = {"schema": "openmvg_mustard_fixed_intrinsic_sfm_v1",
                       "status": "failed_registration_or_sparse_gate",
                       "sfm_report": {"views": 48, "poses": 46, "intrinsics": 1,
                                      "tracks": 162, "residuals": 1292},
                       "stage": {"status": "completed", "returncode": 0,
                                 "model_sha256": model_sha, "log_sha256": log_sha,
                                 "command": ["-f", "NONE"]},
                       "inputs": {"source_receipt_sha256": fixed.prior.FAIL_RECEIPT_SHA,
                                  "initial_focal_px": 1536.0, "names": names},
                       "output_inventory": inventory}
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            with patch.object(fixed, "RECEIPT_SHA", fixed.prior.sha(receipt_path)), \
                 patch.object(fixed, "MODEL_SHA", model_sha), \
                 patch.object(fixed, "SFM_LOG_SHA", log_sha):
                checked = fixed.verify_fixed_model(root)
                self.assertEqual(checked["sfm_report"]["poses"], 46)
                (root / "sparse/extra.bin").write_bytes(b"unexpected")
                with self.assertRaisesRegex(RuntimeError, "run contract differs"):
                    fixed.verify_fixed_model(root)

    def test_fresh_output_and_11gib_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "already"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                fixed.preflight(existing)
            with patch.object(fixed.prior.build.fifth.four.fork.base, "tree_bytes", return_value=0), \
                 patch.object(fixed.prior.build.fifth.four.fork.base, "free_bytes",
                              side_effect=[fixed.prior.FLOOR, fixed.prior.FLOOR]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    fixed.prior.capacity(root / "new", prospective=True)


if __name__ == "__main__":
    unittest.main()
