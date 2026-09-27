"""Synthetic-only gates for the fixed-intrinsic mustard SfM ablation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_fixed_intrinsic_ablation as ablation
from scripts.classical_backend import openmvg_sceaux_photo_control_v2 as cli


class FixedIntrinsicAblationTest(unittest.TestCase):
    def test_command_changes_only_intrinsic_option_and_output(self):
        output = Path("/synthetic/new")
        argv = ablation.command(output)
        self.assertEqual(argv[-4:], ["-s", "INCREMENTAL", "-f", "NONE"])
        self.assertEqual(argv[argv.index("-M") + 1], "matches.e.bin")
        self.assertEqual(argv[argv.index("-i") + 1], str(output / "matches/sfm_data.json"))
        self.assertEqual(argv[argv.index("-m") + 1], str(output / "matches"))
        self.assertEqual(argv[argv.index("-o") + 1], str(output / "sparse"))
        self.assertNotIn("-P", argv)

    def test_usage_rejects_uncompiled_option(self):
        argv = ablation.command(Path("/synthetic/new"))
        usage = "Usage:\n" + "\n".join(f"[-{key}|--some_option]" for key in "imMosf")
        with patch.object(cli.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": usage.encode(), "stderr": b""})()), \
             patch.object(cli.base, "sha", return_value="synthetic-binary-sha"):
            cli.audit_cli_usage([("sfm", argv, "artifact")])
            with self.assertRaisesRegex(RuntimeError, "planned CLI option absent"):
                cli.audit_cli_usage([("sfm", argv + ["-Z", "bad"], "artifact")])

    def test_reused_files_and_photo_hashes_are_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "matches").mkdir()
            (source / "images").mkdir()
            names = [f"NP3_{index:03}.jpg" for index in range(48)]
            scene = {"root_path": str(source / "images"), "views": [{}] * 48,
                     "extrinsics": [], "structure": [], "control_points": [],
                     "intrinsics": [{"value": {"ptr_wrapper": {"data": {
                         "width": 1280, "height": 1024, "focal_length": 1536.0,
                         "principal_point": [640.0, 512.0], "disto_k1": [0.0]}}}}]}
            (source / "matches/sfm_data.json").write_text(json.dumps(scene))
            for relative in ("matches/image_describer.json", "matches/matches.e.bin"):
                (source / relative).write_bytes(b"photo-derived")
            for index in range(6):
                (source / "matches" / f"auxiliary_{index}.txt").write_bytes(b"sealed auxiliary")
            for name in names:
                (source / "images" / name).write_bytes(name.encode())
                for suffix in ("feat", "desc"):
                    (source / "matches" / f"{name[:-4]}.{suffix}").write_bytes(name.encode())
            inventory = ablation.core.output_inventory(source)
            self.assertIn("matches/sfm_data.json", inventory)
            self.assertTrue(all("\\" not in relative for relative in inventory))
            receipt = {"status": "failed_registration_or_sparse_gate",
                       "sfm_report": {"views": 48, "poses": 46},
                       "stages": [{"status": "completed"}] * 4 +
                                 [{"status": "completed", "command": ["-f", "ADJUST_ALL"]}],
                       "photos": {"names": names, "photo_sha256": {
                           name: ablation.core.sha(source / "images" / name) for name in names}},
                       "output_inventory": inventory}
            (source / "receipt.json").write_text(json.dumps(receipt))
            with patch.object(ablation, "SOURCE_RECEIPT_SHA", ablation.core.sha(source / "receipt.json")):
                checked = ablation.verify_inputs(source)
                self.assertEqual(len(checked["reused_file_sha256"]), 105)
                destination = source / "new"
                (destination / "matches").mkdir(parents=True)
                with patch.object(ablation, "capacity", return_value={}):
                    staged = ablation.stage_matches(destination, checked["reused_file_sha256"], source)
                self.assertEqual(staged, checked["reused_file_sha256"])
                self.assertEqual({p.name for p in (destination / "matches").iterdir()},
                                 {p.name for p in (source / "matches").iterdir()})
                (destination / "matches/auxiliary_0.txt").write_bytes(b"tampered staged copy")
                with self.assertRaisesRegex(RuntimeError, "staged match file differs"):
                    ablation.verify_staged_matches(destination, staged)
                (source / "matches/NP3_000.feat").write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "full -001 source inventory differs"):
                    ablation.verify_inputs(source)

    def test_fresh_output_and_full_disk_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                ablation.preflight(output)
            with patch.object(ablation.core.base, "tree_bytes", return_value=0), \
                 patch.object(ablation.core.base, "free_bytes",
                              side_effect=[ablation.DISK_FLOOR, ablation.DISK_FLOOR]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    ablation.capacity(Path(temporary) / "new", prospective=True)


if __name__ == "__main__":
    unittest.main()
