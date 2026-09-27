"""Synthetic-only GLOBAL SfM command and one-shot guard tests."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_global_sfm_control as control


class MustardGlobalControlTest(unittest.TestCase):
    def test_only_engine_differs_from_original_incremental_sfm_command(self):
        output = Path("/synthetic/global")
        baseline = control.mustard.commands(output)[-1][1]
        global_argv = control.command(output)
        self.assertEqual(baseline[baseline.index("-s") + 1], "INCREMENTAL")
        baseline[baseline.index("-s") + 1] = "GLOBAL"
        self.assertEqual(global_argv, baseline)
        self.assertEqual(global_argv[global_argv.index("-f") + 1], "ADJUST_ALL")
        self.assertEqual(global_argv[global_argv.index("-M") + 1], "matches.e.bin")
        self.assertNotIn("-T", global_argv)  # pinned source default is SOFTL1=3, not LiGT=4
        self.assertNotIn("-R", global_argv)  # pinned source default is L2=2
        self.assertNotIn("-P", global_argv)

    def test_compiled_option_audit_rejects_new_unavailable_flag(self):
        argv = control.command(Path("/synthetic/global"))
        usage = "Usage:\n" + "\n".join(f"[-{key}|--some_option]" for key in "imMosf")
        with patch.object(control.cli.subprocess, "run", return_value=type("Result", (), {
                "returncode": 1, "stdout": usage.encode(), "stderr": b""})()), \
             patch.object(control.cli.base, "sha", return_value="synthetic-binary-sha"):
            control.cli.audit_cli_usage([("sfm_global", argv, "artifact")])
            with self.assertRaisesRegex(RuntimeError, "planned CLI option absent"):
                control.cli.audit_cli_usage([("sfm_global", argv + ["-Z", "bad"], "artifact")])

    def test_fresh_output_and_full_resource_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "already"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                control.preflight(existing)
            with patch.object(control.core.base, "tree_bytes", return_value=0), \
                 patch.object(control.core.base, "free_bytes",
                              side_effect=[control.DISK_FLOOR, control.DISK_FLOOR]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    control.capacity(Path(temporary) / "new", prospective=True)

    def test_non_macos_native_run_fails_closed(self):
        with patch.object(control.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "requires macOS"):
                control.run(Path("/synthetic/uncreated"))

    def test_reviewed_global_source_seal_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source/src/software/SfM/main_SfM.cpp"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"reviewed synthetic GLOBAL source")
            relative = "src/software/SfM/main_SfM.cpp"
            with patch.object(control.core, "BUILD", Path(temporary)), \
                 patch.object(control, "SOURCE_SEALS", {relative: control.core.sha(path)}):
                self.assertEqual(control.verify_source()[relative], control.core.sha(path))
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "source differs"):
                    control.verify_source()


if __name__ == "__main__":
    unittest.main()
