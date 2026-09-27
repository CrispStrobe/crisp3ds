"""Synthetic-only one-shot fifth-target guards; no OpenMVG install required in CI."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("openmvg_geometric_filter_build",
                                              HERE / "openmvg_geometric_filter_build.py")
fifth = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fifth)


class FifthTargetTest(unittest.TestCase):
    def test_module_help_and_default_read_only(self):
        result = subprocess.run([sys.executable, "-m",
                                 "scripts.classical_backend.openmvg_geometric_filter_build", "--help"],
                                cwd=HERE.parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with mock.patch.object(fifth, "preflight", return_value={"status": "read-only"}) as preflight, \
             mock.patch.object(fifth, "build_once") as live, \
             mock.patch.object(sys, "argv", ["openmvg_geometric_filter_build.py"]), \
             mock.patch("builtins.print"):
            fifth.main()
        preflight.assert_called_once()
        live.assert_not_called()

    def test_command_has_only_fifth_target_and_two_threads(self):
        self.assertEqual(fifth.command(), ["ninja", "-C", str(fifth.ROOT / "build"),
                                           "-f", "frozen-build.ninja", "-j", "2",
                                           "openMVG_main_GeometricFilter"])

    def test_preflight_rejects_repeated_attempt_before_source_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "geometric-filter-build-manifest.json"
            attempt.touch()
            with mock.patch.object(fifth, "ROOT", root), \
                 mock.patch.object(fifth, "MANIFEST", attempt), \
                 mock.patch.object(fifth.four.fork, "reject_git_context"), \
                 mock.patch.object(fifth, "sealed_four_state") as source:
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    fifth.preflight()
            source.assert_not_called()

    def test_exact_two_command_delta_has_no_new_library(self):
        self.synthetic_closure()

    def test_closure_rejects_novel_library(self):
        with self.assertRaisesRegex(RuntimeError, "new library"):
            self.synthetic_closure(new_library=True)

    def test_closure_rejects_patented_macro(self):
        with self.assertRaisesRegex(RuntimeError, "LiGT/patented"):
            self.synthetic_closure(patented=True)

    def test_dry_run_rejects_cmake_regeneration(self):
        fake = mock.Mock(stdout="[1/2] Re-running CMake\n[2/2] Linking", stderr="")
        with mock.patch.object(fifth.subprocess, "run", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "safe two-step"):
                fifth.dry_run()

    def test_four_receipt_seal_rejects_changed_manifest(self):
        with mock.patch.object(fifth.four.fork.base, "sha256", return_value="bad"):
            with self.assertRaisesRegex(RuntimeError, "receipt/log changed"):
                fifth.sealed_four_state()

    def test_log_cap_terminates_child_and_preserves_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "tmp").mkdir()
            log = root / "logs/06-geometric-filter-build.log"
            state = {"running": True, "killed": False}

            class FakeProcess:
                pid = 123
                returncode = None

                def poll(self):
                    return None if state["running"] else self.returncode

                def wait(self, timeout=None):
                    self.returncode = -15
                    state["running"] = False
                    return self.returncode

            def fake_popen(_command, **kwargs):
                kwargs["stdout"].write(b"12345")
                kwargs["stdout"].flush()
                return FakeProcess()

            def fake_killpg(_pid, _sig):
                state["killed"] = True
                state["running"] = False

            manifest = {"stages": []}
            with mock.patch.object(fifth, "ROOT", root), \
                 mock.patch.object(fifth, "LOG", log), \
                 mock.patch.object(fifth, "MANIFEST", root / "fifth.json"), \
                 mock.patch.object(fifth.four.fork, "LOG_CAP", 4), \
                 mock.patch.object(fifth.four, "save_json"), \
                 mock.patch.object(fifth.subprocess, "Popen", side_effect=fake_popen), \
                 mock.patch.object(fifth.os, "killpg", side_effect=fake_killpg, create=True), \
                 mock.patch.object(fifth.four.fork.base, "sha256", return_value="sealed"):
                with self.assertRaisesRegex(RuntimeError, "log cap"):
                    fifth.run_logged(manifest)
            self.assertTrue(state["killed"])
            self.assertEqual(log.stat().st_size, 4)
            self.assertEqual(manifest["stages"][0]["status"], "stopped")

    def synthetic_closure(self, new_library=False, patented=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            common = [f"common compile {index}" for index in range(365)]
            old = "\n".join(common + [": && c++ -o Darwin-arm64-Release/openMVG_main_SfM "
                                        "libold.a -framework Accelerate"]) + "\n"
            (root / "logs/04-target-commands.log").write_text(old)
            compile_line = f"c++ -c {root / 'source' / fifth.MAIN_REL}"
            if patented:
                compile_line += " -DUSE_PATENTED_LIGT"
            link_line = ": && c++ -o Darwin-arm64-Release/openMVG_main_GeometricFilter libold.a"
            if new_library:
                link_line += " libnew.a"
            lines = common + [compile_line, link_line]
            output = ("\n".join(lines) + "\n").encode()
            fake = mock.Mock(stdout=output)
            with mock.patch.object(fifth, "ROOT", root), \
                 mock.patch.object(fifth, "FIFTH_CLOSURE_SHA", hashlib.sha256(output).hexdigest()), \
                 mock.patch.object(fifth, "FIFTH_DELTA_SHA", tuple(hashlib.sha256(line.encode()).hexdigest()
                                                              for line in lines[-2:])), \
                 mock.patch.object(fifth.subprocess, "run", return_value=fake):
                return fifth.target_closure()


if __name__ == "__main__":
    unittest.main()
