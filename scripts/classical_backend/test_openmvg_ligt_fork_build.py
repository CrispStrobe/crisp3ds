"""Synthetic resume-only guards; CI needs no OpenMVG checkout or macOS tools."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("openmvg_ligt_fork_build",
                                              HERE / "openmvg_ligt_fork_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class ForkBuildTest(unittest.TestCase):
    def test_module_help_and_default_read_only(self):
        result = subprocess.run([sys.executable, "-m", "scripts.classical_backend.openmvg_ligt_fork_build",
                                 "--help"], cwd=HERE.parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with mock.patch.object(build, "preflight", return_value={"status": "read-only"}) as preflight, \
             mock.patch.object(build, "build_once") as live, \
             mock.patch.object(sys, "argv", ["openmvg_ligt_fork_build.py"]), \
             mock.patch("builtins.print"):
            build.main()
        preflight.assert_called_once()
        live.assert_not_called()

    def test_command_is_only_four_targets_two_threads(self):
        command = build.build_command()
        self.assertEqual(command[:7], ["ninja", "-C", str(build.ROOT / "build"),
                                       "-f", "frozen-build.ninja", "-j", "2"])
        self.assertEqual(command[7:], list(build.fork.base.TARGETS))
        self.assertNotIn("install", command)

    def test_preflight_rejects_missing_or_repeated_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "absent"
            with mock.patch.object(build, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    build.preflight()
            root.mkdir()
            (root / "build-attempt-manifest.json").touch()
            with mock.patch.object(build, "ROOT", root), \
                 mock.patch.object(build, "BUILD_MANIFEST", root / "build-attempt-manifest.json"), \
                 mock.patch.object(build, "FROZEN_NINJA", root / "build/frozen-build.ninja"), \
                 mock.patch.object(build.fork, "reject_git_context"):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    build.preflight()

    def test_preflight_accepts_only_sealed_synthetic_graph(self):
        self.synthetic_preflight()

    def test_preflight_rejects_changed_cache_hash(self):
        with self.assertRaisesRegex(RuntimeError, "seal mismatch"):
            self.synthetic_preflight(bad_cache=True)

    def test_preflight_rejects_changed_graph(self):
        with self.assertRaisesRegex(RuntimeError, "graph changed"):
            self.synthetic_preflight(bad_graph=True)

    def test_preflight_rejects_fetch_command(self):
        with self.assertRaisesRegex(RuntimeError, "fetch/install"):
            self.synthetic_preflight(fetch=True)

    def test_postbuild_seal_rejects_reconfigure(self):
        with mock.patch.object(build.fork.base, "sha256", return_value="bad"):
            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                build.postbuild_seals()

    def test_postbuild_seal_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "logs/04-target-commands.log").write_text("sealed commands")
            (root / "fork-manifest.json").write_text(json.dumps({
                "input": {"seals": {"source": "sealed"}},
                "patched_fork": {"patched_cmake_sha256": build.PATCHED_CMAKE_SHA}}))
            expected = {root / "fork-manifest.json": build.CONFIGURED_MANIFEST_SHA,
                        root / "build/CMakeCache.txt": build.CACHE_SHA,
                        root / "build/frozen-build.ninja": build.GRAPH_SHA["ninja_sha256"]}
            expected.update({root / "logs" / name: digest for name, digest in build.LOG_SHA.items()})
            with mock.patch.object(build, "ROOT", root), \
                 mock.patch.object(build, "FROZEN_NINJA", root / "build/frozen-build.ninja"), \
                 mock.patch.object(build.fork.base, "sha256", side_effect=lambda path: expected[path]), \
                 mock.patch.object(build.fork, "graph_gate", return_value=dict(build.GRAPH_SHA)), \
                 mock.patch.object(build.fork, "seal_patch_and_source", return_value={"source": "changed"}):
                with self.assertRaisesRegex(RuntimeError, "source/patch/Eigen seals changed"):
                    build.postbuild_seals()

    def test_layout_requires_exact_four_main_and_archive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            main_dir = source / "src/software/SfM"
            main_dir.mkdir(parents=True)
            for target in build.fork.base.TARGETS:
                (main_dir / f"main_{target.removeprefix('openMVG_main_')}.cpp").touch()
            ninja = "\n".join(f"build Darwin-arm64-Release/{target}: rule" for target in
                              build.fork.base.TARGETS)
            ninja += "\nbuild Darwin-arm64-Release/libopenMVG_multiview.a: rule"
            ninja += "\nbuild Darwin-arm64-Release/liblib_CoinUtils.a: CoinModelUseful2.cpp.o"
            build.verify_output_layout(ninja, source)
            with self.assertRaisesRegex(RuntimeError, "archive path changed"):
                build.verify_output_layout(ninja.replace("liblib_CoinUtils.a", "other.a"), source)

    def test_dry_run_rejects_cmake_regeneration(self):
        output = "[1/386] Re-running CMake\n[386/386] openMVG_main_SfM"
        fake = mock.Mock(stdout=output, stderr="")
        with mock.patch.object(build.subprocess, "run", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "regenerate CMake"):
                build.dry_run_graph()

    def test_build_log_cap_terminates_child_and_records_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "tmp").mkdir()
            log = root / "logs/05-four-target-build.log"
            state = {"running": True, "signals": []}

            class FakeProcess:
                pid = 123
                returncode = None

                def poll(self):
                    return None if state["running"] else self.returncode

                def wait(self, timeout=None):
                    state["running"] = False
                    self.returncode = -15
                    return self.returncode

            def fake_popen(_command, **kwargs):
                kwargs["stdout"].write(b"12345")
                kwargs["stdout"].flush()
                return FakeProcess()

            def fake_killpg(pid, sig):
                state["signals"].append((pid, sig))
                state["running"] = False

            manifest = {"stages": []}
            with mock.patch.object(build, "ROOT", root), \
                 mock.patch.object(build, "BUILD_LOG", log), \
                 mock.patch.object(build, "BUILD_MANIFEST", root / "build-attempt-manifest.json"), \
                 mock.patch.object(build.fork, "LOG_CAP", 4), \
                 mock.patch.object(build, "save_json"), \
                 mock.patch.object(build.subprocess, "Popen", side_effect=fake_popen), \
                 mock.patch.object(build.os, "killpg", side_effect=fake_killpg, create=True), \
                 mock.patch.object(build.fork.base, "sha256", return_value="sealed"):
                with self.assertRaisesRegex(RuntimeError, "log cap"):
                    build.run_build_log(manifest)
            self.assertEqual(manifest["stages"][0]["status"], "stopped")
            self.assertTrue(state["signals"])
            self.assertEqual(log.stat().st_size, 4)

    def synthetic_preflight(self, bad_cache=False, bad_graph=False, fetch=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("source", "build", "logs", "tmp"):
                (root / name).mkdir()
            graph = dict(build.GRAPH_SHA, compile_units=407, forbidden_inputs=0)
            configured = {"status": "configured_graph_clean_no_compile",
                          "stages": [{"status": "completed"}] * 5,
                          "input": {"root": str(root), "seals": {"patch_sha256": build.fork.PATCH_SHA}},
                          "patched_fork": {"patched_cmake_sha256": build.PATCHED_CMAKE_SHA},
                          "graph_gate": graph}
            live_seals = {"patch_sha256": build.fork.PATCH_SHA, "source": {}}
            configured["input"]["seals"] = live_seals
            (root / "fork-manifest.json").write_text(json.dumps(configured))
            (root / "build/CMakeCache.txt").touch()
            (root / "build/build.ninja").touch()
            for name in build.LOG_SHA:
                (root / "logs" / name).write_text("curl http://example.com" if fetch and name ==
                                                   "04-target-commands.log" else "commands")
            expected = {root / "fork-manifest.json": build.CONFIGURED_MANIFEST_SHA,
                        root / "build/CMakeCache.txt": build.CACHE_SHA}
            expected.update({root / "logs" / name: digest for name, digest in build.LOG_SHA.items()})
            expected[root / "build/build.ninja"] = build.GRAPH_SHA["ninja_sha256"]
            if bad_cache:
                expected[root / "build/CMakeCache.txt"] = "bad"
            live_graph = dict(graph)
            if bad_graph:
                live_graph["ninja_sha256"] = "bad"
            with mock.patch.object(build, "ROOT", root), \
                 mock.patch.object(build, "BUILD_LOG", root / "logs/05-four-target-build.log"), \
                 mock.patch.object(build, "BUILD_MANIFEST", root / "build-attempt-manifest.json"), \
                 mock.patch.object(build, "RECEIPT", root / "license-closure-receipt.json"), \
                 mock.patch.object(build, "FROZEN_NINJA", root / "build/frozen-build.ninja"), \
                 mock.patch.object(build.fork, "reject_git_context"), \
                 mock.patch.object(build.fork.base, "sha256", side_effect=lambda path: expected[path]), \
                 mock.patch.object(build.fork, "seal_patch_and_source", return_value=live_seals), \
                 mock.patch.object(build.fork, "verify_patched_fork", return_value=configured["patched_fork"]), \
                 mock.patch.object(build.fork, "graph_gate", return_value=live_graph), \
                 mock.patch.object(build, "verify_output_layout"), \
                 mock.patch.object(build, "dry_run_graph", return_value={"steps": 386}), \
                 mock.patch.object(build.fork, "capacity", return_value={"fork_bytes": 100}):
                return build.preflight()


if __name__ == "__main__":
    unittest.main()
