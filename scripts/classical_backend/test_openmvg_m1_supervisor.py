"""Synthetic guardrail tests; no OpenMVG fetch, build, or photo run."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE = Path(__file__).with_name("openmvg_m1_supervisor.py")
spec = importlib.util.spec_from_file_location("openmvg_m1_supervisor", MODULE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class SupervisorTest(unittest.TestCase):
    def test_exact_source_and_target_seal(self):
        self.assertEqual(len(supervisor.SOURCE_REV), 40)
        self.assertEqual(len(supervisor.SUBMODULES), 3)
        command = supervisor.build_command(Path("/external/build"))
        self.assertEqual(command[-2:], ["--parallel", "2"])
        self.assertEqual(command[command.index("--target") + 1:-2], list(supervisor.TARGETS))
        self.assertNotIn("all", command)
        self.assertNotIn("install", command)
        options = supervisor.configure_command(Path("/external/source"), Path("/external/build"))
        self.assertIn("-DOpenMVG_USE_LIGT=OFF", options)
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", options)

    def test_disk_reservation_and_margin(self):
        root = Path("/external/new")
        with mock.patch.object(supervisor, "tree_bytes", return_value=0), \
             mock.patch.object(supervisor, "free_bytes", side_effect=[13 * supervisor.GIB, 11 * supervisor.GIB]):
            self.assertEqual(supervisor.capacity(root, Path("/internal"), True)["external_required"], 13 * supervisor.GIB)
        with mock.patch.object(supervisor, "tree_bytes", return_value=0), \
             mock.patch.object(supervisor, "free_bytes", side_effect=[13 * supervisor.GIB - 1, 11 * supervisor.GIB]):
            with self.assertRaisesRegex(RuntimeError, "disk guard"):
                supervisor.capacity(root, Path("/internal"), True)

    def test_managed_tree_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_text("x")
            (root / "link").symlink_to(root / "file")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                supervisor.tree_bytes(root)

    def test_process_tree_rss_includes_descendants_only(self):
        table = "100 1 200\n101 100 300\n102 101 400\n900 1 9999\n"
        self.assertEqual(supervisor.process_rss_kib(100, table), 900)

    def test_sealed_input_rejects_non_sealed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train-names.txt").write_text("NP3_000.jpg\n")
            (root / "stage-report.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                supervisor.sealed_inputs(root)

    def test_build_refuses_existing_root_before_any_fetch(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(supervisor, "approved_root"), \
                 mock.patch.object(supervisor.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "nonfresh"):
                    supervisor.build(Path(temp), Path(temp))
                popen.assert_not_called()

    def test_cmake_implicit_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cmake = Path(temp) / "src" / "CMakeLists.txt"
            cmake.parent.mkdir()
            cmake.write_text("FetchContent_Declare(hidden_dep URL https://example.invalid/pkg)\n")
            with self.assertRaisesRegex(RuntimeError, "fetch during configure"):
                supervisor.reject_implicit_fetches(Path(temp))

    def test_log_cap_stops_process_and_records_manifest(self):
        class FakeProcess:
            pid = 123
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -15
                return self.returncode

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tmp").mkdir()
            (root / "logs").mkdir()
            log = root / "logs" / "test.log"
            manifest = {"stages": []}

            def start(*args, **kwargs):
                kwargs["stdout"].write(b"x" * 65)
                kwargs["stdout"].flush()
                return FakeProcess()

            with mock.patch.object(supervisor, "LOG_CAP", 64), \
                 mock.patch.object(supervisor, "capacity"), \
                 mock.patch.object(supervisor.subprocess, "Popen", side_effect=start), \
                 mock.patch.object(supervisor.subprocess, "check_output", return_value="123 1 1\n"), \
                 mock.patch.object(supervisor.os, "killpg", create=True) as killpg:
                with self.assertRaisesRegex(RuntimeError, "log cap exceeded"):
                    supervisor.guarded_run(["synthetic"], root, root, 10, {}, log, manifest)
            killpg.assert_called_once_with(123, supervisor.signal.SIGTERM)
            self.assertEqual(manifest["stages"][0]["status"], "stopped")
            self.assertEqual(manifest["stages"][0]["log_bytes"], 64)
            self.assertTrue(manifest["stages"][0]["log_truncated_at_cap"])
            self.assertTrue((root / "build-manifest.json").is_file())

    def test_resume_command_only_adds_reviewed_policy_floor(self):
        source, build = Path("/external/source"), Path("/external/build")
        original = supervisor.configure_command(source, build)
        resumed = supervisor.resume_configure_command(source, build)
        self.assertEqual(resumed[:-1], original)
        self.assertEqual(resumed[-1], "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")

    def test_resume_preflight_rejects_changed_first_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "build").mkdir()
            (root / "source").mkdir()
            (root / "logs").mkdir()
            (root / "build-manifest.json").write_text("{}")
            (root / "logs" / "04-configure.log").write_text("changed")
            def fake_sha(path):
                if path.name == "build-manifest.json":
                    return supervisor.FIRST_MANIFEST_SHA
                return "wrong"
            with mock.patch.object(supervisor, "approved_root"), \
                 mock.patch.object(supervisor, "sha256", side_effect=fake_sha), \
                 mock.patch.object(supervisor.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "first configure log changed"):
                    supervisor.resume_preflight(root, root)
                popen.assert_not_called()

    def test_resume_preflight_is_read_only_on_sealed_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "build").mkdir()
            (root / "source").mkdir()
            (root / "logs").mkdir()
            (root / "logs" / "04-configure.log").write_text("first log")
            first = {"stages": [{"status": s} for s in
                    ("completed", "completed", "completed", "stopped")],
                    "source_inventory": {"source_revision": supervisor.SOURCE_REV}}
            (root / "build-manifest.json").write_text(json.dumps(first))
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            def fake_sha(path):
                return (supervisor.FIRST_MANIFEST_SHA if path.name == "build-manifest.json"
                        else supervisor.FIRST_CONFIG_LOG_SHA)
            with mock.patch.object(supervisor, "approved_root"), \
                 mock.patch.object(supervisor, "sha256", side_effect=fake_sha), \
                 mock.patch.object(supervisor, "source_inventory", return_value=first["source_inventory"]), \
                 mock.patch.object(supervisor, "reject_implicit_fetches"), \
                 mock.patch.object(supervisor, "capacity", return_value={"managed_bytes": 100}), \
                 mock.patch.object(supervisor, "tree_bytes", return_value=100):
                result = supervisor.resume_preflight(root, root)
            self.assertEqual(result["resume_configure"][-1], "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_attempt3_command_adds_only_two_local_eigen_hints(self):
        source, build = Path("/external/source"), Path("/external/build")
        previous = supervisor.resume_configure_command(source, build)
        attempt3 = supervisor.eigen_configure_command(source, build)
        self.assertEqual(attempt3[:-2], previous)
        self.assertEqual(attempt3[-2:], [
            f"-DEigen3_DIR:PATH={supervisor.EIGEN_CONFIG}",
            f"-DEIGEN_DIR:PATH={supervisor.EIGEN_INCLUDE}"])
        self.assertIn("-DOpenMVG_USE_LIGT=OFF", attempt3)

    def test_attempt3_cache_guard_rejects_changed_eigen_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "CMakeCache.txt"
            cache.write_text("Eigen3_DIR:PATH=/opt/homebrew/share/eigen3/cmake\n")
            with self.assertRaisesRegex(RuntimeError, "cache changed"):
                supervisor.verify_attempt2_cache(Path(temp))
            audited = "\n".join(f"{key}:STRING={value}" for key, value in {
                "Eigen3_DIR": "/wrong/eigen3/cmake", "EIGEN_DIR": "/opt/homebrew/include/eigen3",
                "CMAKE_POLICY_VERSION_MINIMUM": "3.5", "OpenMVG_USE_LIGT": "OFF",
                "CXSPARSE": "OFF", "SUITESPARSE": "OFF", "EIGENSPARSE": "ON",
                "LAPACK": "ON"}.items())
            cache.write_text(audited)
            with mock.patch.object(supervisor, "sha256", return_value=supervisor.SECOND_CACHE_SHA):
                with self.assertRaisesRegex(RuntimeError, "not as audited"):
                    supervisor.verify_attempt2_cache(Path(temp))

    def test_post_configure_gate_requires_local_34_in_cache_log_and_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp)
            cache = {"Eigen3_DIR": str(supervisor.EIGEN_CONFIG),
                     "EIGEN_DIR": str(supervisor.EIGEN_INCLUDE),
                     "CMAKE_POLICY_VERSION_MINIMUM": "3.5", "OpenMVG_USE_LIGT": "OFF"}
            (build / "CMakeCache.txt").write_text("\n".join(
                f"{key}:STRING={value}" for key, value in cache.items()))
            log = build / "08.log"
            log.write_text(f"-- -- Found Eigen version 3.4.0: {supervisor.EIGEN_INCLUDE}\n")
            ninja = build / "build.ninja"
            ninja.write_text(f"INCLUDES = -isystem {supervisor.EIGEN_INCLUDE}\n")
            supervisor.verify_pinned_eigen_configuration(build, log)
            log.write_text("-- Configuring done\n")
            with self.assertRaisesRegex(RuntimeError, "did not report"):
                supervisor.verify_pinned_eigen_configuration(build, log)
            log.write_text("-- -- Found Eigen version ..\n")
            with self.assertRaisesRegex(RuntimeError, "did not report"):
                supervisor.verify_pinned_eigen_configuration(build, log)
            log.write_text(f"-- -- Found Eigen version 3.4.0: {supervisor.EIGEN_INCLUDE}\n")
            ninja.write_text("INCLUDES = -isystem /unrelated/include\n")
            with self.assertRaisesRegex(RuntimeError, "compile rules"):
                supervisor.verify_pinned_eigen_configuration(build, log)
            ninja.write_text(f"INCLUDES = -isystem {supervisor.EIGEN_INCLUDE} -isystem /opt/homebrew/include/eigen3\n")
            with self.assertRaisesRegex(RuntimeError, "compile rules"):
                supervisor.verify_pinned_eigen_configuration(build, log)
            ninja.write_text(f"INCLUDES = -isystem {supervisor.EIGEN_INCLUDE}\n")
            cache["EIGEN_INCLUDE_DIR"] = "/opt/homebrew/include/eigen3"
            (build / "CMakeCache.txt").write_text("\n".join(
                f"{key}:STRING={value}" for key, value in cache.items()))
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                supervisor.verify_pinned_eigen_configuration(build, log)

    def test_attempt3_preflight_rejects_changed_second_log_before_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "build").mkdir()
            (root / "logs").mkdir()
            for name in ("build-manifest.json", "build-manifest-attempt1.json"):
                (root / name).write_text("{}")
            for name in ("04-configure.log", "06-resume-configure.log"):
                (root / "logs" / name).write_text("synthetic")
            def fake_sha(path):
                return {"build-manifest-attempt1.json": supervisor.FIRST_MANIFEST_SHA,
                        "build-manifest.json": supervisor.SECOND_MANIFEST_SHA,
                        "04-configure.log": supervisor.FIRST_CONFIG_LOG_SHA,
                        "06-resume-configure.log": "wrong"}[path.name]
            with mock.patch.object(supervisor, "approved_root"), \
                 mock.patch.object(supervisor, "sha256", side_effect=fake_sha), \
                 mock.patch.object(supervisor.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "second configure log changed"):
                    supervisor.eigen_resume_preflight(root, root)
                popen.assert_not_called()

    def test_build_only_preflight_rejects_changed_configured_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "build").mkdir()
            (root / "logs").mkdir()
            for name in ("build-manifest-attempt1.json", "build-manifest-attempt2.json",
                         "build-manifest.json"):
                (root / name).write_text("{}")
            def fake_sha(path):
                return {"build-manifest-attempt1.json": supervisor.FIRST_MANIFEST_SHA,
                        "build-manifest-attempt2.json": supervisor.SECOND_MANIFEST_SHA,
                        "build-manifest.json": "wrong"}[path.name]
            with mock.patch.object(supervisor, "approved_root"), \
                 mock.patch.object(supervisor, "sha256", side_effect=fake_sha), \
                 mock.patch.object(supervisor.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "configured attempt seal changed"):
                    supervisor.eigen_build_preflight(root, root)
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
