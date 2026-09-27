"""Synthetic guards for the configure-only OpenMVG LiGT-off oracle fork."""

import importlib.util
import json
from pathlib import Path
import subprocess
from unittest import mock
import sys
import tempfile
import unittest


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("openmvg_ligt_fork_supervisor",
                                              HERE / "openmvg_ligt_fork_supervisor.py")
fork = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fork)


class ForkSupervisorTest(unittest.TestCase):
    def test_package_module_invocation_without_optional_deps(self):
        result = subprocess.run([sys.executable, "-m", "scripts.classical_backend.openmvg_ligt_fork_supervisor",
                                 "--help"], cwd=HERE.parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--configure-fork", result.stdout)

    def test_default_cli_is_read_only(self):
        self.assertNotIn("--configure-fork", ["openmvg_ligt_fork_supervisor.py"])
        with mock.patch.object(fork, "preflight", return_value={"operation": "read-only"}) as preflight, \
             mock.patch.object(fork, "configure_fork") as configure, \
             mock.patch.object(sys, "argv", ["openmvg_ligt_fork_supervisor.py"]), \
             mock.patch("builtins.print"):
            fork.main()
        preflight.assert_called_once()
        configure.assert_not_called()

    def test_configure_has_pinned_off_policy_eigen_and_compile_export(self):
        command = fork.configure_command(Path("/fork/source"), Path("/fork/build"))
        self.assertIn("-DOpenMVG_USE_LIGT=OFF", command)
        self.assertIn("-DCMAKE_POLICY_VERSION_MINIMUM=3.5", command)
        self.assertIn(f"-DEigen3_DIR:PATH={fork.base.EIGEN_CONFIG}", command)
        self.assertIn(f"-DEIGEN_DIR:PATH={fork.base.EIGEN_INCLUDE}", command)
        self.assertIn("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", command)
        self.assertNotIn("--build", command)

    def test_patch_context_and_exact_expected_text(self):
        original = fork.FIXTURE.read_text()
        expected = fork.expected_patched_text(original)
        self.assertNotEqual(original, expected)
        self.assertIn("file(GLOB REMOVEFILELIGT ./LiGT/*.cpp)", expected)
        self.assertIn("list(REMOVE_ITEM multiview_files_header ${REMOVEFILELIGT_HEADER})", expected)
        with self.assertRaisesRegex(RuntimeError, "context changed"):
            fork.expected_patched_text(original.replace("LiGT_*.cpp", "other.cpp"))

    def test_patch_hash_tamper_stops_before_source_checks(self):
        with mock.patch.object(fork.base, "sha256", return_value="bad"), \
             mock.patch.object(fork.base, "source_inventory") as inventory:
            with self.assertRaisesRegex(RuntimeError, "patch/fixture seal"):
                fork.seal_patch_and_source()
        inventory.assert_not_called()

    def test_preflight_fails_nonfresh_root_before_any_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "used"
            root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                fork.preflight(root)

    def test_preflight_rejects_inherited_git_context(self):
        with mock.patch.dict(fork.os.environ, {"GIT_DIR": "/other/checkout/.git"}):
            with self.assertRaisesRegex(RuntimeError, "Git worktree/index context"):
                fork.reject_git_context()

    def test_capacity_reserves_entire_one_gib_and_floors(self):
        root = fork.ROOT
        with mock.patch.object(fork.base, "approved_root"), \
             mock.patch.object(fork.base, "tree_bytes", return_value=100), \
             mock.patch.object(fork.base, "free_bytes", side_effect=[fork.FLOOR + fork.TREE_CAP, fork.FLOOR]):
            checked = fork.capacity(root, prospective=True)
        self.assertEqual(checked["external_required"], fork.FLOOR + fork.TREE_CAP - 100)
        with mock.patch.object(fork.base, "approved_root"), \
             mock.patch.object(fork.base, "tree_bytes", return_value=0), \
             mock.patch.object(fork.base, "free_bytes", side_effect=[fork.FLOOR + fork.TREE_CAP - 1,
                                                                      fork.FLOOR]):
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                fork.capacity(root, prospective=True)

    def test_copy_timeout_without_external_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            old = test_root / "old"
            old.mkdir()
            (old / "small.txt").write_text("x")
            new_root = test_root / "new"
            new_root.mkdir()
            with mock.patch.object(fork, "OLD_SOURCE", old), \
                 mock.patch.object(fork.time, "monotonic", side_effect=[0, fork.COPY_TIMEOUT + 1]):
                with self.assertRaisesRegex(RuntimeError, "wall-time cap"):
                    fork.copy_sealed_source(new_root / "source", new_root)
            self.assertFalse((new_root / "source/small.txt").exists())

    def test_copy_incremental_cap_before_file_write(self):
        with tempfile.TemporaryDirectory() as directory:
            test_root = Path(directory)
            old = test_root / "old"
            old.mkdir()
            (old / "too_big.txt").write_text("123456789")
            new_root = test_root / "new"
            new_root.mkdir()
            with mock.patch.object(fork, "OLD_SOURCE", old), \
                 mock.patch.object(fork, "TREE_CAP", 8):
                with self.assertRaisesRegex(RuntimeError, "tree cap"):
                    fork.copy_sealed_source(new_root / "source", new_root)
            self.assertFalse((new_root / "source/too_big.txt").exists())

    def test_graph_gate_rejects_ligt_source_header_object_and_define(self):
        for token in ("LiGT/LiGT_algorithm.cpp", "LiGT/LiGT_algorithm.hpp",
                      "LiGT/LiGT_algorithm.o", "-DUSE_PATENTED_LIGT"):
            with self.subTest(token=token), self.assertRaisesRegex(RuntimeError, "LiGT/patented"):
                self.graph_fixture(extra_ninja=token)

    def test_graph_gate_rejects_compile_command_even_if_ninja_clean(self):
        with self.assertRaisesRegex(RuntimeError, "compile_commands"):
            self.graph_fixture(extra_compile="-DUSE_PATENTED_LIGT")

    def test_graph_gate_rejects_ligt_in_target_command_closure(self):
        with self.assertRaisesRegex(RuntimeError, "four-target closure"):
            self.graph_fixture(extra_target_command="c++ -c LiGT/LiGT_algorithm.cpp")

    def test_graph_gate_rejects_missing_target(self):
        with self.assertRaisesRegex(RuntimeError, "target missing"):
            self.graph_fixture(missing_target=True)

    def test_graph_gate_accepts_clean_synthetic_graph(self):
        result = self.graph_fixture()
        self.assertEqual(result["forbidden_inputs"], 0)
        self.assertEqual(result["compile_units"], 1)

    def graph_fixture(self, extra_ninja="", extra_compile="", extra_target_command="",
                      missing_target=False):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text("CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON\n")
            (build / "compile_commands.json").write_text(json.dumps([
                {"file": "ordinary.cpp", "command": f"c++ {extra_compile} -c ordinary.cpp"}]))
            targets = " ".join(fork.base.TARGETS[:-1] if missing_target else fork.base.TARGETS)
            (build / "build.ninja").write_text(f"{targets}\n{extra_ninja}\n")
            with mock.patch.object(fork.base, "verify_pinned_eigen_configuration"):
                return fork.graph_gate(build, build / "configure.log",
                                       targets + "\n" + extra_target_command)


if __name__ == "__main__":
    unittest.main()
