"""Pinned-source fixture test for the evaluation-only OpenMVG LiGT OFF patch."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).parent
PATCH = HERE / "openmvg_v21_ligt_off.patch"
FIXTURE = HERE / "fixtures/openmvg_v21_multiview_CMakeLists.txt"
ATTRIBUTES = HERE.parents[1] / ".gitattributes"
SOURCE_SHA256 = "139ed9e2793e907c3fa74ea805c5b418336f51812ab7c56036aeb14e866233a3"


def fixture_git_env() -> dict[str, str]:
    # CI may export Git worktree/index variables. Keep `git apply` anchored to
    # this disposable fixture, never to the checkout running the tests.
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                "GIT_PREFIX", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    return env


def apply_fixture_patch(root: Path, check: bool = False) -> None:
    env = fixture_git_env()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True,
                   text=True, env=env)
    command = ["git", "-C", str(root), "apply"]
    if check:
        command.append("--check")
    subprocess.run([*command, str(PATCH)], check=True, capture_output=True, text=True,
                   env=env)


class LiGTOffPatchTest(unittest.TestCase):
    def test_fixture_is_exact_pinned_v21_source(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), SOURCE_SHA256)
        text = FIXTURE.read_text()
        self.assertIn("./LiGT/*.cpp", text)
        self.assertIn("file(GLOB_RECURSE REMOVEFILELIGT LiGT_*.cpp)", text)

    @unittest.skipUnless(shutil.which("git"), "git needed for checkout EOL test")
    def test_fixture_and_patch_stay_lf_with_autocrlf_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ("scripts/classical_backend/fixtures/openmvg_v21_multiview_CMakeLists.txt",
                     "scripts/classical_backend/openmvg_v21_ligt_off.patch")
            for path, source in ((".gitattributes", ATTRIBUTES), (paths[0], FIXTURE),
                                 (paths[1], PATCH)):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            env = fixture_git_env()
            subprocess.run(["git", "init", "-q", str(root)], check=True, env=env,
                           capture_output=True, text=True)
            subprocess.run(["git", "-C", str(root), "add", ".gitattributes", *paths],
                           check=True, env=env, capture_output=True, text=True)
            for path in paths:
                (root / path).unlink()
            subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=true",
                            "checkout-index", "--force", "--", *paths],
                           check=True, env=env, capture_output=True, text=True)
            self.assertEqual(hashlib.sha256((root / paths[0]).read_bytes()).hexdigest(), SOURCE_SHA256)
            self.assertEqual((root / paths[1]).read_bytes(), PATCH.read_bytes())

    @unittest.skipUnless(shutil.which("git"), "git is needed for patch syntax check")
    def test_patch_applies_only_exclusion_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "src/openMVG/multiview/CMakeLists.txt"
            target.parent.mkdir(parents=True)
            shutil.copyfile(FIXTURE, target)
            apply_fixture_patch(root, check=True)
            apply_fixture_patch(root)
            original = FIXTURE.read_text()
            expected = original.replace(
                "file(GLOB_RECURSE REMOVEFILELIGT LiGT_*.cpp)",
                "file(GLOB REMOVEFILELIGT ./LiGT/*.cpp)").replace(
                "file(GLOB_RECURSE REMOVEFILELIGT_HEADER LiGT_*.hpp)",
                "file(GLOB REMOVEFILELIGT_HEADER ./LiGT/*.hpp)").replace(
                "list(REMOVE_ITEM multiview_files_cpp ${REMOVEFILELIGT_HEADER})",
                "list(REMOVE_ITEM multiview_files_header ${REMOVEFILELIGT_HEADER})")
            self.assertEqual(target.read_text(), expected)
            self.assertIn("list(REMOVE_ITEM multiview_files_cpp ${REMOVEFILELIGT})", expected)
            self.assertIn("list(REMOVE_ITEM multiview_files_header ${REMOVEFILELIGT_HEADER})", expected)

    def test_glob_exclusion_contract_with_synthetic_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ordinary.cpp").touch()
            (root / "ordinary.hpp").touch()
            (root / "LiGT").mkdir()
            (root / "LiGT/LiGT_algorithm.cpp").touch()
            (root / "LiGT/LiGT_algorithm.hpp").touch()
            all_cpp = set(root.glob("*.cpp")) | set(root.glob("./LiGT/*.cpp"))
            all_hpp = set(root.glob("*.hpp")) | set(root.glob("./LiGT/*.hpp"))
            off_cpp = all_cpp - set(root.glob("./LiGT/*.cpp"))
            off_hpp = all_hpp - set(root.glob("./LiGT/*.hpp"))
            self.assertEqual({p.name for p in off_cpp}, {"ordinary.cpp"})
            self.assertEqual({p.name for p in off_hpp}, {"ordinary.hpp"})
            self.assertIn(root / "LiGT/LiGT_algorithm.cpp", all_cpp)

    @unittest.skipUnless(shutil.which("git"), "git is needed for fixture worktree isolation")
    def test_fixture_patch_ignores_inherited_git_worktree_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "src/openMVG/multiview/CMakeLists.txt"
            target.parent.mkdir(parents=True)
            shutil.copyfile(FIXTURE, target)
            with mock.patch.dict(os.environ, {"GIT_DIR": "/does/not/exist",
                                             "GIT_WORK_TREE": "/does/not/exist"}):
                apply_fixture_patch(root)
            self.assertIn("file(GLOB REMOVEFILELIGT ./LiGT/*.cpp)", target.read_text())

    @unittest.skipUnless(shutil.which("git") and shutil.which("cmake"),
                         "git and cmake are needed for the CMake fixture check")
    def test_patched_cmake_off_excludes_ligt_and_on_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/openMVG/multiview"
            source.mkdir(parents=True)
            shutil.copyfile(FIXTURE, source / "CMakeLists.txt")
            (source / "ordinary.cpp").touch()
            (source / "ordinary.hpp").touch()
            (source / "LiGT").mkdir()
            (source / "LiGT/LiGT_algorithm.cpp").touch()
            (source / "LiGT/LiGT_algorithm.hpp").touch()
            apply_fixture_patch(root)
            selection = (source / "CMakeLists.txt").read_text().split("add_library(openMVG_multiview", 1)[0]
            for enabled in (False, True):
                script = source / "check.cmake"
                script.write_text(
                    f"set(OpenMVG_USE_LIGT {'ON' if enabled else 'OFF'})\n" + selection +
                    "\nforeach(item IN LISTS multiview_files_cpp multiview_files_header)\n"
                    "  if(item MATCHES \"LiGT\")\n"
                    "    message(STATUS \"LIGT_SELECTED\")\n"
                    "  endif()\n"
                    "endforeach()\n")
                result = subprocess.run(["cmake", "-P", str(script)], cwd=source, check=True,
                                        capture_output=True, text=True)
                self.assertEqual("LIGT_SELECTED" in result.stdout, enabled)


if __name__ == "__main__":
    unittest.main()
