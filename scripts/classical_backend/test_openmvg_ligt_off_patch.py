"""Pinned-source fixture test for the evaluation-only OpenMVG LiGT OFF patch."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).parent
PATCH = HERE / "openmvg_v21_ligt_off.patch"
FIXTURE = HERE / "fixtures/openmvg_v21_multiview_CMakeLists.txt"
SOURCE_SHA256 = "139ed9e2793e907c3fa74ea805c5b418336f51812ab7c56036aeb14e866233a3"


class LiGTOffPatchTest(unittest.TestCase):
    def test_fixture_is_exact_pinned_v21_source(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), SOURCE_SHA256)
        text = FIXTURE.read_text()
        self.assertIn("./LiGT/*.cpp", text)
        self.assertIn("file(GLOB_RECURSE REMOVEFILELIGT LiGT_*.cpp)", text)

    @unittest.skipUnless(shutil.which("git"), "git is needed for patch syntax check")
    def test_patch_applies_only_exclusion_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "src/openMVG/multiview/CMakeLists.txt"
            target.parent.mkdir(parents=True)
            shutil.copyfile(FIXTURE, target)
            subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=root, check=True,
                           capture_output=True, text=True)
            subprocess.run(["git", "apply", str(PATCH)], cwd=root, check=True,
                           capture_output=True, text=True)
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
            subprocess.run(["git", "apply", str(PATCH)], cwd=root, check=True,
                           capture_output=True, text=True)
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
