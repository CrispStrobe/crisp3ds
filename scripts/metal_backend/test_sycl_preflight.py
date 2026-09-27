"""Read-only contracts for the proposed AliceVision SYCL CPU build path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.metal_backend import sycl_preflight as audit


class SyclPreflightTests(unittest.TestCase):
    def test_version_parser(self):
        self.assertEqual(audit.version_tuple("cmake version 3.30.7"), (3, 30, 7))
        self.assertEqual(audit.version_tuple("20.1"), (20, 1, 0))
        self.assertIsNone(audit.version_tuple("unknown"))

    def test_source_pin_and_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "CMakeLists.txt").write_text("ALICEVISION_BUILD_DEPENDENCIES add_subdirectory(src)")
            def runner(argv, cwd=None):
                if "HEAD" in argv:
                    return audit.ALICEVISION_REVISION
                if "--show-toplevel" in argv:
                    return str(source.resolve())
                return ""
            self.assertEqual(audit.source_contract(source, audit.ALICEVISION_REVISION,
                                                   {"CMakeLists.txt": ("add_subdirectory(src)",)}, runner), [])
            self.assertIn("source git HEAD", audit.source_contract(source, audit.ALICEVISION_REVISION,
                         {"CMakeLists.txt": ()}, lambda argv, cwd=None: "bad")[0])
            self.assertIn("source contract changed", audit.source_contract(source, audit.ALICEVISION_REVISION,
                          {"CMakeLists.txt": ("missing",)}, runner)[0])
            def dirty(argv, cwd=None):
                return " M CMakeLists.txt" if "status" in argv else runner(argv, cwd)
            self.assertIn("dirty", audit.source_contract(source, audit.ALICEVISION_REVISION,
                          {"CMakeLists.txt": ()}, dirty)[0])

    def test_cache_rejects_autodownload_and_wrong_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "acpp"
            package.mkdir()
            cache = root / "CMakeCache.txt"
            cache.write_text("\n".join(f"{key}:STRING={value}" for key, value in
                                       {**audit.REQUIRED_CACHE, "CMAKE_HOME_DIRECTORY": str(root.resolve()),
                                        "AdaptiveCpp_DIR": str(package)}.items()))
            self.assertEqual(audit.cache_contract(cache, root), [])
            cache.write_text(cache.read_text().replace("ALICEVISION_BUILD_DEPENDENCIES:STRING=OFF",
                                                       "ALICEVISION_BUILD_DEPENDENCIES:STRING=ON"))
            self.assertTrue(any("ALICEVISION_BUILD_DEPENDENCIES" in issue
                                for issue in audit.cache_contract(cache, root)))

    def test_artifacts_require_exact_nonempty_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in audit.TARGETS:
                (root / name).write_bytes(b"binary")
                (root / name).chmod(0o755)
            self.assertEqual(audit.artifact_contract(root), [])
            (root / audit.TARGETS[0]).write_bytes(b"")
            self.assertEqual(len(audit.artifact_contract(root)), 1)

    def test_adaptivecpp_full_profile_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "CMakeCache.txt"
            cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={root.resolve()}\n"
                             "ACPP_COMPILER_FEATURE_PROFILE:STRING=minimal\n")
            self.assertIn("full compiler", audit.adaptivecpp_cache_contract(cache, root)[0])
            cache.write_text(cache.read_text().replace("minimal", "full"))
            self.assertEqual(audit.adaptivecpp_cache_contract(cache, root), [])

    def test_missing_contracts_never_claim_build_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = lambda argv, cwd=None: {"cmake": "cmake version 3.30.7",
                                                "llvm-config": "LLVM version 20.1.0",
                                                "ninja": "1.12.1", "acpp": "AdaptiveCpp 25.02"}[argv[0]]
            report = audit.inspect(folder, runner=runner,
                                   disk=lambda path: SimpleNamespace(free=audit.MIN_FREE_BYTES))
            self.assertFalse(report["contract_met"])
            self.assertIsNone(report["build_ready"])
            self.assertEqual(report["checks"]["disk"], [])
            relaxed = audit.inspect(folder, runner=runner, min_free_gib=0,
                                    disk=lambda path: SimpleNamespace(free=0))
            self.assertEqual(relaxed["checks"]["disk"], [])
            with self.assertRaisesRegex(ValueError, "min_free_gib"):
                audit.inspect(folder, min_free_gib=-1)


if __name__ == "__main__":
    unittest.main()
