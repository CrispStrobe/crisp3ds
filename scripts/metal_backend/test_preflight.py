import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.metal_backend.preflight import inspect, version_at_least


class PreflightTest(unittest.TestCase):
    def test_version_gate(self):
        self.assertTrue(version_at_least("cmake version 4.1.1", (3, 28, 0)))
        self.assertFalse(version_at_least("cmake version 3.27.9", (3, 28, 0)))
        self.assertFalse(version_at_least(None, (3, 28, 0)))

    def test_missing_compiler_is_not_ready(self):
        def runner(*argv):
            if argv == ("xcrun", "--find", "metal"):
                return None
            return "cmake version 4.1.1" if argv[0] == "cmake" else "/present"
        result = inspect(Path("."), runner=runner,
                         disk=lambda _: SimpleNamespace(free=12 * 1024**3),
                         system=lambda: "Darwin", machine=lambda: "arm64")
        self.assertFalse(result["toolchain_present"])
        self.assertIsNone(result["build_ready"])

    def test_present_tools_do_not_imply_build_ready(self):
        def runner(*argv):
            return "cmake version 4.1.1" if argv[0] == "cmake" else "/present"
        result = inspect(Path("."), runner=runner,
                         disk=lambda _: SimpleNamespace(free=9 * 1024**3),
                         system=lambda: "Darwin", machine=lambda: "arm64")
        self.assertTrue(result["toolchain_present"])
        self.assertFalse(result["disk_floor_10gib_met"])
        self.assertIsNone(result["build_ready"])


if __name__ == "__main__":
    unittest.main()
