"""Synthetic sixth-target build gates; never compiles or converts a model."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_converter_build as build


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ConverterBuildTest(unittest.TestCase):
    def test_exact_two_command_delta_and_no_new_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            old_link = "c++ -o Darwin-arm64-Release/openMVG_main_SfM libfoo.a -framework Accelerate"
            common = [f"existing command {index}" for index in range(360)] + [old_link]
            (root / "logs/04-target-commands.log").write_text("\n".join(common) + "\n")
            # The source path is exactly ROOT/source/src/software/SfM/main_*.cpp.
            compile_line = f"c++ -c {root / 'source' / build.MAIN_REL} -o converter.o"
            for library, expected_failure in (("libfoo.a", False), ("libbar.a", True)):
                link = ("c++ -o Darwin-arm64-Release/openMVG_main_ConvertSfM_DataFormat "
                        + library + " -framework Accelerate")
                lines = [*common, compile_line, link]
                payload = ("\n".join(lines) + "\n").encode()
                delta = ("\n".join(sorted((compile_line, link))) + "\n").encode()
                with patch.object(build, "ROOT", root), \
                     patch.object(build, "CLOSURE_SHA", digest(payload)), \
                     patch.object(build, "SORTED_DELTA_SHA", digest(delta)), \
                     patch.object(build.fifth, "target_closure", return_value=("", [])), \
                     patch.object(build.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, payload, b"")):
                    if expected_failure:
                        with self.assertRaisesRegex(RuntimeError, "introduces library"):
                            build.target_closure()
                    else:
                        self.assertEqual(build.target_closure()["new_command_count"], 2)

    def test_dry_run_requires_exactly_two_steps(self):
        result = subprocess.CompletedProcess([], 0, "[1/3] compile\n[3/3] link\n", "")
        with patch.object(build.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "not exactly two"):
                build.dry_run()

    def test_existing_attempt_blocks_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "converter-build-manifest.json").write_text("{}")
            with patch.object(build, "ROOT", root), \
                 patch.object(build, "MANIFEST", root / "converter-build-manifest.json"), \
                 patch.object(build.fifth.four.fork, "reject_git_context"):
                with self.assertRaisesRegex(RuntimeError, "attempt already exists"):
                    build.preflight()


if __name__ == "__main__":
    unittest.main()
