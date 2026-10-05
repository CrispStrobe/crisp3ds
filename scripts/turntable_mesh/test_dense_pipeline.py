"""Checks for the stage runner of the pipeline driver (standard library only)."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.turntable_mesh.dense_pipeline import bounded


class BoundedTest(unittest.TestCase):
    def test_exit_code_and_log_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "stage.log"
            result = bounded([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], log, 60,
                             dict(os.environ))
            self.assertEqual(result["exit_code"], 3)
            self.assertFalse(result["timed_out"])
            self.assertIn("hello", log.read_text())

    def test_deadline_ends_the_stage_and_its_child(self):
        # The stage starts a child of its own; both must be gone at the deadline,
        # otherwise the child would still be writing its marker afterwards.
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "marker"
            child = ("import time, pathlib, sys\n"
                     "time.sleep(6)\n"
                     "pathlib.Path(sys.argv[1]).write_text('alive')\n")
            parent = ("import subprocess, sys, time\n"
                      "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
                      "time.sleep(120)\n")
            result = bounded([sys.executable, "-c", parent, child, marker], Path(folder) / "stage.log", 2,
                             dict(os.environ))
            self.assertTrue(result["timed_out"])
            self.assertIsNone(result["exit_code"])
            self.assertLess(result["seconds"], 60)
            import time
            time.sleep(7)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
