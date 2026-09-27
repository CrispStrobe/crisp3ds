import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.metal_backend.smoke import run


class SmokeGuardTest(unittest.TestCase):
    def test_disk_floor_blocks_before_scratch_or_compile(self):
        with patch("scripts.metal_backend.smoke.shutil.disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)), \
             patch("scripts.metal_backend.smoke.subprocess.run") as execute, \
             patch("scripts.metal_backend.smoke.Path.mkdir") as mkdir:
            with self.assertRaisesRegex(RuntimeError, "Insufficient free space"):
                run(Path("."))
            execute.assert_not_called()
            mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
