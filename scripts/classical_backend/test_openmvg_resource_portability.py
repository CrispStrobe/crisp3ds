"""Import-time portability and fail-closed native OpenMVG execution tests."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts.classical_backend import openmvg_photo_control
from scripts.classical_backend import openmvg_mustard_pose_export
from scripts.classical_backend import openmvg_mustard_fixed_intrinsic_ablation
from scripts.classical_backend import openmvg_mustard_fixed_pose_export


MODULES = (
    openmvg_photo_control,
    openmvg_mustard_pose_export,
    openmvg_mustard_fixed_intrinsic_ablation,
    openmvg_mustard_fixed_pose_export,
)


class ResourcePortabilityTests(unittest.TestCase):
    def test_modules_import_without_resource(self) -> None:
        try:
            with mock.patch.dict(sys.modules, {"resource": None}):
                for module in MODULES:
                    with self.subTest(module=module.__name__):
                        self.assertIsNone(importlib.reload(module).resource)
        finally:
            for module in MODULES:
                importlib.reload(module)

    def test_missing_resource_fails_before_preflight_or_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for module in MODULES:
                with self.subTest(module=module.__name__):
                    output = Path(folder) / module.__name__.rsplit(".", 1)[-1]
                    with (mock.patch.object(module, "resource", None),
                          mock.patch.object(module, "preflight", side_effect=AssertionError("preflight called"))):
                        with self.assertRaisesRegex(RuntimeError, "macOS resource limits"):
                            module.run(output)
                    self.assertFalse(output.exists())

    def test_non_macos_fails_even_with_resource_available(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for module in MODULES:
                with self.subTest(module=module.__name__):
                    output = Path(folder) / module.__name__.rsplit(".", 1)[-1]
                    with (mock.patch.object(module.sys, "platform", "win32"),
                          mock.patch.object(module, "resource", object()),
                          mock.patch.object(module, "preflight", side_effect=AssertionError("preflight called"))):
                        with self.assertRaisesRegex(RuntimeError, "macOS resource limits"):
                            module.run(output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
