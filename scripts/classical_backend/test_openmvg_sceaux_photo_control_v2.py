"""Synthetic v2 parser and failed-run seals; no photographs or SfM execution."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_sceaux_photo_control_v2 as v2


USAGES = {
    "listing": "ifogc",
    "features": "iompu",
    "putative": "iornp",
    "geometric": "imogpr",
    "sfm": "imMosf",
}


def fake_usage(name: str):
    return "Usage:\n" + "\n".join(f"[-{letter}|--option_{letter}]" for letter in USAGES[name])


class V2ControlTest(unittest.TestCase):
    def test_corrected_command_and_compiled_usage_contract(self):
        stages = v2.commands(Path("/synthetic"))
        self.assertNotIn("-n", stages[1][1])
        names = [stage[0] for stage in stages]
        with patch.object(v2.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 1, b"", fake_usage(name).encode()) for name in names
        ]), patch.object(v2.base, "sha", return_value="synthetic"):
            observed = v2.audit_cli_usage(stages)
        self.assertEqual(observed["features"]["planned"], ["-i", "-o", "-m", "-p"])

    def test_each_stage_rejects_option_missing_from_its_compiled_usage(self):
        for index in range(5):
            stages = v2.commands(Path("/synthetic"))
            name, argv, artifact = stages[index]
            stages[index] = name, [*argv, "-z", "value"], artifact
            names = [stage[0] for stage in stages]
            with self.subTest(stage=name), patch.object(v2.subprocess, "run", side_effect=[
                subprocess.CompletedProcess([], 1, b"", fake_usage(item).encode()) for item in names
            ]), patch.object(v2.base, "sha", return_value="synthetic"):
                with self.assertRaisesRegex(RuntimeError, f"compiled {name} usage"):
                    v2.audit_cli_usage(stages)

    def test_old_feature_n_would_fail_compiled_usage(self):
        stages = v2.base.commands(Path("/synthetic"))
        names = [stage[0] for stage in stages]
        with patch.object(v2.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 1, b"", fake_usage(name).encode()) for name in names
        ]), patch.object(v2.base, "sha", return_value="synthetic"):
            with self.assertRaisesRegex(RuntimeError, "compiled features usage"):
                v2.audit_cli_usage(stages)

    def test_failed_first_receipt_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "preserved -001 seal changed"):
                v2.verify_failed_first(Path(temporary))


if __name__ == "__main__":
    unittest.main()
