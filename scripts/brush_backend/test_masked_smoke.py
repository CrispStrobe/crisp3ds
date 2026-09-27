import json
from pathlib import Path
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.brush_backend import masked_smoke
from scripts.classical_backend import run as classical_run


class MaskedSmokeTests(unittest.TestCase):
    def test_sealed_bridge_preflight_rejects_tampered_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bridge-report.json").write_text('{"status":"complete"}')
            with patch.object(masked_smoke, "BRIDGE", root):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    masked_smoke.verify_input()

    def test_stage_success_requires_finite_splat_and_rehashed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            identity = {"bridge_report_sha256": "a" * 64, "files": 123,
                        "prepared_dataset": "sealed", "source_mask_report_sha256": "b" * 64}
            plan = {"status": "preflight", "command": ["fake-brush"], "inputs": identity,
                    "binary_sha256": masked_smoke.BINARY_SHA256}

            def fake_stage(run, *_args, **_kwargs):
                (run / "export_20.ply").write_bytes(b"ply")
                return {"status": "complete", "seconds": 0.1, "peak_sampled_child_rss_bytes": 1024}

            with patch.object(masked_smoke, "OUTPUT", output), \
                 patch.object(masked_smoke, "MOUNT", Path(tmp)), \
                 patch.object(masked_smoke, "preflight", return_value=plan), \
                 patch.object(masked_smoke, "stage", side_effect=fake_stage), \
                 patch.object(masked_smoke, "verify_input", return_value=identity), \
                 patch.object(masked_smoke, "sha256", return_value=masked_smoke.BINARY_SHA256), \
                 patch.object(masked_smoke, "validate_splat_ply", return_value={"vertices": 1}), \
                 patch.object(masked_smoke, "RESERVE", 0):
                result = masked_smoke.run()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["exports"]["export_20.ply"]["vertices"], 1)
            self.assertFalse(result["quality_claim"])

    def test_stage_failure_preserves_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            identity = {"bridge_report_sha256": "a" * 64, "files": 123,
                        "prepared_dataset": "sealed", "source_mask_report_sha256": "b" * 64}
            plan = {"status": "preflight", "command": ["fake-brush"], "inputs": identity}
            failed = {"name": "train", "status": "failed", "failure": "deadline exceeded", "exit_code": -15}
            with patch.object(masked_smoke, "OUTPUT", output), \
                 patch.object(masked_smoke, "MOUNT", Path(tmp)), \
                 patch.object(masked_smoke, "preflight", return_value=plan), \
                 patch.object(masked_smoke, "stage", side_effect=masked_smoke.StageError(failed)), \
                 patch.object(masked_smoke, "verify_input", return_value=identity), \
                 patch.object(masked_smoke, "sha256", return_value=masked_smoke.BINARY_SHA256), \
                 patch.object(masked_smoke, "RESERVE", 0):
                result = masked_smoke.run()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(json.loads((output / "report.json").read_text())["stage"]["failure"], "deadline exceeded")

    def test_shared_stage_extra_internal_reserve_stops_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, internal = root / "run", root / "internal"
            output.mkdir()
            internal.mkdir()

            def disk_usage(path):
                return SimpleNamespace(free=0 if Path(path) == internal else 1 << 50)

            with patch.object(classical_run.shutil, "disk_usage", side_effect=disk_usage):
                with self.assertRaises(classical_run.StageError) as error:
                    classical_run.stage(output, "guard", [sys.executable, "-c", "import time; time.sleep(30)"],
                                        time.monotonic() + 5, 1 << 20, 1 << 20, 1 << 30,
                                        extra_reserve_paths=(internal,))
            self.assertIn("extra 10 GiB", error.exception.result["failure"])


if __name__ == "__main__":
    unittest.main()
