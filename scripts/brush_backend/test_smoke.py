from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.brush_backend import smoke
from scripts.brush_backend.ply import REQUIRED, validate as validate_ply


def valid_ply() -> bytes:
    props = sorted(REQUIRED)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in props) + "end_header\n"
    values = [1.0 if name == "rot_0" else 0.25 for name in props]
    return header.encode() + struct.pack("<" + "f" * len(props), *values)


class SmokeTests(unittest.TestCase):
    def test_fixed_20_step_no_eval_command(self):
        argv = smoke.command(Path("/binary"), Path("/photos"), Path("/fresh"))
        self.assertEqual(argv[argv.index("--total-steps") + 1], "20")
        self.assertEqual(argv[argv.index("--seed") + 1], "42")
        self.assertEqual(argv[argv.index("--max-splats") + 1], "50000")
        self.assertNotIn("--eval-split-every", argv)
        self.assertNotIn("--with-viewer", argv)

    def test_folder_bytes_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file").write_bytes(b"abc")
            self.assertEqual(smoke.folder_bytes(root), 3)
            (root / "link").symlink_to(root / "file")
            with self.assertRaisesRegex(ValueError, "symlink"):
                smoke.folder_bytes(root)

    def _fake_run(self, root: Path, script: str, *, deadline: float,
                  output_limit: int | None = None, rss: int = 1024) -> dict:
        output = root / "output"
        (root / ".local-tools/tmp").mkdir(parents=True)
        fake = {
            "binary_sha256": "c" * 64,
            "dataset": {"image_sha256": {"view.jpg": "a" * 64},
                        "model_file_sha256": {"images.bin": "b" * 64}},
            "command": [sys.executable, "-c", script],
        }
        after = {"image_sha256": fake["dataset"]["image_sha256"],
                 "model_file_sha256": fake["dataset"]["model_file_sha256"]}
        with patch.object(smoke, "ROOT", root), patch.object(smoke, "MOUNT", root), \
             patch.object(smoke, "OUTPUT", output), patch.object(smoke, "DATASET", root), \
             patch.object(smoke, "MAX_SECONDS", deadline), patch.object(smoke, "MIN_FREE", 0), \
             patch.object(smoke, "MAX_OUTPUT", output_limit or smoke.MAX_OUTPUT), \
             patch.object(smoke, "preflight", return_value=fake), \
             patch.object(smoke, "validate_dataset", return_value=after), \
             patch.object(smoke, "validate_008", return_value={}), \
             patch.object(smoke, "sha256", return_value="c" * 64), \
             patch.object(smoke, "rss_bytes", return_value=rss):
            return smoke.run()

    def test_tiny_fake_export_completes_without_native_brush(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = f"from pathlib import Path; Path('export_20.ply').write_bytes({valid_ply()!r})"
            report = self._fake_run(Path(tmp), script, deadline=5)
            self.assertEqual(report["status"], "complete")
            self.assertIn("export_20.ply", report["exports"])
            self.assertFalse(report["gpu_unified_memory_measured"])

    def test_ply_rejects_truncation_and_nonfinite_gaussian(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export_20.ply"
            payload = valid_ply()
            path.write_bytes(payload)
            self.assertEqual(validate_ply(path)["vertices"], 1)
            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, "payload"):
                validate_ply(path)
            path.write_bytes(payload.replace(struct.pack("<f", 0.25), struct.pack("<f", float("nan")), 1))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                validate_ply(path)

    @unittest.skipIf(sys.platform == "win32", "process-group kill is macOS-only")
    def test_timeout_kills_fake_process_and_preserves_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._fake_run(Path(tmp), "import time; time.sleep(30)", deadline=0.05)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"], "timeout")
            self.assertTrue((Path(tmp) / "output/report.json").is_file())

    def test_output_cap_rejects_even_fast_finished_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = f"from pathlib import Path; Path('export_20.ply').write_bytes({valid_ply()!r})"
            report = self._fake_run(Path(tmp), script, deadline=5, output_limit=100)
            self.assertEqual(report["status"], "failed")
            self.assertIn("byte cap", report["failure"])

    @unittest.skipIf(sys.platform == "win32", "process-group kill is macOS-only")
    def test_rss_cap_stops_fake_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._fake_run(Path(tmp), "import time; time.sleep(30)",
                                    deadline=5, rss=smoke.MAX_RSS + 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"], "sampled process RSS cap")


if __name__ == "__main__":
    unittest.main()
