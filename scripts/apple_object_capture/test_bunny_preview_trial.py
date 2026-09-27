from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.apple_object_capture import bunny_preview_trial as trial


class BunnyPreviewTrialTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.external = self.base / "external"
        self.external.mkdir()
        self.source = self.base / "source"
        self.source.mkdir()
        self.input = self.external / "input"
        self.output = self.external / "output"
        self.probe = self.base / "probe"
        self.probe.write_bytes(b"synthetic compiled probe")
        self.probe_hash = trial.digest(self.probe)
        hashes = {}
        mapping = {}
        for i in range(73):
            name = f"frame_{i:04}.png"
            (self.source / name).write_bytes(f"png {i}".encode())
            hashes[name] = trial.digest(self.source / name)
            mapping[name] = f"bunny_{(i + 7) % 73}_rgb.png"
        self.photos = {"count": 73, "total_photo_bytes": sum((self.source / n).stat().st_size for n in hashes),
                       "photo_sha256": hashes, "source_frame_by_photo": mapping,
                       "prepare_manifest_sha256": "a" * 64,
                       "comparison_inputs_sha256": "b" * 64}
        self.root_patch = patch.object(trial, "ROOT", self.external)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.hash_patch = patch.object(trial, "PROBE_SHA", self.probe_hash)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.photos_patch = patch.object(trial.bunny, "verify_photos", return_value=self.photos)
        self.photos_patch.start()
        self.addCleanup(self.photos_patch.stop)
        self.disk_patch = patch.object(trial.shutil, "disk_usage",
                                       return_value=types.SimpleNamespace(free=trial.FLOOR + (2 << 30)))
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def test_read_only_preflight_and_exact_stage(self):
        plan = trial.preflight(self.input, self.output, self.source, self.probe)
        self.assertEqual(plan["status"], "read_only_preflight")
        self.assertEqual(plan["limits"]["launcher_output_cap_bytes"], 512 << 20)
        self.assertFalse(self.input.exists())
        result = trial.stage_input(self.input, self.output, self.source, self.probe)
        self.assertEqual(result["status"], "staged_pending_live_review")
        self.assertEqual({p.name for p in (self.input / "images").iterdir()}, set(self.photos["photo_sha256"]))
        self.assertEqual(result["staged_photo_sha256"], self.photos["photo_sha256"])
        ready = trial.check_staged(self.input, self.output, self.source, self.probe)
        self.assertEqual(ready["status"], "staged_verified_pending_live_review")
        self.assertFalse(self.output.exists())

    def test_changed_staged_photo_rejected(self):
        trial.stage_input(self.input, self.output, self.source, self.probe)
        (self.input / "images" / "frame_0001.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "staged image bytes"):
            trial.check_staged(self.input, self.output, self.source, self.probe)

    def test_changed_photo_rejected_before_stage(self):
        (self.source / "frame_0001.png").write_bytes(b"changed")
        result = trial.stage_input(self.input, self.output, self.source, self.probe)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())

    def test_existing_output_and_low_disk_abstain(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            trial.preflight(self.input, self.output, self.source, self.probe)
        self.output.rmdir()
        with patch.object(trial.shutil, "disk_usage",
                          return_value=types.SimpleNamespace(free=trial.FLOOR)):
            with self.assertRaisesRegex(ValueError, "headroom"):
                trial.preflight(self.input, self.output, self.source, self.probe)
        self.assertFalse(self.input.exists())


if __name__ == "__main__":
    unittest.main()
