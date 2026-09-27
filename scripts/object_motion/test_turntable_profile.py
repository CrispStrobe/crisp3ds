"""Unit checks for the fresh turntable sparse wrapper; never start native SfM."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.object_motion import turntable_profile as profile


class TurntableProfileTest(unittest.TestCase):
    def test_input_seal_rejects_changed_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / "photo.jpg"
            mask = root / "mask.png"
            source = root / "source.json"
            manifest = root / "manifest.json"
            photo.write_bytes(b"original")
            mask.write_bytes(b"mask")
            source.write_bytes(b"source")
            record = {"name": "photo.jpg", "path": str(photo), "sha256": profile.digest(photo),
                      "pose_support_mask": str(mask), "mask_sha256": profile.digest(mask)}
            manifest.write_text(json.dumps({"source_manifest_sha256": profile.digest(source), "images": [record]}))
            with patch.object(profile.run, "validate", return_value=[record]):
                self.assertEqual(profile.input_seal(manifest)["photos"]["photo.jpg"], profile.digest(photo))
                photo.write_bytes(b"changed")
                with patch.object(profile.run, "validate", side_effect=ValueError("changed photo")):
                    with self.assertRaisesRegex(ValueError, "changed photo"):
                        profile.input_seal(manifest)

    def test_output_must_be_fresh_on_external_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "fresh"
            with patch.object(profile, "is_external_filesystem", return_value=True):
                self.assertEqual(profile.validate_output(dest, root), dest.resolve())
            dest.mkdir()
            with patch.object(profile, "is_external_filesystem", return_value=True):
                with self.assertRaises(FileExistsError):
                    profile.validate_output(dest, root)

    def test_supervisor_marks_worker_failure_and_caps_log(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "output"
            command = [profile.sys.executable, "-c", "import sys; print('X'*1000); sys.exit(7)"]
            with patch.object(profile, "LOG_CAP", 32), patch.object(profile, "FREE_FLOOR", 0):
                result = profile.supervise(command, dest, 10)
            self.assertEqual(result["status"], "worker_failed")
            self.assertEqual(result["returncode"], 7)
            self.assertTrue(result["log_truncated"])
            self.assertLess((dest / "worker.log").stat().st_size, 120)

    def test_model_hashes_reject_tamper_and_foreign_path(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "output"
            model = dest / "models" / "0"
            model.mkdir(parents=True)
            names = ("cameras.bin", "images.bin", "points3D.bin")
            for name in names:
                (model / name).write_bytes(name.encode())
            summary = {"model_dir": str(model),
                       "model_files_sha256": {name: profile.digest(model / name) for name in names}}
            self.assertTrue(profile.verify_model_hashes(summary, dest))
            (model / "images.bin").write_bytes(b"changed")
            self.assertFalse(profile.verify_model_hashes(summary, dest))
            summary["model_dir"] = str(Path(temp))
            self.assertFalse(profile.verify_model_hashes(summary, dest))


if __name__ == "__main__":
    unittest.main()
