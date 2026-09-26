"""Small contract tests for pose-support preparation and diagnostics."""

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from scripts.object_motion.prepare import red_scanlines, validate_records
from scripts.object_motion.diagnose import in_support
from scripts.object_motion.run import run_trial


class ObjectMotionTest(unittest.TestCase):
    def test_red_box_support_and_gray_background(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is required for image preparation")
        im = Image.new("RGB", (1280, 1024), (180, 180, 195))
        draw = ImageDraw.Draw(im)
        draw.rectangle((510, 280, 730, 590), fill=(100, 20, 35))
        draw.rectangle((560, 350, 680, 480), fill=(230, 230, 225))
        draw.rectangle((450, 600, 800, 680), fill=(20, 20, 20))
        lines = red_scanlines(im)
        record = {"pose_support_scanlines": lines}
        self.assertTrue(in_support(record, (620, 400)))
        self.assertFalse(in_support(record, (460, 640)))
        self.assertFalse(in_support(record, (900, 400)))

    def test_gapped_scanlines_do_not_shift_lookup(self):
        record = {"pose_support_scanlines": [(200, 10, 20), (203, 30, 40)]}
        self.assertFalse(in_support(record, (15, 201)))
        self.assertTrue(in_support(record, (35, 203)))
        self.assertFalse(in_support(record, (35, 204)))

    def test_rejects_photo_symlink_and_unexpected_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            photos = source / "photos"
            photos.mkdir()
            records = [{"path": f"photos/NP3_{a:03}.jpg", "turntable_angle_degrees": a}
                       for a in range(0, 360, 6)]
            for rec in records:
                (source / rec["path"]).touch()
            validate_records(source, records)
            records[0], records[1] = records[1], records[0]
            with self.assertRaises(ValueError):
                validate_records(source, records)
            records[0], records[1] = records[1], records[0]
            (photos / "NP3_000.jpg").unlink()
            try:
                (photos / "NP3_000.jpg").symlink_to(photos / "NP3_006.jpg")
            except OSError:
                self.skipTest("symlink creation is not permitted on this host")
            with self.assertRaises(ValueError):
                validate_records(source, records)

    def test_trial_no_model_is_failure_with_bounded_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fake_disk = type("Disk", (), {"free": 20 * 1024 ** 3})()
            command = [sys.executable, "-c", "import json,sys,pathlib; "
                       "p=pathlib.Path(sys.argv[1]); "
                       "(p/'summary.json').write_text(json.dumps({'model_count':0})); "
                       "print('X'*100)", str(base / "trial")]
            with patch("scripts.object_motion.run.shutil.disk_usage", return_value=fake_disk), \
                 patch("scripts.object_motion.run.LOG_CAP", 32):
                result = run_trial(command, base / "trial", 10, base)
            self.assertEqual(result["status"], "no_model")
            self.assertTrue(result["log_truncated"])
            self.assertLess((base / "trial/trial.log").stat().st_size, 150)

    def test_trial_output_cap_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fake_disk = type("Disk", (), {"free": 20 * 1024 ** 3})()
            command = [sys.executable, "-c", "import pathlib,sys,time; "
                       "(pathlib.Path(sys.argv[1])/'blob').write_bytes(b'X'*1024); "
                       "time.sleep(10)", str(base / "trial")]
            with patch("scripts.object_motion.run.shutil.disk_usage", return_value=fake_disk), \
                 patch("scripts.object_motion.run.OUTPUT_CAP", 100):
                result = run_trial(command, base / "trial", 8, base)
            self.assertEqual(result["status"], "output_limit")

    def test_two_camera_model_is_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fake_disk = type("Disk", (), {"free": 20 * 1024 ** 3})()
            command = [sys.executable, "-c", "import json,sys,pathlib; "
                       "p=pathlib.Path(sys.argv[1]); "
                       "(p/'summary.json').write_text(json.dumps({'model_count':1,'registered':2}))",
                       str(base / "trial")]
            with patch("scripts.object_motion.run.shutil.disk_usage", return_value=fake_disk):
                result = run_trial(command, base / "trial", 10, base)
            self.assertEqual(result["status"], "insufficient_sparse")


if __name__ == "__main__":
    unittest.main()
