"""Synthetic-only mustard TRAIN photo control checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_photo_control as control


class MustardPhotoControlTest(unittest.TestCase):
    def test_train_photo_seals_reject_holdout_and_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            images = source / "images"
            images.mkdir()
            names = [f"NP3_{i * 6:03}.jpg" for i in range(48)]
            hashes = {}
            for name in names:
                content = name.encode()
                (images / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            names_file = source / "train-names.txt"
            report_file = source / "stage-report.json"
            names_file.write_text("\n".join(names) + "\n")
            names_hash = control.core.sha(names_file)
            report_file.write_text(json.dumps({"status": "complete", "train_names": names,
                                               "train_names_sha256": names_hash,
                                               "heldout_names_excluded": [f"NP3_{i + 500:03}.jpg" for i in range(12)],
                                               "train_photo_sha256": hashes}))
            with patch.object(control, "NAMES_SHA", names_hash), \
                 patch.object(control, "REPORT_SHA", control.core.sha(report_file)), \
                 patch.object(control, "jpeg_dimensions", return_value=(1280, 1024)) as dimension_probe:
                self.assertEqual(control.verify_photos(source)["names"], names)
                (images / "NP3_999.jpg").write_bytes(b"holdout")
                with self.assertRaisesRegex(RuntimeError, "unexpected files"):
                    control.verify_photos(source)
                (images / "NP3_999.jpg").unlink()
                (images / names[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "photo seal differs"):
                    control.verify_photos(source)
                (images / names[0]).write_bytes(names[0].encode())
                dimension_probe.return_value = (1279, 1024)
                with self.assertRaisesRegex(RuntimeError, "dimensions differ"):
                    control.verify_photos(source)

    def test_dimension_mismatch_blocks_focal_assumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            photo = Path(temporary) / "image.jpg"
            photo.write_bytes(b"synthetic")
            with patch.object(control.subprocess, "run", return_value=type("Result", (), {
                "stdout": "pixelWidth: 1279\npixelHeight: 1024\n"})()):
                self.assertEqual(control.jpeg_dimensions(photo), (1279, 1024))

    def test_plain_photo_command_uses_heuristic_and_no_feature_thread_flag(self):
        stages = control.commands(Path("/synthetic"))
        listing = stages[0][1]
        self.assertEqual(listing[listing.index("-f") + 1], "1536")
        self.assertNotIn("-n", stages[1][1])
        self.assertEqual(stages[3][1][-2:], ["-g", "e"])
        self.assertNotIn("-P", listing + stages[4][1])

    def test_report_rejects_partial_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "SfMReconstruction_Report.html"
            report.write_text("#views: 48<br>#poses: 24<br>#intrinsics: 1<br>"
                              "#tracks: 100<br>#residuals: 200<br>")
            self.assertEqual(control.report_counts(report)["poses"], 24)
            report.write_text("#views: 49<br>#poses: 24<br>#intrinsics: 1<br>"
                              "#tracks: 100<br>#residuals: 200<br>")
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                control.report_counts(report)


if __name__ == "__main__":
    unittest.main()
