"""Synthetic contract checks; never launches OpenMVG or reads the photo set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_photo_control as control


class PhotoControlTest(unittest.TestCase):
    def test_sealed_input_rejects_extra_image_and_changed_photo(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample = Path(temporary)
            images = sample / "images"
            images.mkdir()
            files = []
            for name in control.NAMES:
                content = name.encode()
                (images / name).write_bytes(content)
                files.append({"path": f"images/{name}", "bytes": len(content),
                              "sha256": hashlib.sha256(content).hexdigest()})
            manifest = sample / "manifest.json"
            manifest.write_text(json.dumps({"files": files}))
            with patch.object(control, "SAMPLE_MANIFEST_SHA", control.sha(manifest)):
                self.assertEqual(len(control.verify_photos(sample)["photos"]), 11)
                (images / "reference_pose.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    control.verify_photos(sample)
                (images / "reference_pose.json").unlink()
                (images / "00000.jpg").write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "seal mismatch"):
                    control.verify_photos(sample)

    def test_fifth_binary_receipt_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / "license-closure-receipt.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "geometric-filter-license-receipt"):
                control.verify_binaries(build)

    def test_changed_fifth_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / "license-closure-receipt.json").write_text("{}")
            (build / "geometric-filter-license-receipt.json").write_text("{}")
            (build / "geometric-filter-build-manifest.json").write_text("{}")
            with patch.object(control, "FOUR_RECEIPT_SHA", control.sha(build / "license-closure-receipt.json")):
                with self.assertRaisesRegex(RuntimeError, "manifest/receipt seal mismatch"):
                    control.verify_binaries(build)

    def test_disk_reserve_includes_full_output_allocation(self):
        with patch.object(control.base, "tree_bytes", return_value=0), \
             patch.object(control.base, "free_bytes", side_effect=[control.FLOOR + control.MARGIN, control.FLOOR + control.MARGIN]):
            with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                control.capacity(Path("/synthetic"), prospective=True)

    def test_command_contract_filters_before_sfm(self):
        with tempfile.TemporaryDirectory() as temporary:
            stages = control.commands(Path(temporary))
        self.assertEqual([stage[0] for stage in stages],
                         ["listing", "features", "putative", "geometric", "sfm"])
        self.assertEqual(stages[2][1][-2:], ["-r", "0.8"])
        self.assertEqual(stages[3][1][-2:], ["-g", "e"])
        self.assertEqual(stages[4][1][stages[4][1].index("-M") + 1], "matches.e.bin")
        self.assertNotIn("-P", stages[0][1] + stages[4][1])

    def test_sfm_report_counts_and_inventory_no_self_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = output / "SfMReconstruction_Report.html"
            report.write_text(" #views: 11<br> #poses: 11<br> #intrinsics: 1<br>"
                              " #tracks: 99<br> #residuals: 250<br>")
            self.assertEqual(control.report_counts(report)["poses"], 11)
            (output / "receipt.json").write_text("temporary receipt")
            inventory = control.output_inventory(output)
            self.assertIn(report.name, inventory)
            self.assertNotIn("receipt.json", inventory)
            report.write_text(" #views: 11<br> #poses: 12<br> #intrinsics: 1<br>"
                              " #tracks: 99<br> #residuals: 250<br>")
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                control.report_counts(report)


if __name__ == "__main__":
    unittest.main()
