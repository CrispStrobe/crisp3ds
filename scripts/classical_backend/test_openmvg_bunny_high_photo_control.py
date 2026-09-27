"""Synthetic contract tests; no real photo or native SfM execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_bunny_high_photo_control as control


def png_stub(content: bytes) -> bytes:
    return (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" +
            struct.pack(">II", control.WIDTH, control.HEIGHT) + content)


class BunnyHighPhotoControlTest(unittest.TestCase):
    def test_exact_input_set_and_hash_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "photos"
            source.mkdir()
            comparison = Path(temporary) / "inputs.json"
            rows, other = [], []
            for index, name in enumerate(control.NAMES):
                data = png_stub(name.encode())
                path = source / name
                path.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                rows.append({"output": name, "output_sha256": digest,
                             "source": f"bunny_{index}_rgb.png",
                             "width": control.WIDTH, "height": control.HEIGHT})
                other.append({"name": name, "source": str(path), "sha256": digest,
                              "bytes": len(data)})
            manifest = source / "prepare-manifest.json"
            manifest.write_text(json.dumps({"schema": "mve_image_only_preprocess_v1",
                                            "profile": "gamma05_clahe2", "status": "succeeded",
                                            "images": rows}))
            comparison.write_text(json.dumps(other))
            with patch.object(control, "PREPARE_SHA", control.core.sha(manifest)), \
                 patch.object(control, "COMPARISON_SHA", control.core.sha(comparison)):
                self.assertEqual(control.verify_photos(source, comparison)["count"], 73)
                (source / "scanner.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    control.verify_photos(source, comparison)
                (source / "scanner.json").unlink()
                (source / control.NAMES[0]).write_bytes(png_stub(b"changed"))
                with self.assertRaisesRegex(RuntimeError, "photo seal"):
                    control.verify_photos(source, comparison)

    def test_high_photo_only_commands_and_heuristic_focal(self):
        stages = control.commands(Path("/synthetic"))
        listing, features = stages[0][1], stages[1][1]
        self.assertEqual(listing[listing.index("-f") + 1], "2098.8")
        self.assertEqual(features[features.index("-p") + 1], "HIGH")
        self.assertNotIn("-n", features)
        self.assertEqual(stages[3][1][-2:], ["-g", "e"])
        self.assertTrue(all("pose" not in " ".join(command).lower()
                            for _, command, _ in stages))

    def test_report_requires_73_views_and_does_not_infer_geometry_from_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.html"
            report.write_text("#views: 73<br>#poses: 73<br>#intrinsics: 1<br>"
                              "#tracks: 100<br>#residuals: 200<br>")
            self.assertEqual(control.report_counts(report)["poses"], 73)
            report.write_text(report.read_text().replace("#views: 73", "#views: 72"))
            with self.assertRaisesRegex(RuntimeError, "view count"):
                control.report_counts(report)

    def test_preflight_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already-exists"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                control.preflight(output)


if __name__ == "__main__":
    unittest.main()
