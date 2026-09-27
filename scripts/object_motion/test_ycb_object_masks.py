"""Training-only and manual-polygon contracts for YCB object support."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.object_motion import ycb_object_masks as masks


def fixture(root):
    photos = root / "photos"
    photos.mkdir()
    rows = []
    for angle in range(0, 360, 6):
        if (angle // 6) % 5 == 4:
            continue
        name = f"NP3_{angle:03}.jpg"
        photo = photos / name
        Image.new("RGB", masks.IMAGE_SIZE, (angle % 256, angle // 2, 180)).save(photo, quality=75)
        rows.append({"angle_degrees": angle, "path": f"photos/{name}",
                     "bytes": photo.stat().st_size, "sha256": masks.digest(photo)})
    package = root / "package.json"
    package.write_text(json.dumps({"schema": "ycb_object_evaluation_package_v1",
                                   "object_id": "035_power_drill", "training_inputs": rows,
                                   "heldout_photos": [{"path": "photos/NP3_024.jpg"}],
                                   "evaluation_only": {"mesh": {"path": "absent-reference.ply"}}}))
    return package, rows


class YcbObjectMaskTests(unittest.TestCase):
    def test_contact_sheet_reads_only_training_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package, rows = fixture(root)
            with patch.dict(masks.PACKAGE_SHA256, {"035_power_drill": masks.digest(package)}), \
                 patch.object(masks, "MIN_FREE_BYTES", 0):
                result = masks.prepare(package, root, root / "contact")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mask_status"], "not_generated_pending_manual_review")
            self.assertEqual(len(result["images"]), 48)
            self.assertTrue((root / "contact/training_full_sheet.jpg").is_file())
            self.assertTrue((root / "contact/training_contact_sheet.jpg").is_file())
            self.assertFalse((root / "contact/masks").exists())
            self.assertLess(sum(path.stat().st_size for path in (root / "contact").rglob("*") if path.is_file()),
                            masks.MAX_OUTPUT_BYTES)
            rows[0]["path"] = "photos/NP3_024.jpg"
            with self.assertRaisesRegex(ValueError, "training image path"):
                masks.training_records(json.loads(package.read_text()) | {"training_inputs": rows})

    def test_manual_include_exclude_keeps_hole_and_rejects_invalid(self):
        region = {"include": [[[10, 10], [80, 10], [80, 60], [10, 60]]],
                  "exclude": [[[40, 30], [60, 30], [60, 50], [40, 50]]]}
        mask, stats = masks.render_manual(region, (100, 100))
        self.assertEqual(mask.getpixel((20, 20)), 255)
        self.assertEqual(mask.getpixel((50, 40)), 0)
        self.assertEqual(mask.getpixel((95, 95)), 0)
        self.assertGreater(stats["support_pixels"], 0)
        region["exclude"] = [[[40, 30], [101, 30], [60, 50]]]
        with self.assertRaisesRegex(ValueError, "outside photo"):
            masks.render_manual(region, (100, 100))

    def test_manual_config_must_cover_exactly_training_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package, rows = fixture(root)
            regions = root / "regions.json"
            regions.write_text(json.dumps({"schema": "ycb_manual_pose_regions_v1",
                                           "object_id": "035_power_drill", "images": {}}))
            with patch.dict(masks.PACKAGE_SHA256, {"035_power_drill": masks.digest(package)}), \
                 patch.object(masks, "MIN_FREE_BYTES", 0):
                with self.assertRaisesRegex(ValueError, "exactly cover 48"):
                    masks.prepare(package, root, root / "bad", regions_path=regions)
            self.assertFalse((root / "bad").exists())

    def test_package_pin_and_duplicate_photo_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package, rows = fixture(root)
            with self.assertRaisesRegex(ValueError, "pinned VPS-verified"):
                masks.load_package(package)
            rows[1]["sha256"] = rows[0]["sha256"]
            with self.assertRaisesRegex(ValueError, "duplicate training"):
                masks.training_records(json.loads(package.read_text()) | {"training_inputs": rows})


if __name__ == "__main__":
    unittest.main()
