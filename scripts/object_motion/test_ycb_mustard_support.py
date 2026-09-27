"""Frozen mustard support masks: color/row geometry and failure contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import ycb_mustard_support as mustard
from scripts.object_motion import ycb_object_masks as common
from scripts.object_motion.test_ycb_object_masks import fixture


def synthetic_photo():
    photo = np.zeros((1024, 1280, 3), dtype=np.uint8)
    photo[:] = (170, 185, 220)  # cool white turntable
    photo[375:555, 550:625] = (125, 110, 45)  # mustard bottle
    photo[445:465, 556:619] = (35, 70, 170)  # blue label, warm sides remain
    photo[480:482, 550:625] = (40, 70, 160)  # two missing seed rows; bridge
    photo[400:550, 546:550] = (20, 20, 12)  # dark edge outside seed hull
    photo[330:370, 700:745] = (25, 25, 25)  # dark checker squares
    photo[340:350, 700:745] = (235, 235, 235)  # white checker squares
    return photo


class MustardSupportTests(unittest.TestCase):
    def test_label_filled_short_gap_bridged_checker_rejected(self):
        mask, stats = mustard.support_mask(synthetic_photo())
        self.assertEqual(mask[450, 580], 255)  # blue label between warm edges
        self.assertEqual(mask[480, 580], 255)  # two-row non-warm gap bridged
        self.assertEqual(mask[350, 720], 0)  # board is cool/neutral
        self.assertEqual(mask[420, 546], 0)  # dark bottle rim can be missed
        self.assertGreaterEqual(stats["bridged_rows"], 2)

    def test_roi_boundary_and_nonwarm_photos_fail(self):
        with self.assertRaisesRegex(ValueError, "row count"):
            mustard.support_mask(np.full((1024, 1280, 3), (170, 185, 220), dtype=np.uint8))
        photo = synthetic_photo()
        photo[300:375, 550:625] = (125, 110, 45)
        with self.assertRaisesRegex(ValueError, "ROI boundary"):
            mustard.support_mask(photo)

    def test_prepare_seals_48_masks_without_claiming_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package, _ = fixture(root)
            data = json.loads(package.read_text())
            data["object_id"] = mustard.OBJECT_ID
            package.write_text(json.dumps(data))
            image = Image.fromarray(synthetic_photo(), "RGB")
            with patch.dict(common.PACKAGE_SHA256, {mustard.OBJECT_ID: common.digest(package)}), \
                 patch.object(common, "MIN_FREE_BYTES", 0), \
                 patch.object(common, "verified_training_photo", return_value=image):
                result = mustard.prepare(package, root, root / "output")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mask_status"], "generated_unreviewed")
            self.assertEqual(len(result["images"]), 48)
            self.assertTrue((root / "output/training_overlay_sheet.jpg").is_file())
            self.assertLess(sum(p.stat().st_size for p in (root / "output").rglob("*") if p.is_file()),
                            common.MAX_OUTPUT_BYTES)


if __name__ == "__main__":
    unittest.main()
