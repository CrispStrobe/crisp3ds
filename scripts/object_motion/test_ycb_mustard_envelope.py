"""Label-preserving row-envelope successor, without reconstruction or scores."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import ycb_mustard_envelope as envelope
from scripts.object_motion import ycb_object_masks as common
from scripts.object_motion.test_ycb_mustard_support import synthetic_photo
from scripts.object_motion.test_ycb_object_masks import fixture


class MustardEnvelopeTests(unittest.TestCase):
    def test_full_label_band_and_protrusion_preserved(self):
        photo = synthetic_photo()
        photo[435:470, 540:635] = (35, 70, 170)  # no warm seeds across 35-row blue band
        photo[445:455, 540:635] = (150, 20, 50)  # red printed strip also non-warm
        mask, stats = envelope.support_mask(photo)
        self.assertEqual(mask[450, 580], 255)
        self.assertEqual(mask[440, 542], 255)  # label extends beyond warm side
        self.assertEqual(mask[350, 720], 0)  # neutral checkerboard rejected
        self.assertGreaterEqual(stats["envelope_rows"] - stats["warm_seed_rows"], 35)
        self.assertLessEqual(stats["expansion_ratio"], envelope.MAX_EXPANSION)

    def test_long_gap_and_boundary_leak_rejected(self):
        photo = synthetic_photo()
        photo[420:490, 550:625] = (35, 70, 170)
        with self.assertRaisesRegex(ValueError, "unsupported row gap"):
            envelope.support_mask(photo)
        photo = synthetic_photo()
        photo[370:450, 755:759] = (125, 110, 45)
        with self.assertRaisesRegex(ValueError, "ROI boundary"):
            envelope.support_mask(photo)

    def test_prepare_keeps_predecessor_separate_and_unreviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package, _ = fixture(root)
            data = json.loads(package.read_text())
            data["object_id"] = "006_mustard_bottle"
            package.write_text(json.dumps(data))
            image = Image.fromarray(synthetic_photo(), "RGB")
            with patch.dict(common.PACKAGE_SHA256, {"006_mustard_bottle": common.digest(package)}), \
                 patch.object(common, "MIN_FREE_BYTES", 0), \
                 patch.object(common, "verified_training_photo", return_value=image):
                result = envelope.prepare(package, root, root / "successor")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mask_status"], "generated_unreviewed")
            self.assertEqual(len(result["images"]), 48)
            self.assertIn("predecessor_helper_sha256", result)
            self.assertTrue((root / "successor/training_masked_sheet.jpg").is_file())
            self.assertLess(sum(p.stat().st_size for p in (root / "successor").rglob("*") if p.is_file()),
                            common.MAX_OUTPUT_BYTES)


if __name__ == "__main__":
    unittest.main()
