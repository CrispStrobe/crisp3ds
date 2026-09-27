import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.classical_backend import cracker_mask_qa as qa


class MaskMathTests(unittest.TestCase):
    def test_excess_omission_iou_and_exact_boundary_distance(self):
        reviewed = np.zeros((15, 15), dtype=bool)
        reviewed[3:11, 3:11] = True
        old = reviewed.copy()
        old[3:11, 11] = True
        old[3:11, 3] = False
        values = qa.compare(old, reviewed)
        self.assertEqual(values["old_excess_fraction_of_reviewed"], 8 / 64)
        self.assertEqual(values["old_omission_fraction_of_reviewed"], 8 / 64)
        self.assertEqual(values["iou"], 56 / 72)
        self.assertAlmostEqual(values["boundary_px"]["a_to_b"]["max_px"], 1.0)
        self.assertAlmostEqual(values["boundary_px"]["b_to_a"]["max_px"], 1.0)

    def test_sanity_finds_component_hole_and_border(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True
        self.assertTrue(qa.mask_sanity(mask)["passes"])
        mask[5, 5] = False
        self.assertEqual(qa.mask_sanity(mask)["holes"], 1)
        mask[0, 0] = True
        result = qa.mask_sanity(mask)
        self.assertEqual(result["components"], 2)
        self.assertEqual(result["border_pixels"], 1)
        self.assertFalse(result["passes"])


class ContractTests(unittest.TestCase):
    NAMES = ("NP3_000.jpg", "NP3_006.jpg")
    REPEATS = ("NP3_000.jpg",)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.rgb = self.source / "dense" / "images"
        self.old = self.source / "masks"
        self.reviewed = root / "reviewed"
        self.repeat = root / "repeat"
        for folder in (self.rgb, self.old, self.reviewed, self.repeat):
            folder.mkdir(parents=True)
        self.good = np.zeros((14, 16), dtype=np.uint8)
        self.good[3:11, 4:12] = 255
        rows = []
        for name in self.NAMES:
            rgb_path = self.rgb / name
            Image.fromarray(np.zeros((14, 16, 3), dtype=np.uint8), "RGB").save(rgb_path)
            old_path = self.old / qa.mask_name(name)
            Image.fromarray(self.good, "L").save(old_path)
            Image.fromarray(self.good, "L").save(self.reviewed / qa.mask_name(name))
            rows.append({"name": name, "native_mask_name": qa.mask_name(name),
                         "undistorted_size": [16, 14],
                         "undistorted_image_sha256": qa.digest(rgb_path),
                         "mask_sha256": qa.digest(old_path)})
        Image.fromarray(self.good, "L").save(self.repeat / qa.mask_name(self.REPEATS[0]))
        (self.old / "report.json").write_text(json.dumps({
            "schema": "classical_dense_masks_v1", "status": "complete",
            "semantics": "photo-derived coarse pose support, not a silhouette or ground truth",
            "images": rows}))

    def audit(self):
        return qa.audit(self.source, self.reviewed, self.repeat,
                        expected_names=self.NAMES, repeat_names=self.REPEATS,
                        expected_report_hash=None)

    def test_exact_inventory_and_repeatability_pass(self):
        result = self.audit()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["repeat_count"], 1)
        self.assertEqual(result["images"][0]["old_vs_reviewed"]["iou"], 1.0)
        self.assertEqual(result["summary"]["old_new_iou"]["worst"], 1.0)
        self.assertEqual(result["summary"]["failed_view_names"], [])
        extra = self.reviewed / "extra.mask.png"
        Image.fromarray(self.good, "L").save(extra)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.audit()

    def test_nonbinary_dimension_and_old_hash_fail_closed(self):
        path = self.reviewed / qa.mask_name(self.NAMES[0])
        bad = self.good.copy()
        bad[3, 4] = 128
        Image.fromarray(bad, "L").save(path)
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            self.audit()
        Image.fromarray(self.good[:, :-1], "L").save(path)
        with self.assertRaisesRegex(ValueError, "native-size"):
            self.audit()
        Image.fromarray(self.good, "L").save(path)
        Image.fromarray(np.zeros_like(self.good), "L").save(self.old / qa.mask_name(self.NAMES[0]))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.audit()

    def test_repeat_boundary_over_three_pixels_abstains(self):
        shifted = np.zeros_like(self.good)
        shifted[3:11, 8:14] = 255
        Image.fromarray(shifted, "L").save(self.repeat / qa.mask_name(self.REPEATS[0]))
        result = self.audit()
        self.assertEqual(result["status"], "abstain")
        self.assertFalse(result["images"][0]["repeat"]["passes_3px_p95"])
        self.assertTrue(result["images"][0]["repeat"]["sanity"]["passes"])

    def test_reviewed_hole_abstains_with_named_failure(self):
        path = self.reviewed / qa.mask_name(self.NAMES[1])
        holed = self.good.copy()
        holed[6, 7] = 0
        Image.fromarray(holed, "L").save(path)
        result = self.audit()
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["images"][1]["reviewed_sanity"]["holes"], 1)
        self.assertEqual(result["summary"]["failed_view_names"], [self.NAMES[1]])


if __name__ == "__main__":
    unittest.main()
