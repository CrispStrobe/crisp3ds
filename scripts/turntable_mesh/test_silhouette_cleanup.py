"""Photo-only small dark holes, real openings, and all-or-nothing publication."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

try:
    import scipy  # noqa: F401
except ImportError:  # the core test job runs without the scientific stack
    raise unittest.SkipTest("needs SciPy") from None

from scripts.turntable_mesh import silhouette_cleanup as cleanup


class CleanupTests(unittest.TestCase):
    def test_small_dark_hole_filled_bright_large_boundary_preserved(self):
        rgb = np.full((150, 150, 3), 90, np.uint8)
        mask = np.ones((150, 150), bool)
        mask[20:23, 20:23] = False
        mask[30:33, 30:33] = False
        rgb[31, 31] = 129
        mask[50:100, 50:100] = False
        mask[0:5, 120:125] = False
        clean, report = cleanup.clean_holes(
            rgb, mask, dark_object_bright_background=True
        )
        expected = mask.copy()
        expected[20:23, 20:23] = True
        np.testing.assert_array_equal(clean, expected)
        self.assertEqual(report["filled_pixels"], 9)
        self.assertFalse(clean[31, 31])
        self.assertFalse(clean[70, 70])
        self.assertFalse(clean[2, 122])
        np.testing.assert_array_equal(mask[20:23, 20:23], False)

    def test_threshold_equality_and_total_budget_fail_closed(self):
        rgb = np.full((100, 100, 3), 128, np.uint8)
        mask = np.ones((100, 100), bool)
        mask[20:25, 20:25] = False
        clean, report = cleanup.clean_holes(
            rgb, mask, dark_object_bright_background=True
        )
        self.assertEqual(report["filled_pixels"], 25)
        self.assertTrue(clean[22, 22])
        mask[40:50, 40:50] = False
        with self.assertRaisesRegex(ValueError, "0.01 fraction"):
            cleanup.clean_holes(rgb, mask, dark_object_bright_background=True)
        with self.assertRaisesRegex(ValueError, "explicit"):
            cleanup.clean_holes(rgb, mask)

    def test_explicit_budget_bounds_and_bright_hole_preservation(self):
        rgb = np.full((100, 100, 3), 90, np.uint8)
        mask = np.ones((100, 100), bool)
        mask[20:32, 20:32] = False
        mask[50:55, 50:55] = False
        rgb[52, 52] = 200
        with self.assertRaisesRegex(ValueError, "0.01 fraction"):
            cleanup.clean_holes(rgb, mask, dark_object_bright_background=True)
        clean, _ = cleanup.clean_holes(
            rgb,
            mask,
            dark_object_bright_background=True,
            maximum_total_filled_foreground_fraction=0.05,
        )
        self.assertTrue(clean[22, 22])
        self.assertFalse(clean[52, 52])
        for invalid in (True, float("nan"), float("inf"), -0.01, 0.051):
            with self.assertRaisesRegex(ValueError, "finite"):
                cleanup.clean_holes(
                    rgb,
                    mask,
                    dark_object_bright_background=True,
                    maximum_total_filled_foreground_fraction=invalid,
                )

    def test_run_failure_publishes_no_masks_and_success_binds_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, masks = root / "images", root / "masks"
            images.mkdir()
            masks.mkdir()
            for i in range(2):
                Image.new("RGB", (100, 100), (90, 90, 90)).save(images / f"{i}.png")
                mask = np.ones((100, 100), np.uint8) * 255
                mask[20:25, 20:25] = 0
                if i:
                    mask[40:50, 40:50] = 0
                Image.fromarray(mask).save(masks / f"{i}.png.png")
            out = root / "failed"
            with self.assertRaisesRegex(ValueError, "0.01 fraction"):
                cleanup.run(images, masks, out, dark_object_bright_background=True)
            self.assertFalse((out / "masks").exists())
            self.assertEqual(
                json.loads((out / "result.json").read_text())["status"], "failed"
            )
            accepted = cleanup.run(
                images,
                masks,
                root / "explicit-budget",
                dark_object_bright_background=True,
                maximum_total_filled_foreground_fraction=0.02,
            )
            self.assertEqual(
                accepted["configuration"]["maximum_total_filled_foreground_fraction"],
                0.02,
            )
            self.assertEqual(accepted["rows"][1]["filled_pixels"], 125)
            Image.open(masks / "0.png.png").save(masks / "1.png.png")
            report = cleanup.run(
                images, masks, root / "success", dark_object_bright_background=True
            )
            self.assertEqual(
                report["source_hashes_before"], report["source_hashes_after"]
            )
            self.assertEqual(len(report["output_hashes"]), 2)
