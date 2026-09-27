"""Small synthetic contracts for the photo-only bunny mask producer."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.object_motion import bunny_masks


class BunnyMaskTests(unittest.TestCase):
    def setUp(self):
        tmp_root = bunny_masks.ROOT / ".local-tools/tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_segment_largest_component_preserves_hole_and_rejects_pedestal(self):
        gray = np.full((200, 300), 220, np.uint8)
        gray[55:155, 110:230] = 70
        gray[90:110, 150:170] = 220
        gray[193:, :] = 30  # dark turntable is below the frozen lower guard
        gray[70:80, 55:65] = 20  # disconnected dark spot is not bunny
        mask, info = bunny_masks.segment(gray)
        self.assertEqual(mask[75, 120], 255)
        self.assertEqual(mask[100, 160], 0)
        self.assertEqual(mask[197, 160], 0)
        self.assertEqual(mask[75, 60], 0)
        self.assertEqual(info["foreground_pixels"], 12000 - 400)

    def test_segment_rejects_missing_or_wrong_type(self):
        with self.assertRaisesRegex(ValueError, "single-channel"):
            bunny_masks.segment(np.zeros((128, 128, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "no dark"):
            bunny_masks.segment(np.full((128, 128), 220, np.uint8))

    def test_prepare_fresh_binary_masks_and_hash_bound_manifest(self):
        photos = self.base / "photos"
        photos.mkdir()
        for index in range(2):
            gray = np.full((128, 128), 220, np.uint8)
            gray[40:100, 40:90] = 50 + index
            self.assertTrue(cv2.imwrite(str(photos / f"frame_{index:04}.png"), gray))
        output = self.base / "output"
        with patch.object(bunny_masks, "EXPECTED_COUNT", 2):
            report = bunny_masks.prepare(photos, output)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["images"]), 2)
        for item in report["images"]:
            mask_path = output / "masks" / f'{item["name"]}.png'
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (128, 128))
            self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
            self.assertEqual(hashlib.sha256(mask_path.read_bytes()).hexdigest(), item["mask_sha256"])
        with patch.object(bunny_masks, "EXPECTED_COUNT", 2):
            with self.assertRaises(FileExistsError):
                bunny_masks.prepare(photos, output)


if __name__ == "__main__":
    unittest.main()
