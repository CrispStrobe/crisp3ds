import unittest

import numpy as np
from PIL import Image

from scripts.object_motion import sam_mask_full as full


class SAMFullTests(unittest.TestCase):
    def test_exact_frozen_three_batch_partition(self):
        angles = [a for a in range(0, 360, 6) if (a // 6) % 5 != 4]
        package = {"object_id": full.smoke.OBJECT_ID, "schema": "ycb_object_evaluation_package_v1",
                   "training_inputs": [{"angle_degrees": a, "path": f"photos/NP3_{a:03}.jpg",
                                        "bytes": 1000, "sha256": f"{i:064x}"}
                                       for i, a in enumerate(angles)]}
        rows = full.exact_training_rows(package)
        self.assertEqual(len(rows), 48)
        self.assertEqual([len(rows[i:i + full.BATCH_SIZE]) for i in (0, 16, 32)], [16, 16, 16])
        self.assertEqual(rows[0]["path"], "photos/NP3_000.jpg")
        self.assertEqual(rows[-1]["path"], "photos/NP3_348.jpg")
        package["training_inputs"].pop()
        with self.assertRaisesRegex(ValueError, "48"):
            full.exact_training_rows(package)

    def test_fake_predictor_preserves_connected_thin_pixels(self):
        image = Image.new("RGB", (1280, 1024), "white")
        raw = np.zeros((1024, 1280), dtype=np.float32)
        raw[350:550, 530:650] = 1
        raw[340:351, 590] = 1
        raw[300:302, 700:702] = 1
        def predict(photo, box):
            self.assertEqual(photo.size, (1280, 1024))
            self.assertEqual(box, full.smoke.BOX)
            return raw[None]
        before, after, stats = full.segment_one(image, predict)
        self.assertEqual(stats["removed_pixels"], 4)
        self.assertEqual(np.asarray(before)[300, 700], 255)
        self.assertEqual(np.asarray(after)[300, 700], 0)
        self.assertEqual(np.asarray(after)[340, 590], 255)


if __name__ == "__main__":
    unittest.main()
