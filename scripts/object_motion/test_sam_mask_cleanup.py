import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.object_motion import sam_mask_cleanup as cleanup


class SAMMaskCleanupTests(unittest.TestCase):
    def test_detached_speck_removed_thin_connected_protrusion_preserved(self):
        raw = np.zeros((1024, 1280), dtype=np.uint8)
        raw[350:550, 530:650] = 255
        raw[330:351, 590] = 255  # thin connected cap/stem must remain
        raw[300:303, 700:703] = 255  # detached background speck
        clean, stats = cleanup.largest_component(raw)
        self.assertTrue(np.all(clean[330:351, 590] == 255))
        self.assertTrue(np.all(clean[300:303, 700:703] == 0))
        self.assertEqual(stats["removed_pixels"], 9)
        self.assertEqual(stats["raw_components_8_connected"], 2)
        self.assertEqual(stats["removed_component_sizes"], [9])

    def test_empty_tie_large_removed_and_nonbinary_rejected(self):
        raw = np.zeros((1024, 1280), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "empty"):
            cleanup.largest_component(raw)
        raw[10:20, 10:20] = 255
        raw[30:40, 30:40] = 255
        with self.assertRaisesRegex(ValueError, "equal-size"):
            cleanup.largest_component(raw)
        raw[30:40, 30:41] = 255
        with self.assertRaisesRegex(ValueError, "10%"):
            cleanup.largest_component(raw)
        raw[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            cleanup.largest_component(raw)

    def test_raw_manifest_pin_and_fresh_output_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "reviewed input"):
                cleanup.validated_raw_manifest(path, "f" * 64)
            path.unlink()
            path.symlink_to(Path(temp) / "missing")
            with self.assertRaisesRegex(ValueError, "regular"):
                cleanup.validated_raw_manifest(path, "f" * 64)


if __name__ == "__main__":
    unittest.main()
