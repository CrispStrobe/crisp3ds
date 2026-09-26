"""Check that fixed photo profiles are deterministic and source preserving."""

import tempfile
from pathlib import Path
import unittest

from PIL import Image

from scripts.mve_full.prepare import digest, prepare, transform


class PrepareTests(unittest.TestCase):
    def test_gamma_lifts_dark_values_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (3, 1))
            image.putdata([(0, 0, 0), (20, 20, 20), (255, 255, 255)])
            source = root / "source.png"
            image.save(source)
            before = digest(source)
            values = list(transform(image, "gamma05").getdata())
            self.assertEqual(values[0], (0, 0, 0))
            self.assertEqual(values[2], (255, 255, 255))
            self.assertGreater(values[1][0], 20)
            self.assertEqual(digest(source), before)

    def test_clahe_is_deterministic_when_available(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 unavailable in this Python; gamma profile remains available")
        image = Image.new("RGB", (32, 32))
        image.putdata([(20 + x % 8, 18 + x % 8, 15 + x % 8) for x in range(32 * 32)])
        one = transform(image, "clahe2")
        two = transform(image, "clahe2")
        self.assertEqual(one.tobytes(), two.tobytes())
        self.assertNotEqual(one.tobytes(), image.tobytes())

    def test_manifest_matches_immutable_sources_and_png_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "originals"
            sources.mkdir()
            for index in range(3):
                Image.new("RGB", (4, 4), (index + 20, index + 25, index + 30)).save(sources / f"{index}.png")
            before = {path.name: digest(path) for path in sources.iterdir()}
            output = root / "prepared"
            report = prepare(sources, output, "gamma05", 3, max_gib=0.001)
            self.assertEqual(report["status"], "succeeded")
            self.assertEqual(len(report["images"]), 3)
            for entry in report["images"]:
                self.assertEqual(entry["source_sha256"], before[entry["source"]])
                self.assertEqual(digest(sources / entry["source"]), before[entry["source"]])
                self.assertEqual(entry["output_sha256"], digest(output / entry["output"]))


if __name__ == "__main__":
    unittest.main()
