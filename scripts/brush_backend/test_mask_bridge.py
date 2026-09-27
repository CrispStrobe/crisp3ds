import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.brush_backend import mask_bridge
from scripts.brush_backend.preflight import sha256


class MaskBridgeTests(unittest.TestCase):
    def test_v030_exact_stem_not_openmvs_dot_mask_stem(self):
        self.assertEqual(mask_bridge.release_mask_name("NP3_000.jpg"), "NP3_000.png")
        self.assertNotEqual(mask_bridge.release_mask_name("NP3_000.jpg"), "NP3_000.mask.png")
        with self.assertRaises(ValueError):
            mask_bridge.release_mask_name("../outside.jpg")
        with self.assertRaisesRegex(ValueError, "collision"):
            mask_bridge.unique_release_mask_names(["same.jpg", "same.png"])

    def test_mask_requires_matching_dimensions_and_binary_grayscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, mask = root / "view.jpg", root / "view.png"
            Image.new("RGB", (4, 2), (10, 20, 30)).save(image)
            alpha = Image.new("L", (4, 2), 0)
            alpha.putpixel((1, 1), 255)
            alpha.save(mask)
            self.assertEqual(mask_bridge.validate_mask(mask, image, (4, 2))["keep_pixels"], 1)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                mask_bridge.validate_mask(mask, image, (2, 4))
            alpha.putpixel((0, 0), 127)
            alpha.save(mask)
            with self.assertRaisesRegex(ValueError, "intermediate"):
                mask_bridge.validate_mask(mask, image, (4, 2))
            Image.new("RGBA", (4, 2), (0, 0, 0, 255)).save(mask)
            with self.assertRaisesRegex(ValueError, "grayscale"):
                mask_bridge.validate_mask(mask, image, (4, 2))

    def test_prepared_dataset_preserves_rgb_model_and_mask_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, masks, output = root / "source", root / "source-masks", root / "output"
            (source / "images").mkdir(parents=True)
            (source / "sparse").mkdir()
            masks.mkdir()
            image = source / "images/NP3_000.jpg"
            mask = masks / "NP3_000.mask.png"
            Image.new("RGB", (4, 2), (10, 20, 30)).save(image)
            alpha = Image.new("L", (4, 2), 0)
            alpha.putpixel((1, 1), 255)
            alpha.save(mask)
            model = source / "sparse/cameras.bin"
            model.write_bytes(b"model")
            mask_report = masks / "report.json"
            mask_report.write_text('{"status":"complete"}')
            total = image.stat().st_size + mask.stat().st_size + model.stat().st_size
            plan = {
                "schema": "brush_v030_ycb_mask_bridge_v1", "status": "preflight",
                "source_image_sha256": {image.name: sha256(image)},
                "source_model_sha256": {model.name: sha256(model)},
                "masks": {"NP3_000.png": {"source": str(mask), "sha256": sha256(mask),
                                          "registered_image": image.name, "size": [4, 2]}},
                "estimated_copy_bytes": total,
                "source_mask_report_sha256": sha256(mask_report),
            }
            with patch.object(mask_bridge, "OUTPUT", output), patch.object(mask_bridge, "SOURCE", source), \
                 patch.object(mask_bridge, "MASKS", masks), \
                 patch.object(mask_bridge, "MOUNT", root), \
                 patch.object(mask_bridge, "MIN_FREE", 0), \
                 patch.object(mask_bridge, "preflight", return_value=plan):
                report = mask_bridge.prepare()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["copied_bytes"], total)
            self.assertEqual((output / "dataset/images/NP3_000.jpg").read_bytes(), image.read_bytes())
            self.assertEqual((output / "dataset/sparse/cameras.bin").read_bytes(), model.read_bytes())
            self.assertEqual((output / "dataset/masks/NP3_000.png").read_bytes(), mask.read_bytes())
            self.assertTrue((output / "bridge-report.json").is_file())
            self.assertEqual(json.loads((output / "bridge-report.json").read_text())["copied_bytes"], total)

    def test_failed_copy_keeps_partial_dataset_and_failed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            (source / "images").mkdir(parents=True)
            (source / "images/view.jpg").write_bytes(b"not-the-declared-hash")
            plan = {"schema": "brush_v030_ycb_mask_bridge_v1", "status": "preflight",
                    "source_image_sha256": {"view.jpg": "0" * 64},
                    "source_model_sha256": {}, "masks": {}, "estimated_copy_bytes": 21}
            with patch.object(mask_bridge, "OUTPUT", output), patch.object(mask_bridge, "SOURCE", source), \
                 patch.object(mask_bridge, "MOUNT", root), patch.object(mask_bridge, "MIN_FREE", 0), \
                 patch.object(mask_bridge, "preflight", return_value=plan):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    mask_bridge.prepare()
            self.assertTrue((source / "images/view.jpg").exists())
            self.assertTrue((output / "dataset/images/view.jpg").exists())
            self.assertEqual(json.loads((output / "bridge-report.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
