import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.neural_bridge import colmap_to_nerfstudio as bridge


class CameraBridgeTests(unittest.TestCase):
    def test_nonidentity_pose_and_independent_projection_golden(self):
        q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]  # World-to-camera +90° around Z.
        t = [1, 2, 3]
        c2w = bridge.opengl_c2w(q, t)
        np.testing.assert_allclose(c2w[:3, :3], [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], atol=1e-14)
        np.testing.assert_allclose(c2w[:3, 3], [-2, 1, -3], atol=1e-14)
        world = np.array([1, 0, 2, 1.])
        gl = np.linalg.inv(c2w) @ world
        u = 100 * gl[0] / -gl[2] + 50
        v = 120 * -gl[1] / -gl[2] + 60
        np.testing.assert_allclose([u, v], [70, 132], atol=1e-12)
        np.testing.assert_allclose(bridge.opengl_c2w([1, 0, 0, 0], [0, 0, 0]),
                                   np.diag([1, -1, -1, 1]))

    def test_export_is_fresh_and_does_not_mutate_sources(self):
        with tempfile.TemporaryDirectory(dir=bridge.ROOT / ".local-tools/tmp") as folder:
            folder = Path(folder)
            model, photos, output = folder / "model", folder / "photos", folder / "out"
            model.mkdir()
            photos.mkdir()
            (model / "cameras.txt").write_text("1 PINHOLE 100 80 70 71 50 40\n")
            poses = "".join(f"{i} 1 0 0 0 0 0 0 1 im{i}.jpg\n\n" for i in (1, 2, 3))
            (model / "images.txt").write_text(poses)
            for i in (1, 2, 3):
                Image.new("RGB", (100, 80), (i, i, i)).save(photos / f"im{i}.jpg")
            original = {str(p): bridge.sha256(p) for p in (*model.iterdir(), *photos.iterdir())}
            report = bridge.build(model, photos, output, ["im1.jpg"], ["im2.jpg"],
                                  ["im3.jpg"], "estimated")
            result = json.loads((output / "transforms.json").read_text())
            self.assertEqual(result["camera_model"], "OPENCV")
            self.assertEqual(result["orientation_override"], "none")
            self.assertEqual(len(result["frames"]), 3)
            self.assertEqual(result["train_filenames"], [result["frames"][0]["file_path"]])
            self.assertEqual(result["frames"][0]["fl_x"], 70)
            self.assertEqual(report["image_or_mask_bytes_copied"], 0)
            self.assertEqual(original, {str(p): bridge.sha256(p) for p in (*model.iterdir(), *photos.iterdir())})
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["provenance.json", "transforms.json"])
            with self.assertRaises(ValueError):
                bridge.build(model, photos, output, ["im1.jpg"], ["im2.jpg"], ["im3.jpg"], "estimated")

    def test_distortion_traversal_and_split_rejections(self):
        with tempfile.TemporaryDirectory(dir=bridge.ROOT / ".local-tools/tmp") as folder:
            root = Path(folder)
            cameras = root / "cameras.txt"
            cameras.write_text("1 SIMPLE_RADIAL 100 80 70 50 40 0.1\n")
            with self.assertRaisesRegex(ValueError, "distorted"):
                bridge.parse_cameras(cameras)
            cameras.write_text("1 SIMPLE_PINHOLE 100 80 70 50 40\n")
            self.assertEqual(bridge.parse_cameras(cameras)[1]["fl_y"], 70)
            (root / "image.jpg").write_bytes(b"test")
            with self.assertRaises(ValueError):
                bridge.safe_source(root, "../image.jpg")
            with self.assertRaises(ValueError):
                bridge.safe_source(root, "/image.jpg")
            with self.assertRaises(ValueError):
                bridge.validate_splits({"a": 1, "b": 2, "c": 3}, ["a"], ["b"], ["a"])
            with self.assertRaises(ValueError):
                bridge.validate_splits({"a": 1, "b": 2, "c": 3}, ["a"], ["b"], ["z"])

    def test_raster_dimensions_binary_mask_and_truncation(self):
        with tempfile.TemporaryDirectory(dir=bridge.ROOT / ".local-tools/tmp") as folder:
            root = Path(folder)
            image = root / "image.jpg"
            mask = root / "mask.png"
            Image.new("RGB", (20, 10)).save(image)
            bridge._validate_raster(image, 20, 10, False)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                bridge._validate_raster(image, 19, 10, False)
            Image.new("L", (20, 10), 1).save(mask)
            bridge._mask_header(mask, 20, 10)
            with self.assertRaisesRegex(ValueError, "black or white"):
                bridge._validate_raster(mask, 20, 10, True)
            Image.new("L", (20, 10), 255).save(mask)
            bridge._validate_raster(mask, 20, 10, True)
            image.write_bytes(image.read_bytes()[:20])
            with self.assertRaises(Exception):
                bridge._validate_raster(image, 20, 10, False)


if __name__ == "__main__":
    unittest.main()
