"""Original-RGB capture manifest contracts, independent of SfM/depth/reference data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.capture_bundle import rgb_capture as capture


def make_photos(root):
    names = []
    for index in range(3):
        name = f"IMG_{index:04}.JPG"
        exif = Image.Exif()
        exif[274] = 6 if index == 0 else 1
        Image.new("RGB", (64, 48), (40 + index * 30, 80, 120)).save(root / name, exif=exif)
        names.append(name)
    return names


class RgbCaptureTests(unittest.TestCase):
    def test_original_jpegs_and_exif_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = make_photos(root)
            before = {name: capture.sha256(root / name) for name in names}
            output = root / "capture.json"
            with patch.object(capture, "MIN_FREE_BYTES", 0):
                bundle = capture.create(root, names, "unknown", output)
            report = capture.validate(root, output)
            self.assertEqual(report["image_count"], 3)
            self.assertFalse(report["metric_scale_asserted"])
            self.assertFalse(report["capture_motion_verified"])
            self.assertIn("unverified", report["projection_compatibility"])
            self.assertEqual(bundle["images"][0]["exif_orientation"], 6)
            self.assertEqual((bundle["images"][0]["stored_width"], bundle["images"][0]["stored_height"]), (64, 48))
            self.assertEqual(before, {name: capture.sha256(root / name) for name in names})
            self.assertIn("never moving-object poses", bundle["pose_use_policy"])

    def test_safe_paths_duplicates_and_content_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = make_photos(root)
            for name in ("../escape.jpg", "/abs.jpg", "x\\y.jpg", "copy.heic", "a/./b.jpg"):
                with self.assertRaises(ValueError):
                    capture.safe_photo(root, name)
            (root / "alias.jpg").symlink_to(root / names[0])
            with self.assertRaisesRegex(ValueError, "symlink"):
                capture.safe_photo(root, "alias.jpg")
            with patch.object(capture, "MIN_FREE_BYTES", 0):
                with self.assertRaisesRegex(ValueError, "duplicate image names"):
                    capture.create(root, [names[0], names[0]], "moving_object", root / "out.json")
                (root / "copy.jpg").write_bytes((root / names[0]).read_bytes())
                with self.assertRaisesRegex(ValueError, "duplicate JPEG"):
                    capture.create(root, [names[0], "copy.jpg"], "moving_object", root / "out.json")

    def test_optional_camera_metadata_requires_explicit_space_and_finite_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = make_photos(root)
            intrinsics = {"coordinate_space": "stored_jpeg_pixels", "source": "user measurement",
                          "width": 64, "height": 48, "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0,
                          "distortion_model": "none", "distortion": []}
            identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
            poses = {"coordinate_space": "arkit_world_camera_to_world", "translation_units": "meters",
                     "source": "ARKit metadata only",
                     "frames": [{"path": name, "matrix4x4": identity} for name in names]}
            with patch.object(capture, "MIN_FREE_BYTES", 0):
                bundle = capture.create(root, names, "moving_object", root / "capture.json",
                                        intrinsics=intrinsics, poses=poses)
            self.assertIn("never moving-object poses", bundle["pose_use_policy"])
            self.assertFalse(bundle["metric_scale_asserted"])
            self.assertEqual(capture.validate(root, root / "capture.json")["status"], "valid")
            intrinsics["fx"] = float("nan")
            with self.assertRaisesRegex(ValueError, "focal"):
                capture.validate_intrinsics(intrinsics, bundle["images"])
            intrinsics["fx"] = 50.0
            intrinsics["distortion"] = None
            with self.assertRaisesRegex(ValueError, "distortion"):
                capture.validate_intrinsics(intrinsics, bundle["images"])
            identity[0][0] = -1.
            with self.assertRaisesRegex(ValueError, "proper"):
                capture.validate_matrix(identity)

    def test_manifest_and_source_tampering_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = make_photos(root)
            output = root / "capture.json"
            with patch.object(capture, "MIN_FREE_BYTES", 0):
                capture.create(root, names, "stationary_object_moving_camera", output)
            payload = output.read_text()
            output.write_text(payload.replace('"metric_scale_asserted": false', '"metric_scale_asserted": true'))
            with self.assertRaisesRegex(ValueError, "scale assertion"):
                capture.validate(root, output)
            output.write_text(payload)
            malformed = json.loads(payload)
            malformed["images"][0]["path"] = ["IMG_0000.JPG"]
            output.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "JPEG record"):
                capture.validate(root, output)
            malformed["images"][0]["path"] = {"name": "IMG_0000.JPG"}
            output.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "JPEG record"):
                capture.validate(root, output)
            malformed = json.loads(payload)
            malformed["images"][0]["exif_orientation"] = True
            output.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "JPEG record"):
                capture.validate(root, output)
            output.write_text(payload)
            (root / names[0]).write_bytes(b"not a JPEG")
            with self.assertRaises((ValueError, OSError)):
                capture.validate(root, output)

    def test_exclusive_publication_defeats_preexisting_and_raced_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = make_photos(root)
            output = root / "capture.json"
            output.write_bytes(b"preexisting user data")
            with patch.object(capture, "MIN_FREE_BYTES", 0):
                with self.assertRaises(FileExistsError):
                    capture.create(root, names, "unknown", output)
                # Simulate the pre-check missing a file placed before exclusive open.
                with patch.object(Path, "exists", return_value=False):
                    with self.assertRaises(FileExistsError):
                        capture.create(root, names, "unknown", output)
            self.assertEqual(output.read_bytes(), b"preexisting user data")


if __name__ == "__main__":
    unittest.main()
