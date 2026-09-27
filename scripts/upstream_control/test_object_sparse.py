"""Input and option contract for the stock YCB object sparse lane."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.upstream_control import image_only, object_sparse
from scripts.upstream_control.test_image_only import Option


class ObjectSparseTests(unittest.TestCase):
    @staticmethod
    def fixture(base):
        source = base / "source"
        photos = source / "photos"
        photos.mkdir(parents=True)
        prepared_dir = base / "prepared"
        mask_dir = prepared_dir / "masks"
        mask_dir.mkdir(parents=True)
        photo_rows, mask_rows = [], []
        for angle, name in zip(range(0, 360, 6), object_sparse.NAMES):
            photo = photos / name
            photo.write_bytes(f"photo-{angle}".encode())
            photo_sha = image_only.digest(photo)
            photo_rows.append({"path": f"photos/{name}", "bytes": photo.stat().st_size,
                               "sha256": photo_sha, "turntable_angle_degrees": angle})
            mask = mask_dir / f"{name}.png"
            mask.write_bytes(f"mask-{angle}".encode())
            mask_rows.append({"name": name, "path": str(photo), "sha256": photo_sha,
                              "pose_support_mask": str(mask), "mask_sha256": image_only.digest(mask)})
        photo_manifest = source / "manifest.json"
        photo_manifest.write_text(json.dumps({"object_id": "003_cracker_box", "photos": photo_rows}))
        mask_manifest = prepared_dir / "manifest.json"
        mask_manifest.write_text(json.dumps({"schema": "object_motion_prepare_v1",
                                             "source_manifest_sha256": image_only.digest(photo_manifest),
                                             "images": mask_rows}))
        return source, mask_manifest

    def test_synthetic_60_view_inventory_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, mask_manifest = self.fixture(Path(temporary))
            with patch.object(object_sparse, "PHOTO_MANIFEST_SHA256", image_only.digest(source / "manifest.json")), \
                 patch.object(object_sparse, "MASK_MANIFEST_SHA256", image_only.digest(mask_manifest)):
                raw, masks, meta = object_sparse.inventory(source, "raw", mask_manifest)
                self.assertEqual(len(raw), 60)
                self.assertEqual(masks, [])
                self.assertIsNone(meta["mask_manifest_sha256"])
                photos, masks, meta = object_sparse.inventory(source, "masked", mask_manifest)
                self.assertEqual([Path(row["path"]).name for row in photos], list(object_sparse.NAMES))
                self.assertEqual([row["name"] for row in masks], list(object_sparse.NAMES))
                self.assertEqual(meta["mask_manifest_sha256"], image_only.digest(mask_manifest))

                photo = source / "photos" / object_sparse.NAMES[0]
                original = photo.read_bytes()
                photo.write_bytes(b"X" + original[1:])
                with self.assertRaisesRegex(ValueError, "photo differs from manifest"):
                    object_sparse.inventory(source, "raw", mask_manifest)
                photo.write_bytes(original)

                mask = mask_manifest.parent / "masks" / f"{object_sparse.NAMES[0]}.png"
                original = mask.read_bytes()
                mask.write_bytes(b"X" + original[1:])
                with self.assertRaisesRegex(ValueError, "mask differs from manifest"):
                    object_sparse.inventory(source, "masked", mask_manifest)

    def test_single_camera_and_mask_are_only_additional_option_changes(self):
        fake = type("Fake", (), {
            "ImageReaderOptions": lambda: Option(camera_model="SIMPLE_RADIAL", camera_params=""),
            "SiftExtractionOptions": lambda: Option(num_threads=-1, max_num_features=8192,
                                                     max_image_size=3200),
            "SiftMatchingOptions": lambda: Option(num_threads=-1),
            "ExhaustiveMatchingOptions": lambda: Option(),
            "TwoViewGeometryOptions": lambda: Option(),
            "IncrementalPipelineOptions": lambda: Option(num_threads=-1, mapper=Option(num_threads=-1),
                                                         max_num_models=50, min_model_size=10),
            "CameraMode": Option(AUTO=Option(name="AUTO"), SINGLE=Option(name="SINGLE")),
            "Device": Option(cpu=Option(name="cpu")),
        })
        effective = image_only.effective_options(fake, fake.CameraMode.SINGLE, "/tmp/masks")
        self.assertEqual(effective["camera_mode"], "SINGLE")
        self.assertEqual(effective["image_reader"]["mask_path"], "/tmp/masks")
        self.assertEqual(effective["sift_extraction"]["max_num_features"], 8192)
        self.assertEqual(effective["sift_extraction"]["max_image_size"], 3200)
        self.assertEqual(effective["incremental_pipeline"]["max_num_models"], 50)
        self.assertEqual(effective["incremental_pipeline"]["min_model_size"], 10)
        self.assertEqual(effective["deviations_from_pycolmap_3_11_1_defaults"]["camera_mode"], "SINGLE")

    def test_existing_destination_rejected_before_any_source_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with patch.object(object_sparse, "inventory", side_effect=AssertionError("read source")):
                with self.assertRaises(FileExistsError):
                    object_sparse.run(path, path, "raw")


if __name__ == "__main__":
    unittest.main()
