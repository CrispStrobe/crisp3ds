import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest

from scripts.brush_backend.fetch_release import safe_members
from scripts.brush_backend.preflight import _images_from_binary, validate_dataset


def image_record(name: str, points: bytes = b"") -> bytes:
    return struct.pack("<I7dI", 1, 1, 0, 0, 0, 0, 0, 0, 1) + name.encode() + b"\0" + struct.pack("<Q", len(points) // 24) + points


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        model = self.dataset / "sparse" / "0"
        model.mkdir(parents=True)
        (self.dataset / "images").mkdir()
        for name in ("cameras.bin", "points3D.bin"):
            (model / name).write_bytes(b"nonempty")
        (model / "images.bin").write_bytes(struct.pack("<Q", 1) + image_record("frame.jpg"))
        (self.dataset / "images" / "frame.jpg").write_bytes(b"image")

    def test_seals_model_and_images_no_train_claim(self):
        report = validate_dataset(self.dataset, self.root / "fresh")
        self.assertEqual(report["registered_images"], 1)
        self.assertFalse(report["training_authorized"])
        self.assertEqual(report["external_heldout_inventory"], "unverified")
        self.assertEqual(len(report["model_file_sha256"]), 3)

    def test_rejects_existing_output_and_unreviewed_masks(self):
        (self.root / "fresh").mkdir()
        with self.assertRaisesRegex(ValueError, "fresh"):
            validate_dataset(self.dataset, self.root / "fresh")
        (self.root / "fresh").rmdir()
        (self.dataset / "masks").mkdir()
        with self.assertRaisesRegex(ValueError, "masks"):
            validate_dataset(self.dataset, self.root / "fresh")

    def test_rejects_truncated_last_points_payload(self):
        path = self.dataset / "sparse" / "0" / "images.bin"
        path.write_bytes(struct.pack("<Q", 1) + image_record("frame.jpg")[:-1] + struct.pack("<Q", 1))
        with self.assertRaises(ValueError):
            _images_from_binary(path)

    def test_rejects_cross_platform_path_and_parent_symlink(self):
        path = self.dataset / "sparse" / "0" / "images.bin"
        path.write_bytes(struct.pack("<Q", 1) + image_record("C:\\outside.jpg"))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_dataset(self.dataset, self.root / "fresh")
        path.write_bytes(struct.pack("<Q", 1) + image_record("frame.jpg"))
        (self.dataset / "images" / "frame.jpg").unlink()
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        (self.dataset / "images" / "frame.jpg").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_dataset(self.dataset, self.root / "fresh")

    def test_split_manifest_must_bind_all_images_and_disjoint_heldout(self):
        hashes = validate_dataset(self.dataset, self.root / "fresh")["image_sha256"]
        manifest = self.root / "split.json"
        body = {"schema": "brush_train_split_v1", "train_image_sha256": hashes, "heldout_image_sha256": {}}
        manifest.write_text(json.dumps(body))
        report = validate_dataset(self.dataset, self.root / "fresh", split_manifest=manifest)
        self.assertTrue(report["input_split_verified"])
        self.assertFalse(report["training_authorized"])
        body["heldout_image_sha256"] = hashes.copy()
        manifest.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_dataset(self.dataset, self.root / "fresh", split_manifest=manifest)
        body["heldout_image_sha256"] = {}
        body["train_image_sha256"] = {"frame.jpg": "0" * 64}
        manifest.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, "train hashes"):
            validate_dataset(self.dataset, self.root / "fresh", split_manifest=manifest)


class ArchiveTests(unittest.TestCase):
    def test_rejects_links_and_parent_escape(self):
        for name, type_ in (("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            with self.subTest(name=name):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = type_
                    archive.addfile(member, io.BytesIO())
                stream.seek(0)
                with tarfile.open(fileobj=stream) as archive:
                    with self.assertRaisesRegex(ValueError, "unsafe"):
                        safe_members(archive)


if __name__ == "__main__":
    unittest.main()
