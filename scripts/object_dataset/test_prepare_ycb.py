"""Bound and geometry-preservation checks for the YCB subset preparer."""

import io
from pathlib import Path
import struct
import tarfile
import tempfile
import time
import unittest

from scripts.object_dataset import prepare_ycb
from scripts.object_dataset.evaluate import inspect_ply


class YCBPreparationTests(unittest.TestCase):
    def setUp(self):
        prepare_ycb.TMP.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=prepare_ycb.TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_selection_is_single_camera_full_rotation(self):
        self.assertEqual(len(prepare_ycb.PHOTO_MEMBERS), 60)
        self.assertEqual(tuple(prepare_ycb.PHOTO_MEMBERS.values()), tuple(range(0, 360, 6)))
        self.assertTrue(all("/NP3_" in x and x.endswith(".jpg") for x in prepare_ycb.PHOTO_MEMBERS))

    def test_safe_tar_only_extracts_named_regular_file(self):
        archive = self.root / "tiny.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in (("obj/other.jpg", b"ignore"), ("obj/NP3_0.jpg", b"\xff\xd8\xffok\xff\xd9")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        stage = self.root / "stage"
        stage.mkdir()
        records = prepare_ycb.extract_selected(archive, {"obj/NP3_0.jpg": Path("photos/0.jpg")},
                                                stage, 3, 100)
        self.assertEqual(len(records), 1)
        self.assertEqual((stage / "photos/0.jpg").read_bytes(), b"\xff\xd8\xffok\xff\xd9")
        self.assertFalse((stage / "obj/other.jpg").exists())

    def test_tar_rejects_traversal_and_links(self):
        seen = set()
        for name in ("../escape", "/absolute", "safe/../escape"):
            info = tarfile.TarInfo(name)
            info.size = 1
            with self.assertRaises(ValueError):
                prepare_ycb.check_member(info, seen)
        link = tarfile.TarInfo("obj/link")
        link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            prepare_ycb.check_member(link, seen)

    def test_ascii_scan_conversion_preserves_geometry(self):
        source = self.root / "original.ply"
        source.write_text("ply\nformat ascii 1.0\nelement vertex 3\n"
                          "property float x\nproperty float y\nproperty float z\n"
                          "property float nx\nelement face 1\n"
                          "property list uchar int vertex_indices\nend_header\n"
                          "0.1 0 0 1\n1 0 0 1\n0 1 0 1\n3 0 1 2\n")
        target = self.root / "converted.ply"
        report = prepare_ycb.convert_ascii_scan(source, target)
        self.assertEqual((report["vertices"], report["faces"]), (3, 1))
        geometry = inspect_ply(target)
        self.assertEqual((geometry["vertices"], geometry["faces"]), (3, 1))
        self.assertEqual(geometry["zero_area_triangle_faces"], 0)
        header_end = target.read_bytes().index(b"end_header\n") + len(b"end_header\n")
        self.assertEqual(struct.unpack_from("<ddd", target.read_bytes(), header_end), (0.1, 0.0, 0.0))

    def test_conversion_refuses_polygons(self):
        source = self.root / "quad.ply"
        source.write_text("ply\nformat ascii 1.0\nelement vertex 4\n"
                          "property float x\nproperty float y\nproperty float z\n"
                          "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                          "0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
        with self.assertRaises(ValueError):
            prepare_ycb.convert_ascii_scan(source, self.root / "out.ply")

    def test_existing_archive_symlink_is_rejected(self):
        payload = self.root / "payload"
        payload.write_bytes(b"source")
        try:
            (self.root / "archive.tgz").symlink_to(payload)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_ycb.fetch_archive(self.root, {"filename": "archive.tgz"}, time.monotonic() + 1)

    def test_scan_conversion_deadline(self):
        source = self.root / "tiny.ply"
        source.write_text("ply\nformat ascii 1.0\nelement vertex 1\n"
                          "property float x\nproperty float y\nproperty float z\n"
                          "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                          "0 0 0\n3 0 0 0\n")
        with self.assertRaises(TimeoutError):
            prepare_ycb.convert_ascii_scan(source, self.root / "late.ply", time.monotonic() - 1)


if __name__ == "__main__":
    unittest.main()
