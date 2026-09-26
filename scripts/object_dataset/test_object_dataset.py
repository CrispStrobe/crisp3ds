import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.object_dataset import evaluate, fetch_3dlf


class GeometryTests(unittest.TestCase):
    def make_ply(self, root, index_type="int", faces=((0, 1, 2),)):
        path = root / "mesh.ply"
        header = ("ply\nformat binary_little_endian 1.0\n"
                  "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                  f"element face {len(faces)}\nproperty list uchar {index_type} vertex_indices\n"
                  "end_header\n").encode()
        vertices = b"".join(struct.pack("<fff", *xyz) for xyz in
                            ((0, 0, 0), (1, 0, 0), (2, 0, 0)))
        code = "i" if index_type == "int" else "I"
        triangles = b"".join(struct.pack("<B" + code * 3, 3, *face) for face in faces)
        path.write_bytes(header + vertices + triangles)
        return path

    def test_signed_and_unsigned_faces_and_zero_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index_type in ("int", "uint"):
                path = self.make_ply(root, index_type)
                report = evaluate.inspect_ply(path)
                self.assertEqual(report["faces"], 1)
                self.assertEqual(report["zero_area_triangle_faces"], 1)
                self.assertEqual(report["faces_with_repeated_indices"], 0)

    def test_negative_signed_face_index_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_ply(Path(temporary), faces=((-1, 1, 2),))
            with self.assertRaisesRegex(ValueError, "face index"):
                evaluate.inspect_ply(path)


class SourceTests(unittest.TestCase):
    def test_subset_requires_independent_mesh_and_50_to_100_photos(self):
        prefix = fetch_3dlf.PREFIX
        info = {"compressed": 100, "expanded": 100}
        entries = {prefix + f"pro/bunny/rgb/bunny_{i}_rgb.png": info for i in range(73)}
        entries.update({prefix + p: info for p in (
            "calib_pro/intrinsics/rgb_optic.json", "pro/bunny/pcd/poses_metric.json",
            "revopoint/bunny/fuse_mesh_rgb.ply")})
        self.assertEqual(len(fetch_3dlf.selection(entries)), 76)
        del entries[prefix + "revopoint/bunny/fuse_mesh_rgb.ply"]
        with self.assertRaisesRegex(ValueError, "required member"):
            fetch_3dlf.selection(entries)

    def test_local_download_matches_manifest_if_present(self):
        root = fetch_3dlf.ROOT
        if not (root / "manifest.json").exists():
            self.skipTest("optional downloaded subset is absent")
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len([f for f in manifest["files"] if f["path"].startswith("pro/bunny/rgb/")]), 73)
        for item in manifest["files"]:
            source = root / item["path"]
            self.assertEqual(source.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
