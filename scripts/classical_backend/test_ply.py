"""Strict OpenMVS PLY and geometry-export regressions."""

import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.classical_backend import geometry
from scripts.object_dataset.evaluate import inspect_ply

ROOT = Path(__file__).resolve().parents[2]


def fixture(*, face=True, bad_index=False, nonfinite=None, mismatch=False,
            aliases=True, duplicate=False, trailing=False):
    f = "float32" if aliases else "float"
    u8 = "uint8" if aliases else "uchar"
    u32 = "uint32" if aliases else "uint"
    header = ["ply", "format binary_little_endian 1.0", "element vertex 3",
              f"property {f} x", f"property {f} y", f"property {f} z",
              f"property {u8} red", f"property {u8} green", f"property {u8} blue",
              f"property {f} nx", f"property {f} ny", f"property {f} nz",
              f"property list {u8} {u32} view_indices",
              f"property list {u8} {f} view_weights"]
    if duplicate:
        header.append(f"property {f} x")
    if face:
        header.extend(["element face 1", f"property list {u8} {u32} vertex_indices"])
    header.append("end_header")
    coords = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    body = bytearray()
    for i, xyz in enumerate(coords):
        if nonfinite == "xyz" and i == 0:
            xyz = (math.nan, xyz[1], xyz[2])
        normal = (0.0, 0.0, math.inf if nonfinite == "normal" and i == 0 else 1.0)
        body.extend(struct.pack("<fffBBBfff", *xyz, 10, 20, 30, *normal))
        body.extend(struct.pack("<BII", 2, 0, 1))
        weights = (0.3, math.nan if nonfinite == "weight" and i == 0 else 0.7)
        body.extend(struct.pack("<Bff", 1 if mismatch and i == 0 else 2, *weights) if not mismatch or i != 0
                    else struct.pack("<Bf", 1, weights[0]))
    if face:
        body.extend(struct.pack("<BIII", 3, 0, 1, 3 if bad_index else 2))
    if trailing:
        body.extend(b"X")
    return ("\n".join(header) + "\n").encode("ascii") + body


class OpenMVSPlyTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".local-tools/tmp").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "native.ply"

    def write(self, payload):
        self.path.write_bytes(payload)

    def test_aliases_visibility_and_exact_geometry_export(self):
        self.write(fixture())
        native_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(geometry.counts(self.path), (3, 1))
        dest = Path(self.tmp.name) / "geometry.ply"
        report = geometry.export_geometry(self.path, dest)
        self.assertEqual(report["source_sha256"], native_hash)
        self.assertEqual((report["vertices"], report["faces"]), (3, 1))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), native_hash)
        evaluated, vertices, triangles = inspect_ply(dest, geometry=True)
        self.assertEqual((evaluated["vertices"], evaluated["faces"]), (3, 1))
        self.assertEqual(len(vertices), 3)
        self.assertEqual(triangles, [(0, 1, 2)])
        with self.assertRaises(FileExistsError):
            geometry.export_geometry(self.path, dest)

    def test_point_cloud_without_face_element_exports_zero_faces(self):
        self.write(fixture(face=False, aliases=False))
        dest = Path(self.tmp.name) / "geometry.ply"
        report = geometry.export_geometry(self.path, dest)
        self.assertEqual((report["vertices"], report["faces"]), (3, 0))
        self.assertEqual(inspect_ply(dest)["faces"], 0)

    def test_truncation_trailing_bad_indices_and_nonfinite(self):
        good = fixture()
        cases = [good[:-1], fixture(bad_index=True), fixture(nonfinite="xyz"),
                 fixture(nonfinite="normal"), fixture(nonfinite="weight"),
                 fixture(mismatch=True), fixture(trailing=True)]
        for case in cases:
            with self.subTest(length=len(case), digest=hashlib.sha256(case).hexdigest()[:8]):
                self.write(case)
                with self.assertRaises(ValueError):
                    geometry.inspect(self.path)

    def test_rejects_duplicate_property_and_excessive_count(self):
        self.write(fixture(duplicate=True))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            geometry.inspect(self.path)
        self.write(fixture().replace(b"element vertex 3", b"element vertex 5000001", 1))
        with self.assertRaisesRegex(ValueError, "outside limits"):
            geometry.inspect(self.path)


if __name__ == "__main__":
    unittest.main()
