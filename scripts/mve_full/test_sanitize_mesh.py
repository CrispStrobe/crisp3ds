"""The post-meshclean repair may remove zero-area faces only."""

import struct
import tempfile
from pathlib import Path
import unittest

from scripts.mve_full.run import ply_counts
from scripts.mve_full.sanitize_mesh import remove_zero_area_faces


class SanitizerTests(unittest.TestCase):
    def test_only_zero_area_face_removed_and_vertices_unchanged(self):
        header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
                  b"property float x\nproperty float y\nproperty float z\n"
                  b"property uchar red\nelement face 2\n"
                  b"property list uchar int vertex_indices\nend_header\n")
        vertices = b"".join(struct.pack("<fffB", *values) for values in
                            ((0, 0, 0, 5), (1, 0, 0, 6), (0, 1, 0, 7), (2, 0, 0, 8)))
        good = b"\x03" + struct.pack("<iii", 0, 1, 2)
        zero = b"\x03" + struct.pack("<iii", 0, 1, 3)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "native.ply", Path(directory) / "mesh.ply"
            payload = header + vertices + good + zero
            source.write_bytes(payload)
            result = remove_zero_area_faces(source, output)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(result["removed_zero_area_faces"], 1)
            self.assertEqual(ply_counts(output), (4, 1))
            self.assertEqual(output.read_bytes(), header.replace(b"element face 2", b"element face 1") + vertices + good)

    def test_refuses_all_zero_area_faces(self):
        header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                  b"property float x\nproperty float y\nproperty float z\n"
                  b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
        vertices = b"".join(struct.pack("<fff", *xyz) for xyz in ((0, 0, 0), (1, 0, 0), (2, 0, 0)))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "native.ply", Path(directory) / "mesh.ply"
            source.write_bytes(header + vertices + b"\x03" + struct.pack("<iii", 0, 1, 2))
            with self.assertRaisesRegex(ValueError, "all mesh faces"):
                remove_zero_area_faces(source, output)
            self.assertFalse(output.exists())
