from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from scripts.apple_object_capture import review_bunny_texture as target


class TextureReviewTest(unittest.TestCase):
    def test_topology_detects_detached_and_boundary_faces(self):
        vertices = [(0, 0, 0)] * 6
        faces = [(0, 1, 2), (3, 4, 5)]
        self.assertEqual(target.topology(vertices, faces), {
            "face_connected_components": 2, "largest_component_faces": 1,
            "boundary_edges": 6, "nonmanifold_edges": 0})

    def test_faces_touching_only_at_vertex_remain_separate(self):
        vertices = [(0, 0, 0)] * 5
        result = target.topology(vertices, [(0, 1, 2), (0, 3, 4)])
        self.assertEqual(result["face_connected_components"], 2)

    def test_uv_array_with_usd_metadata_suffix(self):
        source = 'texCoord2f[] primvars:st = [(0.1, 0.2), (0.3, 0.4)] (\n'
        self.assertEqual(target._array(source, "texCoord2f[] primvars:st"),
                         [(0.1, 0.2), (0.3, 0.4)])

    def test_archive_rejects_extra_file_before_usdcat(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "bad.usdz"
            image = folder / "image.png"
            Image.new("RGB", (2, 2), "gray").save(image)
            with ZipFile(source, "w") as archive:
                archive.writestr("scene.usdc", b"binary")
                archive.write(image, "0/texture.png")
                archive.writestr("extra.txt", b"unexpected")
            tool = folder / "usdcat"
            tool.write_bytes(b"tool")
            with self.assertRaisesRegex(ValueError, "one bounded USD scene"):
                target.inspect_usdz(source, tool)


if __name__ == "__main__":
    unittest.main()
