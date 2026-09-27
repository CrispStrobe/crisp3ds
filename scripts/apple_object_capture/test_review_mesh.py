from __future__ import annotations

import unittest

from scripts.apple_object_capture import review_mesh


SOURCE = '''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "ObjectCapture" {
    def Scope "Geometry" {
        def Mesh "Mesh" {
            float3[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            uniform token subdivisionScheme = "none"
        }
    }
}
'''


class ReviewMeshTests(unittest.TestCase):
    def test_constrained_triangle_roundtrip(self):
        vertices, faces = review_mesh.parse_mesh(SOURCE)
        self.assertEqual(vertices, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0)])
        self.assertEqual(faces, [(0, 1, 2)])
        self.assertIn(b"element face 1\n", review_mesh.ply_bytes(vertices, faces))

    def test_rejects_authored_transform(self):
        with self.assertRaisesRegex(ValueError, "transforms"):
            review_mesh.parse_mesh(SOURCE.replace('def Mesh "Mesh" {',
                                                   'def Mesh "Mesh" {\n xformOp:translate = (1, 0, 0)'))

    def test_rejects_nontriangle_and_bad_index(self):
        with self.assertRaisesRegex(ValueError, "triangles"):
            review_mesh.parse_mesh(SOURCE.replace("faceVertexCounts = [3]",
                                                  "faceVertexCounts = [4]"))
        with self.assertRaisesRegex(ValueError, "face index"):
            review_mesh.parse_mesh(SOURCE.replace("faceVertexIndices = [0, 1, 2]",
                                                  "faceVertexIndices = [0, 1, 3]"))

    def test_rejects_additional_mesh(self):
        with self.assertRaisesRegex(ValueError, "exactly one Mesh"):
            review_mesh.parse_mesh(SOURCE.replace('def Mesh "Mesh" {',
                                                   'def Mesh "Other" {}\n def Mesh "Mesh" {'))


if __name__ == "__main__":
    unittest.main()
