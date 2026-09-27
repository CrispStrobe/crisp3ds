"""Synthetic-only camera/mask/voxel guards for proposed cracker control."""

import unittest

import numpy as np

from scripts.classical_backend.visual_hull_core import (
    PinholeView, boundary_touched, carve_chunk, exposed_cube_mesh,
    sparse_cube, voxel_centers,
)


class VisualHullCoreTest(unittest.TestCase):
    def view(self, supported_pixel=(1, 1)):
        mask = np.zeros((4, 4), np.uint8)
        mask[supported_pixel[1], supported_pixel[0]] = 255
        return PinholeView(np.eye(3), np.zeros(3), (1, 1, 1.5, 1.5), mask)

    def test_colmap_pixel_edges_and_all_visible_silhouettes(self):
        points = np.asarray([(0, 0, 1), (1, 0, 1), (0, 0, -1), (9, 0, 1)], float)
        views = (self.view(), self.view(), self.view())
        kept, visible = carve_chunk(points, views, min_visible=3)
        self.assertEqual(kept.tolist(), [True, False, False, False])
        self.assertEqual(visible.tolist(), [3, 3, 0, 0])
        shifted = (self.view(), self.view(), self.view((2, 1)))
        kept, _ = carve_chunk(points, shifted, min_visible=3)
        self.assertFalse(kept.any())
        outside = PinholeView(np.eye(3), np.array((9., 0., 0.)),
                              (1, 1, 1.5, 1.5), self.view().mask)
        kept, seen = carve_chunk(points[:1], (self.view(), self.view(), outside), min_visible=2)
        self.assertEqual((kept.tolist(), seen.tolist()), ([True], [2]))
        kept, _ = carve_chunk(points[:1], (self.view(), self.view(), outside), min_visible=3)
        self.assertFalse(kept.any())

    def test_sparse_bounds_are_scanner_free_and_finite(self):
        points = np.asarray([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        center, side = sparse_cube(points)
        np.testing.assert_allclose(center, (0.5, 0.5, 0.5))
        self.assertAlmostEqual(side, 1.5)
        with self.assertRaisesRegex(ValueError, "sparse points"):
            sparse_cube(points[:7])

    def test_reflection_and_nonbinary_mask_fail_closed(self):
        reflected = self.view()
        with self.assertRaisesRegex(ValueError, "invalid fixed"):
            PinholeView(np.diag((-1, 1, 1)), reflected.translation, reflected.k, reflected.mask)
        bad = np.zeros((4, 4), np.uint8)
        bad[0, 0] = 128
        with self.assertRaisesRegex(ValueError, "invalid fixed"):
            PinholeView(np.eye(3), np.zeros(3), (1, 1, 1.5, 1.5), bad)

    def test_voxel_order_and_neutral_exposed_face_mesh(self):
        centers = voxel_centers(0, 2, 4, np.zeros(3), 4)
        np.testing.assert_allclose(centers, [(-1.5, -1.5, -1.5), (-1.5, -1.5, -0.5)])
        occupied = np.zeros((5, 5, 5), bool)
        occupied[2, 2, 2] = True
        self.assertFalse(boundary_touched(occupied))
        vertices, faces = exposed_cube_mesh(occupied, np.zeros(3), 5)
        self.assertEqual((len(vertices), len(faces)), (8, 12))
        volume = np.einsum("ij,ij->", vertices[faces[:, 0]],
                           np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])) / 6
        self.assertAlmostEqual(volume, 1.0)
        occupied[2, 2, 3] = True
        vertices, faces = exposed_cube_mesh(occupied, np.zeros(3), 5)
        self.assertEqual((len(vertices), len(faces)), (12, 20))
        occupied[0, 0, 0] = True
        self.assertTrue(boundary_touched(occupied))
        with self.assertRaisesRegex(ValueError, "truncated"):
            exposed_cube_mesh(occupied, np.zeros(3), 5)


if __name__ == "__main__":
    unittest.main()
