import unittest

from PIL import Image

try:
    from .diagnose_seeds import bilinear, patch, project, zncc
except ImportError:
    from diagnose_seeds import bilinear, patch, project, zncc


class DiagnoseSeedsTest(unittest.TestCase):
    def test_zncc_identical_opposite_and_constant(self):
        values = [0, 1, 5, 2, 8, 3]
        self.assertAlmostEqual(zncc(values, values), 1)
        self.assertAlmostEqual(zncc(values, [-value for value in values]), -1)
        self.assertIsNone(zncc(values, [3] * len(values)))

    def test_translated_patch_and_boundary(self):
        image = Image.new("L", (13, 13))
        for y in range(13):
            for x in range(13):
                image.putpixel((x, y), (x*x*13 + y*y*17 + x*y*11) % 251)
        center = patch(image, 6, 6)
        self.assertAlmostEqual(zncc(center, center), 1)
        self.assertLess(zncc(center, patch(image, 7, 6)), .95)
        self.assertIsNone(patch(image, 2, 6))
        self.assertAlmostEqual(bilinear(image, 6.5, 6),
                               (image.getpixel((6, 6)) + image.getpixel((7, 6))) / 2)

    def test_projection_world_to_camera_convention(self):
        view = {"rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1],
                "translation": [2, 0, 0],
                "camera": {"fx": 10, "fy": 20, "cx": 4, "cy": 5}}
        # R*[1,2,4]+t = [0,1,4].
        self.assertEqual(project((1, 2, 4), view), (4, 10))
        self.assertIsNone(project((1, 2, -1), view))


if __name__ == "__main__":
    unittest.main()
