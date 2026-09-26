"""Geometry convention checks for the MVE scene converter."""

from pathlib import Path
import unittest

from scripts.mve_spike.convert_scene import camera_coordinates, mve_intrinsics, project


ROOT = Path(__file__).resolve().parents[2]


class ConversionTests(unittest.TestCase):
    def test_object_to_camera_and_pixel_center_conventions(self):
        calibration = {"width": 1600, "height": 1200, "fx": 1500, "fy": 1500,
                       "cx": 800, "cy": 600}
        rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        translation = [-100, -200, 700]
        self.assertEqual(camera_coordinates(rotation, translation, [100, 200, 0]), [0, 0, 700])
        self.assertEqual(project(calibration, rotation, translation, [100, 200, 0]), (800, 600))
        flen, paspect, ppx, ppy = mve_intrinsics(calibration)
        self.assertEqual(flen, 1500 / 1600)
        self.assertEqual(paspect, 1)
        self.assertAlmostEqual(1600 * ppx - 0.5, 800)
        self.assertAlmostEqual(1200 * ppy - 0.5, 600)


if __name__ == "__main__":
    unittest.main()
