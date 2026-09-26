import math
from pathlib import Path
import tempfile
import unittest

from scripts.tree_dense.import_scene import (mve_intrinsics, parse_seeds, project,
                                             quaternion_rotation, read_cameras,
                                             read_images, scaled_camera)


class ImportGeometryTest(unittest.TestCase):
    def test_colmap_world_to_camera_and_pixel_center_resize(self):
        rotation = quaternion_rotation([1,0,0,0])
        self.assertEqual(rotation, [1,0,0,0,1,0,0,0,1])
        original = dict(width=1600,height=1200,fx=1200,fy=1100,cx=800,cy=600)
        small = scaled_camera(original,800,600)
        point = [0.2,-0.1,4]
        before = project(dict(original,cx=original["cx"]-.5,cy=original["cy"]-.5),
                         rotation,[0,0,1],point)
        after = project(small,rotation,[0,0,1],point)
        self.assertAlmostEqual(after[0], (before[0]+.5)*.5-.5)
        self.assertAlmostEqual(after[1], (before[1]+.5)*.5-.5)
        focal, aspect, ppx, ppy = mve_intrinsics(small)
        self.assertAlmostEqual(focal*800,small["fx"])
        self.assertAlmostEqual(focal*800*aspect,small["fy"])
        self.assertAlmostEqual(ppx*800,small["cx"]+.5)
        self.assertAlmostEqual(ppy*600,small["cy"]+.5)
        self.assertAlmostEqual(small["cx"],399.5)
        self.assertAlmostEqual(small["cy"],299.5)

    def test_rejects_distorted_camera_and_malformed_image_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"cameras.txt"
            path.write_text("1 OPENCV 800 600 700 700 400 300 0.1 0 0 0\n")
            with self.assertRaisesRegex(ValueError,"unsupported/distorted"):
                read_cameras(path)
            path.write_text("1 PINHOLE 800 600 700 710 400 300\n")
            self.assertEqual(read_cameras(path)[1]["fy"],710)
            images=Path(directory)/"images.txt"
            images.write_text("1 1 0 0 0 0 0 0 1 photo.jpg\n")
            with self.assertRaisesRegex(ValueError,"missing COLMAP image observation"):
                read_images(images)
            images.write_text("1 1 0 0 0 0 0 0 1 photo.jpg\n\n")
            self.assertEqual(read_images(images)[1]["source_observations"],0)

    def test_rejects_unmeasured_or_bad_seed(self):
        camera=dict(width=800,height=600,fx=700,fy=700,cx=400,cy=300)
        rotation=quaternion_rotation([1,0,0,0])
        views=[dict(camera=camera,rotation=rotation,translation=[0,0,0]),
               dict(camera=camera,rotation=rotation,translation=[-1,0,0])]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"seeds.txt"
            path.write_text("")
            with self.assertRaisesRegex(ValueError,"no measured"):
                parse_seeds(path,views)
            p=[0,0,5]
            a=project(camera,rotation,views[0]["translation"],p)
            b=project(camera,rotation,views[1]["translation"],p)
            path.write_text(f"0 0 5 0 {a[0]} {a[1]} 1 {b[0]} {b[1]} 0 0\n")
            self.assertEqual(len(parse_seeds(path,views)[0]),1)
            path.write_text(f"0 0 5 0 {a[0]+3} {a[1]} 1 {b[0]} {b[1]} 0 0\n")
            with self.assertRaisesRegex(ValueError,"reprojection"):
                parse_seeds(path,views)


if __name__ == "__main__":
    unittest.main()
