"""Analytic sparse observation/pose-mask audit fixtures, no PyCOLMAP required."""

from types import SimpleNamespace as Row
from unittest import TestCase

import numpy as np

from scripts.object_motion import mustard_track_support as audit


class Camera:
    def __init__(self):
        self.model = Row(name="SIMPLE_RADIAL")
        self.width, self.height = 1280, 1024
        self.params = [1536.0, 640.0, 512.0, 0.0]


class Image:
    def __init__(self, image_id, name, xy):
        self.image_id, self.name = image_id, name
        self.points2D = [Row(xy=np.asarray(value, dtype=float), point3D_id=7) for value in xy]

    def projection_center(self):
        return np.asarray([float(self.image_id), 0.0, 1.0])


class Model:
    def __init__(self):
        self.images = {1: Image(1, "a.jpg", [(1.2, 1.2), (2.2, 1.2)]),
                       2: Image(2, "b.jpg", [(1.2, 1.2)])}
        self.cameras = {1: Camera()}
        self.points3D = {7: Row(xyz=[0.0, 0.0, 1.0], track=Row(elements=[
            Row(image_id=1, point2D_idx=0), Row(image_id=1, point2D_idx=1),
            Row(image_id=2, point2D_idx=0)]))}

    def reg_image_ids(self):
        return [1, 2]

    def num_reg_images(self):
        return 2


class MustardTrackSupportTests(TestCase):
    def test_all_observations_counted_without_collapsing_repeat_image(self):
        model = Model()
        masks = {"a.jpg": np.zeros((4, 4), np.uint8), "b.jpg": np.zeros((4, 4), np.uint8)}
        masks["a.jpg"][1, 1] = 255
        masks["b.jpg"][1, 1] = 255
        report = audit.model_observation_support(model, masks, ["a.jpg", "b.jpg"])
        self.assertEqual(report["observations"], {"observations": 3,
                         "inside_coarse_pose_mask": 2,
                         "outside_coarse_pose_mask": 1, "outside_image_grid": 0})
        self.assertEqual(report["tracks"]["duplicate_same_image_track_count"], 1)
        self.assertEqual(report["tracks"]["duplicate_same_image_observations"], 1)
        self.assertEqual(report["tracks"]["at_least_3_distinct_views"], 0)
        self.assertEqual(report["camera"]["params"], [1536.0, 640.0, 512.0, 0.0])

    def test_exact_duplicate_backlink_and_camera_nonfinite_rejected(self):
        model = Model()
        masks = {"a.jpg": np.ones((4, 4), np.uint8), "b.jpg": np.ones((4, 4), np.uint8)}
        model.points3D[7].track.elements[1].point2D_idx = 0
        with self.assertRaisesRegex(ValueError, "duplicate exact"):
            audit.model_observation_support(model, masks, ["a.jpg", "b.jpg"])
        model = Model()
        model.images[1].points2D[0].point3D_id = 9
        with self.assertRaisesRegex(ValueError, "backlink"):
            audit.model_observation_support(model, masks, ["a.jpg", "b.jpg"])
        model = Model()
        model.cameras[1].params[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            audit.model_observation_support(model, masks, ["a.jpg", "b.jpg"])
