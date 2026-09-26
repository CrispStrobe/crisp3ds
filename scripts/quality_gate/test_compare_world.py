"""Hand-derived tests for changed-pose, fixed-truth world scoring."""

import math
from pathlib import Path
import tempfile
import unittest

from scripts.quality_gate.compare_depth import read_camera
from scripts.quality_gate.compare_world import (
    compare_errors, rectangle_distance, score, world_point,
)


def camera(translation=(0, 0, 0)):
    return {"focal_length": (1.0,), "pixel_aspect": (1.0,),
            "principal_point": (.1, .1),
            "rotation": (1, 0, 0, 0, 1, 0, 0, 0, 1),
            "translation": translation, "radial_distortion": (0, 0)}


class WorldComparisonTests(unittest.TestCase):
    def test_radial_depth_backprojects_to_world_and_rectangle_distance(self):
        point = world_point(camera(), 3, 4, (5, 5), (5, 5), 10 * math.sqrt(2))
        self.assertEqual(tuple(round(value, 9) for value in point), (6.0, 8.0, 10.0))
        plane = {"x": [5, 6], "y": [7, 8], "z": 10}
        self.assertAlmostEqual(rectangle_distance(point, plane), 0)
        self.assertAlmostEqual(rectangle_distance((8, 8, 13), plane), math.sqrt(13))

    def test_missing_and_wrong_share_fixed_denominator(self):
        errors = [0.0, 3.0, None, 1.0]
        plane = [0.0, 3.0, None, 1.0]
        result = score(set(range(4)), errors, plane)
        self.assertEqual(result["coverage"], .75)
        self.assertEqual(result["all_valid_rectangle_bad_fraction"]["2"], .5)
        self.assertEqual(result["all_valid_rectangle_bad_fraction"]["5"], .25)

    def test_empty_truth_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            compare_errors((3, 3), [None] * 9, [], (3, 3),
                           [0.0] * 9, [0.0] * 9, camera(), camera(), 1)

    def test_malformed_or_nonfinite_scene_camera_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            view = scene / "views/view_0000.mve"
            view.mkdir(parents=True)
            template = ("focal_length = 1\npixel_aspect = 1\n"
                        "principal_point = .5 .5\nradial_distortion = 0 0\n"
                        "rotation = {rotation}\ntranslation = 0 0 0\n")
            path = view / "meta.ini"
            path.write_text(template.format(rotation="1 0 0 0 1 0 0 0 1"))
            read_camera(scene, 0)
            for rotation, message in (("1 0 0 0 1 0 0 0 nan", "invalid camera"),
                                      ("1 0 0 0 1 0 0 0 2", "nonorthogonal"),
                                      ("1 0 0 0 1 0 0 0 -1", "reflected")):
                path.write_text(template.format(rotation=rotation))
                with self.assertRaisesRegex(ValueError, message):
                    read_camera(scene, 0)

    def test_deleting_bad_prediction_cannot_pass(self):
        plane = {"x": [-100, 100], "y": [-100, 100], "z": 10}
        labels = [0] * 25
        baseline = [10.0] * 25
        candidate = [10.0] * 25
        baseline[0] = 13.0
        candidate[0] = 0.0
        result = compare_errors((5, 5), labels, [plane], (5, 5),
                                baseline, candidate, camera(), camera(), 1)
        self.assertEqual(result["baseline"]["full_object"]["all_valid_rectangle_bad_fraction"]["2"],
                         result["candidate"]["full_object"]["all_valid_rectangle_bad_fraction"]["2"])
        self.assertFalse(result["provisional_relative_gate_pass"])

    def test_changed_pose_is_compared_in_same_object_frame(self):
        plane = {"x": [-100, 100], "y": [-100, 100], "z": 10}
        labels = [0] * 25
        baseline = [10.0] * 25
        candidate = [10.0] * 25
        # Changed world-to-camera x translation shifts reconstructed X by -1 mm;
        # this has a physical 1 mm point error only if the finite plane is narrow.
        narrow = {"x": [0, 0], "y": [-100, 100], "z": 10}
        result = compare_errors((5, 5), labels, [narrow], (5, 5),
                                baseline, candidate, camera(), camera((1, 0, 0)), 1)
        self.assertNotEqual(result["baseline"]["full_object"]["matched_rectangle_mae_mm"],
                            result["candidate"]["full_object"]["matched_rectangle_mae_mm"])
        self.assertEqual(result["baseline"]["full_object"]["coverage"], 1)
        self.assertEqual(result["candidate"]["full_object"]["coverage"], 1)


if __name__ == "__main__":
    unittest.main()
