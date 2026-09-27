import json
from pathlib import Path
import unittest

import numpy as np

from scripts.object_motion import sam_point_trial as point
from scripts.object_motion import ycb_object_masks as common


class SAMPointTrialTests(unittest.TestCase):
    def test_frozen_manifest_and_original_pixel_coordinates(self):
        manifest = Path(__file__).parents[2] / "tests" / "datasets" / "sam21_mustard_point_prompts.json"
        self.assertEqual(common.digest(manifest), point.PROMPT_SHA)
        document = json.loads(manifest.read_text())
        rows = [{"path": f"photos/{item['name']}", "sha256": item["source_sha256"]}
                for item in document["images"]]
        prompts = point.validated_prompts(manifest, rows)
        self.assertEqual([item["name"] for item in prompts], list(point.NAMES))
        for item in prompts:
            xy, labels = point.point_arrays(item["points_xy_label"])
            self.assertEqual(xy.dtype, np.float32)
            self.assertEqual(xy[0].tolist(), [600.0, 525.0])
            self.assertEqual(labels.tolist(), [1, 0, 0])

    def test_bounds_duplicate_and_labels_rejected(self):
        valid = [[600, 525, 1], [720, 620, 0], [700, 400, 0]]
        for bad in ([(480, 525, 1), [720, 620, 0], [760, 400, 0]],
                    [[600, 525, 1], [600, 525, 0], [700, 400, 0]],
                    [[600, 525, 1], [720, 620, 1], [700, 400, 0]],
                    [[600, 525, True], [720, 620, 0], [700, 400, 0]]):
            with self.assertRaises(ValueError):
                point.point_arrays(bad)
        xy, labels = point.point_arrays(valid)
        self.assertEqual(xy.shape, (3, 2))
        self.assertEqual(labels.shape, (3,))

    def test_membership_fail_closed_and_binary_raw(self):
        prompts = [[600, 525, 1], [720, 620, 0], [700, 400, 0]]
        output = np.zeros((1, 1024, 1280), dtype=np.float32)
        output[0, 525, 600] = 1
        raw = point.binary_raw(output)
        self.assertEqual(point.point_membership(raw, prompts), [1, 0, 0])
        raw[525, 600] = 0
        with self.assertRaisesRegex(ValueError, "violates"):
            point.point_membership(raw, prompts)
        raw[525, 600] = 255
        raw[620, 720] = 255
        with self.assertRaisesRegex(ValueError, "violates"):
            point.point_membership(raw, prompts)
        output[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonbinary"):
            point.binary_raw(output)


if __name__ == "__main__":
    unittest.main()
