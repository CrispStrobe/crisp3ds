import hashlib
import unittest

import numpy as np

from scripts.object_dataset import sensor_spatial as spatial


class SpatialTests(unittest.TestCase):
    def test_fixed_cell_boundaries(self):
        ids = [0, 159, 160, 640 * 119 + 159, 640 * 120,
               640 * 479 + 639]
        np.testing.assert_array_equal(spatial.cell_ids(ids), [0, 0, 1, 0, 4, 15])
        with self.assertRaises(ValueError):
            spatial.cell_ids([640 * 480])

    def test_missing_inclusive_rates_and_hit_only_errors(self):
        stats = spatial.summarize([1, 1, 1, 1], [1.004, 1.009, np.nan, 1.02],
                                  [True, True, True, False])
        self.assertEqual((stats["supported"], stats["first_hits"], stats["no_hit"]),
                         (3, 2, 1))
        self.assertAlmostEqual(stats["first_hit_fraction"], 2 / 3)
        self.assertAlmostEqual(stats["within_5mm_all_supported_fraction"], 1 / 3)
        self.assertAlmostEqual(stats["within_10mm_all_supported_fraction"], 2 / 3)
        self.assertAlmostEqual(stats["hit_only_absolute_m"]["mean"], .0065)

    def test_empty_cell_and_interior_are_explicit(self):
        rows = spatial.partition([0, 160], [1, 1], [1.002, np.nan], [True, False])
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["coarse"]["first_hits"], 1)
        self.assertEqual(rows[1]["coarse"]["no_hit"], 1)
        self.assertEqual(rows[1]["eroded_interior"]["supported"], 0)
        self.assertIsNone(rows[1]["eroded_interior"]["first_hit_fraction"])
        self.assertIsNone(rows[15]["coarse"]["within_5mm_all_supported_fraction"])

    def test_frozen_input_shared_support_validation(self):
        frames = []
        candidates = [{"label": label, "frames": []} for label in spatial.LABELS]
        ids = list(range(2048))
        digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
        for angle in spatial.ANGLES:
            frames.append({"angle": angle, "selected_indices": ids,
                           "selection": {"selected_sha256": digest}})
            for candidate in candidates:
                candidate["frames"].append({"angle": angle, "observed_m": [1.] * 2048,
                    "predicted_m": [1.] * 2047 + [None], "interior_member": [True] * 2048})
        report = {"schema": "berkeley_sensor_depth_v1", "status": "complete",
                  "frames": frames, "candidates": candidates}
        spatial._verified_input(report)
        candidates[1]["frames"][0]["observed_m"][0] = 1.1
        with self.assertRaisesRegex(ValueError, "support or prediction"):
            spatial._verified_input(report)
        candidates[1]["frames"][0]["observed_m"][0] = 1.
        frames[0]["selected_indices"][1] = 0
        with self.assertRaisesRegex(ValueError, "unique ordered"):
            spatial._verified_input(report)


if __name__ == "__main__":
    unittest.main()
