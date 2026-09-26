"""Hand-counted checks for fixed-denominator radial depth comparisons."""

import math
import unittest

from scripts.quality_gate.compare_depth import (
    compare_arrays, regions_from_truth, require_paired, truth_at,
)


class QualityGateTests(unittest.TestCase):
    def test_missing_and_wrong_both_count_on_full_truth_population(self):
        truth = [10.0] * 9
        baseline = [10.0] * 7 + [13.0, 0.0]
        candidate = [10.0] * 7 + [0.0, 0.0]
        result = compare_arrays((3, 3), truth, baseline, candidate, 1)
        base = result["baseline"]["full_object"]
        cand = result["candidate"]["full_object"]
        self.assertEqual(base["all_valid_bad_fraction"]["2"], 2 / 9)
        self.assertEqual(cand["all_valid_bad_fraction"]["2"], 2 / 9)
        self.assertEqual(base["matched_mae_mm"], 3 / 8)
        self.assertEqual(cand["matched_mae_mm"], 0)
        self.assertFalse(result["provisional_relative_gate_pass"])
        self.assertEqual(result["shared_support_selection_biased"]["full_object"]["pixels"], 7)

    def test_fixed_truth_boundary_and_interior(self):
        truth = [None] * 25
        for y in range(1, 4):
            for x in range(1, 4):
                truth[y * 5 + x] = 10.0
        regions = regions_from_truth(truth, (5, 5), 1)
        self.assertEqual(len(regions["full_object"]), 9)
        self.assertEqual(len(regions["boundary_band"]), 8)
        self.assertEqual(regions["interior"], {12})
        regions_2 = regions_from_truth(truth, (5, 5), 2)
        self.assertEqual(len(regions_2["boundary_band"]), 9)

    def test_adjacent_surface_step_is_boundary_even_inside_object(self):
        truth = [10.0] * 25
        plane_ids = [0 if x < 2 else 1 for y in range(5) for x in range(5)]
        regions = regions_from_truth(truth, (5, 5), 1, plane_ids)
        self.assertIn(12, regions["depth_step_band"])
        self.assertNotIn(12, regions["silhouette_band"])
        self.assertIn(12, regions["boundary_band"])

    def test_off_axis_truth_is_radial_and_uses_estimated_camera(self):
        camera = {"focal_length": (1.0,), "pixel_aspect": (1.0,),
                  "principal_point": (0.05, 0.05),
                  "rotation": (1, 0, 0, 0, 1, 0, 0, 0, 1),
                  "translation": (0, 0, 0), "radial_distortion": (0, 0)}
        plane = {"x": [5.9, 6.1], "y": [7.9, 8.1], "z": 10}
        self.assertAlmostEqual(truth_at(6, 8, (10, 10), camera, (10, 10), [plane]),
                               10 * math.sqrt(2))
        self.assertIsNone(truth_at(0, 0, (10, 10), camera, (10, 10), [plane]))

    def test_unpaired_resolution_and_camera_rejected(self):
        camera = {"rotation": (1, 0, 0), "focal_length": (1.0,)}
        require_paired((10, 10), (10, 10), camera, dict(camera))
        with self.assertRaisesRegex(ValueError, "resolutions"):
            require_paired((10, 10), (5, 5), camera, camera)
        with self.assertRaisesRegex(ValueError, "camera metadata"):
            require_paired((10, 10), (10, 10), camera,
                           {"rotation": (1, 0, 0), "focal_length": (1.1,)})

    def test_candidate_cannot_gain_by_dropping_bad_edge_depth(self):
        truth = [10.0] * 25
        baseline = [10.0] * 25
        candidate = [10.0] * 25
        baseline[0] = 13.0
        candidate[0] = 0.0
        result = compare_arrays((5, 5), truth, baseline, candidate, 1)
        self.assertEqual(result["baseline"]["boundary_band"]["all_valid_bad_fraction"]["2"],
                         result["candidate"]["boundary_band"]["all_valid_bad_fraction"]["2"])
        self.assertFalse(result["provisional_relative_gate_pass"])


if __name__ == "__main__":
    unittest.main()
