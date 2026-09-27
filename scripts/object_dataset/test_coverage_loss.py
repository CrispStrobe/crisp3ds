"""Analytic hit-transition and pooled-count checks without live meshes."""

import hashlib
import unittest

import numpy as np

from scripts.object_dataset import coverage_loss as loss


class CoverageLossTests(unittest.TestCase):
    def test_all_four_hit_states_and_shared_hit_residuals(self):
        # Both hit, 008-only hit, 015-only hit, neither hit. No missing ray
        # contributes to the conditional shared-hit error distribution.
        result = loss.paired_summary([1, 1, 1, 1], [1.01, 1, None, None],
                                     [1.005, None, 1.02, None], [True] * 4)
        self.assertEqual(result["transitions"], {
            "supported": 4, "both_hit": 1, "lost_008_hit_015_missing": 1,
            "gained_008_missing_015_hit": 1, "both_missing": 1})
        self.assertAlmostEqual(result["shared_hit_residual_m"]["rough008_absolute"]["mean"], .01)
        self.assertAlmostEqual(result["shared_hit_residual_m"]["rough015_absolute"]["mean"], .005)
        self.assertEqual(result["shared_hit_residual_m"]["015_lower_absolute_count"], 1)
        self.assertEqual(result["missing_inclusive_within_threshold"]["0.005"]["rough008_count"], 1)
        self.assertEqual(result["missing_inclusive_within_threshold"]["0.005"]["rough015_count"], 1)
        self.assertEqual(result["shared_hit_within_threshold"]["0.010"]["denominator_both_hit"], 1)

    def test_empty_region_and_bad_values(self):
        result = loss.paired_summary([1, 2], [None, None], [None, None], [False, False])
        self.assertEqual(result["transitions"]["supported"], 0)
        self.assertIsNone(result["hit_fraction_all_supported"]["rough008"])
        self.assertIsNone(result["shared_hit_residual_m"]["rough008_absolute"])
        with self.assertRaises(ValueError):
            loss.paired_summary([1], [float("inf")], [1], [True])
        with self.assertRaises(ValueError):
            loss.paired_summary([0], [1], [1], [True])

    def test_three_views_reconcile_pooled_and_grid_partitions(self):
        frames = []
        candidates = {label: {"label": label, "mesh_sha256": label, "frames": []}
                      for label in ("rough008", "rough014", "rough015")}
        for angle in loss.ANGLES:
            ids = list(range(2048))
            digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
            frames.append({"angle": angle, "selected_indices": ids,
                           "selection": {"selected_sha256": digest}})
            interior = [i % 2 == 0 for i in ids]
            baseline = [1.0] * 2048
            recovered = [1.0] * 2048
            baseline[0] = None  # gained, in interior
            recovered[1] = None  # lost, in boundary band
            baseline[2] = recovered[2] = None  # both missing, in interior
            for label, predictions in (("rough008", baseline), ("rough014", baseline),
                                       ("rough015", recovered)):
                candidates[label]["frames"].append({"angle": angle, "observed_m": [1.0] * 2048,
                    "predicted_m": predictions, "interior_member": interior})
        report = {"schema": "berkeley_sensor_depth_v1", "status": "complete",
                  "protocol": {"max_shared_rays_per_view": 2048}, "frames": frames,
                  "candidates": list(candidates.values())}
        result = loss.analyze(report, "test-parent")
        pooled = result["pooled_regions"]
        self.assertEqual(pooled["coarse"]["transitions"]["supported"], 6144)
        self.assertEqual(pooled["coarse"]["transitions"]["gained_008_missing_015_hit"], 3)
        self.assertEqual(pooled["coarse"]["transitions"]["lost_008_hit_015_missing"], 3)
        self.assertEqual(pooled["interior"]["transitions"]["gained_008_missing_015_hit"], 3)
        self.assertEqual(pooled["boundary_band"]["transitions"]["lost_008_hit_015_missing"], 3)
        self.assertEqual(sum(cell["regions"]["coarse"]["transitions"]["supported"]
                             for cell in result["views"][0]["cells"]), 2048)
        report["candidates"][2]["frames"][0]["observed_m"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "candidate rays"):
            loss.analyze(report, "test-parent")


if __name__ == "__main__":
    unittest.main()
