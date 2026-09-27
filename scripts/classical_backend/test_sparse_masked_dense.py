"""Pure preflight tests; no native reconstruction is launched."""

import unittest

from pathlib import Path
from unittest.mock import patch

from scripts.classical_backend.sparse_masked_dense import (REPAIR_GATES, depth_budget,
                                                           finite_positive_depth,
                                                           mask_warp_manifest, native_plan,
                                                           validate_repair_metrics, validate_seals,
                                                           worker_python)


class SparseMaskedDenseBudgetTests(unittest.TestCase):
    def test_frozen_48_view_full_detail_fits_three_gib_not_two(self):
        sizes = {f"NP3_{index:03}.jpg": (1280, 1024) for index in range(48)}
        profile = depth_budget(sizes, 64 << 20)
        self.assertEqual(profile["requested_level"], 0)
        self.assertEqual(profile["images"]["NP3_000.jpg"]["pixel_budget_size"], [1280, 1024])
        self.assertGreater(profile["predicted_two_generation_peak_bytes"], 2 << 30)
        self.assertLess(profile["predicted_two_generation_peak_bytes"], 3 << 30)
        with self.assertRaisesRegex(ValueError, "exceeds output cap"):
            depth_budget(sizes, 64 << 20, 2 << 30)

    def test_undistorted_larger_than_original_is_capped_at_1280(self):
        profile = depth_budget({"view.jpg": (1400, 1060)}, 1)
        self.assertEqual(profile["images"]["view.jpg"]["predicted_max_resolution"], 1280)
        self.assertEqual(profile["images"]["view.jpg"]["pixel_budget_size"], [1280, 970])

    def test_invalid_budget_rejected(self):
        for images, used, cap in (({}, 0, 3 << 30), ({"a": (1280, 1024)}, -1, 3 << 30),
                                  ({"a": (1280, 1024)}, 0, 0), ({"a": (0, 1024)}, 0, 3 << 30)):
            with self.subTest(images=images, used=used, cap=cap), self.assertRaises(ValueError):
                depth_budget(images, used, cap)

    def test_native_handoffs_keep_masks_and_dense_geometry_explicit(self):
        output = (Path("work") / "candidate").resolve()
        with patch("scripts.classical_backend.sparse_masked_dense.tool_path",
                   side_effect=lambda directory, name: directory / name):
            plan = dict(native_plan(output, Path("tools")))
        self.assertEqual(list(plan), ["import", "densify", "rough_mesh"])
        self.assertEqual(plan["densify"][plan["densify"].index("--resolution-level") + 1], "0")
        self.assertEqual(plan["densify"][plan["densify"].index("--max-resolution") + 1], "1280")
        self.assertEqual(plan["densify"][plan["densify"].index("--ignore-mask-label") + 1], "0")
        self.assertEqual(plan["densify"][plan["densify"].index("--mask-path") + 1], str(output / "masks"))
        self.assertEqual(plan["rough_mesh"][plan["rough_mesh"].index("-i") + 1],
                         str(output / "dense.mvs"))
        self.assertEqual(plan["rough_mesh"][plan["rough_mesh"].index("-p") + 1],
                         str(output / "dense.ply"))
        self.assertNotIn("RefineMesh", repr(plan))

    def test_worker_python_preserves_venv_executable_path(self):
        with patch("scripts.classical_backend.sparse_masked_dense.sys.executable",
                   str(Path("isolated-venv") / "bin" / "python")):
            with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve venv symlink")):
                self.assertEqual(worker_python(),
                                 str((Path("isolated-venv") / "bin" / "python").absolute()))

    def test_depth_guard_rejects_nonfinite_and_nonpositive(self):
        self.assertTrue(finite_positive_depth(0.001))
        for value in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertFalse(finite_positive_depth(value))

    def test_repair_gate_requires_complete_true_set_and_real_denominators(self):
        report = {"schema": "mustard_sparse_track_repair_v1", "status": "candidate_unreviewed",
                  "baseline": {"points": 100, "observations": 1000},
                  "output": {"points": 90, "observations": 900,
                             "duplicate_same_image_tracks": 0},
                  "gates": {name: True for name in REPAIR_GATES},
                  "repair": {"surviving_point_original_observations": 950,
                             "preserved_camera_and_pose": True, "backlink_valid": True,
                             "duplicate_track_count_after": 0},
                  "residuals": {"baseline_common_original": {"mean_pixels": 1.0, "p95_pixels": 3.0,
                                                               "denominator": 950, "invalid_count": 0},
                                "repaired_all_original_common": {"mean_pixels": 1.1, "p95_pixels": 3.3,
                                                                  "denominator": 950, "invalid_count": 0},
                                "repaired_selected": {"denominator": 900, "invalid_count": 0}}}
        self.assertEqual(validate_repair_metrics(report)["observations"], 900)
        for key in REPAIR_GATES:
            bad = {**report, "gates": {**report["gates"], key: False}}
            with self.subTest(gate=key), self.assertRaises(ValueError):
                validate_repair_metrics(bad)
        with self.assertRaises(ValueError):
            validate_repair_metrics({**report, "gates": {}})
        with self.assertRaises(ValueError):
            validate_repair_metrics({**report, "output": {**report["output"], "observations": 899}})
        with self.assertRaises(ValueError):
            validate_repair_metrics({**report, "residuals": {**report["residuals"],
                                    "repaired_all_original_common": {"mean_pixels": 1.2, "p95_pixels": 3.3,
                                                                    "denominator": 950, "invalid_count": 0}}})

    def test_mask_manifest_requires_exact_stage_names_and_hashes(self):
        names = ["NP3_000.jpg", "NP3_006.jpg"]
        stage = {"schema": "mustard_train_only_stage_v1", "status": "complete",
                 "train_names": names,
                 "cleaned_masks": {name: {"sha256": "a" * 64} for name in names}}
        self.assertEqual([row["name"] for row in mask_warp_manifest(stage, names)["images"]], names)
        with self.assertRaises(ValueError):
            mask_warp_manifest({**stage, "cleaned_masks": {names[0]: {"sha256": "a" * 64}}}, names)
        with self.assertRaises(ValueError):
            mask_warp_manifest(stage, names[::-1])
        with self.assertRaises(ValueError):
            mask_warp_manifest({**stage, "cleaned_masks": {names[0]: {"sha256": "bad"},
                                                    names[1]: {"sha256": "a" * 64}}}, names)

    def test_repair_qa_and_lineage_must_bind_same_bytes(self):
        names = [f"NP3_{i:03}.jpg" for i in range(48)]
        photo_hashes = {name: "a" * 64 for name in names}
        mask_hashes = {name: "b" * 64 for name in names}
        baseline_hashes = {name: "c" * 64 for name in ("cameras.bin", "images.bin", "points3D.bin")}
        output_hashes = {name: "d" * 64 for name in baseline_hashes}
        stage = {"schema": "mustard_train_only_stage_v1", "status": "complete",
                 "train_names": names, "train_photo_sha256": photo_hashes,
                 "cleaned_masks": {name: {"sha256": mask_hashes[name]} for name in names}}
        report = {"schema": "mustard_sparse_track_repair_v1", "status": "candidate_unreviewed",
                  "baseline": {"points": 100, "observations": 1000, "registered_names": names,
                               "result_sha256": "1" * 64, "model_files_sha256": baseline_hashes},
                  "stage": {"report_sha256": "2" * 64, "train_photo_sha256": photo_hashes,
                            "cleaned_mask_sha256": mask_hashes},
                  "output": {"points": 90, "observations": 900, "registered_names": names,
                             "duplicate_same_image_tracks": 0, "model_files_sha256": output_hashes},
                  "repair": {"surviving_point_original_observations": 950,
                             "preserved_camera_and_pose": True, "backlink_valid": True,
                             "duplicate_track_count_after": 0},
                  "gates": {name: True for name in REPAIR_GATES},
                  "residuals": {"baseline_common_original": {"mean_pixels": 1.0, "p95_pixels": 3.0,
                                                               "denominator": 950, "invalid_count": 0},
                                "repaired_all_original_common": {"mean_pixels": 1.0, "p95_pixels": 3.0,
                                                                  "denominator": 950, "invalid_count": 0},
                                "repaired_selected": {"denominator": 900, "invalid_count": 0}}}
        supervisor = {"schema": "mustard_sparse_track_repair_supervisor_v1",
                      "status": "candidate_unreviewed", "reason": None,
                      "seconds": 10.0, "peak_worker_rss_kib": 1000}
        qa = {"schema": "mustard_sparse_track_repair_qa_v1", "status": "accepted",
              "report_sha256": "3" * 64, "supervisor_sha256": "4" * 64,
              "model_files_sha256": output_hashes}
        params = {"report_sha": "3" * 64, "supervisor_sha": "4" * 64,
                  "baseline_result_sha": "1" * 64,
                  "baseline_model_hashes": baseline_hashes, "stage_report_sha": "2" * 64,
                  "output_model_hashes": output_hashes}
        self.assertEqual(validate_seals(report, supervisor, qa, stage, **params), names)
        with self.assertRaises(ValueError):
            validate_seals(report, supervisor, {**qa, "report_sha256": "4" * 64}, stage, **params)
        with self.assertRaises(ValueError):
            validate_seals(report, {**supervisor, "status": "failed"}, qa, stage, **params)
        with self.assertRaises(ValueError):
            validate_seals(report, supervisor, {**qa, "supervisor_sha256": "0" * 64}, stage, **params)
        with self.assertRaises(ValueError):
            validate_seals({**report, "stage": {**report["stage"], "cleaned_mask_sha256": {}}},
                           supervisor, qa, stage, **params)
        with self.assertRaises(ValueError):
            validate_seals({**report, "output": {**report["output"], "registered_names": names[:-1]}},
                           supervisor, qa, stage, **params)


if __name__ == "__main__":
    unittest.main()
