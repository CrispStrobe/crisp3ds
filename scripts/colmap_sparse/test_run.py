"""Tests for the global feature holdout boundary."""

import unittest
from unittest import mock
from pathlib import Path
import tempfile

import numpy as np

from scripts.colmap_sparse import run
from scripts.colmap_sparse.run import holdout_partition, heldout_correspondences


class HoldoutTests(unittest.TestCase):
    def test_orientations_at_same_position_and_scale_stay_together(self):
        rows = np.array([[float(i), 2.0, 3.0, 0.0] for i in range(100)], np.float32)
        rows = np.repeat(rows, 2, axis=0)
        rows[1::2, 3] = 1.0
        heldout, train = holdout_partition(rows, "view.png")
        self.assertEqual(len(heldout), 40)
        self.assertEqual(len(set(heldout) & set(train)), 0)
        self.assertEqual(len(heldout) + len(train), len(rows))
        for i in range(0, len(rows), 2):
            self.assertEqual(i in heldout, i + 1 in heldout)

    def test_validation_matches_refer_only_to_heldout_ids(self):
        a = np.array([[0, 0], [100, 0], [0, 100], [80, 80]], np.uint8)
        b = np.array([[0, 0], [100, 0], [0, 100], [80, 80]], np.uint8)
        pairs = heldout_correspondences(a, [0, 2, 3], b, [0, 2, 3])
        self.assertEqual(pairs, [(0, 0), (2, 2), (3, 3)])

    def test_rejects_other_keypoint_layouts_and_invalid_scale(self):
        run.validate_keypoint_layout(np.ones((2, 4), np.float32))
        run.validate_keypoint_layout(np.ones((2, 6), np.float32))
        with self.assertRaisesRegex(ValueError, "unsupported COLMAP keypoint layout"):
            run.validate_keypoint_layout(np.ones((2, 5), np.float32))
        bad = np.ones((2, 4), np.float32)
        bad[0, 2] = 0
        with self.assertRaisesRegex(ValueError, "invalid COLMAP keypoint"):
            run.validate_keypoint_layout(bad)

    def test_supervisor_passes_requested_fresh_output_to_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp) / "staging"
            staging.mkdir()
            requested = Path(temp) / "runs" / "replay"
            with mock.patch.object(run, "STAGING", staging), \
                 mock.patch.object(run, "stage_images", return_value=[]), \
                 mock.patch("scripts.research_job.guard.run_child",
                            return_value={"status": "succeeded"}) as guard:
                run.main(["--baseline", "--output", str(requested)])
            self.assertEqual(guard.call_args.kwargs["output_dir"], requested)
            command = guard.call_args.args[0]
            self.assertEqual(command[command.index("--output") + 1], str(requested))
            self.assertIn("--baseline", command)
            self.assertTrue(requested.parent.is_dir())

    def test_effective_options_include_frozen_camera_and_cpu_limits(self):
        import pycolmap
        reader, sift, matching, _, _, mapper = run.build_options(pycolmap)
        self.assertEqual(reader.camera_params, "921.6,384,256,0")
        self.assertEqual(sift.max_num_features, 3000)
        self.assertEqual(sift.num_threads, 2)
        self.assertEqual(matching.num_threads, 2)
        self.assertEqual(mapper.num_threads, 2)
        self.assertFalse(mapper.ba_refine_principal_point)
        self.assertIsInstance(run.COLMAP_RANDOM_SEED, int)

    def test_baselines_use_default_sift_limit_and_no_supplied_camera_params(self):
        import pycolmap
        for originals, model, mode in ((True, "SIMPLE_RADIAL", pycolmap.CameraMode.AUTO),
                                       (False, "SIMPLE_PINHOLE", pycolmap.CameraMode.SINGLE)):
            reader, sift, _, _, _, mapping, selected_mode = run.baseline_options(
                pycolmap, originals)
            self.assertEqual(reader.camera_model, model)
            self.assertEqual(reader.camera_params, "")
            self.assertEqual(reader.default_focal_length_factor, 1.2)
            self.assertEqual(sift.max_num_features, 8192)
            self.assertEqual(sift.max_image_size, 3200)
            self.assertEqual(selected_mode, mode)
            self.assertTrue(mapping.multiple_models)

    def test_changed_input_or_software_hash_fails(self):
        run.assert_unchanged({"a": "before"}, {"a": "before"})
        with self.assertRaisesRegex(ValueError, "changed during reconstruction"):
            run.assert_unchanged({"a": "before"}, {"a": "after"})


if __name__ == "__main__":
    unittest.main()
