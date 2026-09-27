"""Bounded delayed self-calibration controls; no native BA in unit tests."""

import unittest

from scripts.object_motion import initialization_delayed_refinement as delayed


class DelayedRefinementTests(unittest.TestCase):
    def test_pinned_bundle_options_are_the_only_intrinsic_changes(self):
        try:
            import pycolmap  # noqa: F401
        except ImportError:
            self.skipTest("pinned PyCOLMAP absent")
        options = delayed.ba_options()
        self.assertTrue(options.refine_focal_length)
        self.assertTrue(options.refine_extra_params)
        self.assertFalse(options.refine_principal_point)
        self.assertTrue(options.refine_extrinsics)
        self.assertFalse(options.use_gpu)
        self.assertEqual(options.solver_options.max_num_iterations, 50)
        self.assertEqual(options.solver_options.num_threads, 2)
        self.assertEqual(options.solver_options.max_solver_time_in_seconds, 50)
        self.assertNotEqual(options.solver_options.trust_region_problem_dump_directory, "/tmp")

    def test_training_residuals_must_be_finite_and_nonempty(self):
        good = {"training_reprojection_px": {"count": 4, "median": .3, "p95": 1.2}}
        self.assertTrue(delayed.finite_training(good))
        self.assertFalse(delayed.finite_training({"training_reprojection_px": {"count": 0, "median": None, "p95": None}}))
        self.assertFalse(delayed.finite_training({"training_reprojection_px": {"count": 4, "median": float("nan"), "p95": 1.2}}))


if __name__ == "__main__":
    unittest.main()
