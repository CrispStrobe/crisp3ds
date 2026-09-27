"""Exact intrinsic-freeze option contract for the one exploratory mapper arm."""

import unittest

from scripts.object_motion import initialization_fixed_intrinsics as fixed


class FixedIntrinsicsTests(unittest.TestCase):
    def test_fixed_final_camera_gate(self):
        self.assertTrue(fixed.fixed_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                              [1536, 640, 512, 0]))
        self.assertFalse(fixed.fixed_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                               [1536.01, 640, 512, 0]))
        self.assertFalse(fixed.fixed_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                               [9950, 640, 512, -228]))

    def test_pinned_pipeline_options_freeze_every_exposed_intrinsic_refinement(self):
        try:
            import pycolmap  # noqa: F401 - the pinned extension's actual API is the contract.
        except ImportError:
            self.skipTest("pinned PyCOLMAP absent")
        options = fixed.fixed_options({"NP3_018.jpg": 4, "NP3_030.jpg": 6})
        self.assertEqual((options.init_image_id1, options.init_image_id2), (4, 6))
        self.assertEqual((options.num_threads, options.mapper.num_threads), (2, 2))
        self.assertFalse(options.ba_refine_focal_length)
        self.assertFalse(options.ba_refine_principal_point)
        self.assertFalse(options.ba_refine_extra_params)
        self.assertFalse(options.mapper.abs_pose_refine_focal_length)
        self.assertFalse(options.mapper.abs_pose_refine_extra_params)
        self.assertFalse(options.multiple_models)
        self.assertEqual(options.max_num_models, 1)


if __name__ == "__main__":
    unittest.main()
