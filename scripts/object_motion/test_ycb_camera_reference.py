"""Frame-composition and strict HDF5 dump parsing contracts."""

import shutil
import unittest

import numpy as np

from scripts.object_motion import ycb_camera_reference as reference


class YcbCameraReferenceTests(unittest.TestCase):
    def test_parse_h5dump_shape_and_finite(self):
        example = '''DATASET "/test" {
   DATATYPE H5T_IEEE_F64LE
   DATASPACE SIMPLE { ( 2, 2 ) / ( 2, 2 ) }
   DATA {
   (0,0): 1, 2,
   (1,0): 3, 4
   }
}'''
        self.assertTrue(np.array_equal(reference.parse_h5dump(example, "/test", (2, 2)),
                                       [[1, 2], [3, 4]]))
        with self.assertRaisesRegex(ValueError, "shape"):
            reference.parse_h5dump(example, "/test", (4,))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            reference.parse_h5dump(example.replace("3, 4", "3, nan"), "/test", (2, 2))

    def test_rigid_and_rotation_error(self):
        identity = np.eye(4)
        self.assertTrue(np.array_equal(reference.rigid(identity, "test"), identity))
        bad = identity.copy()
        bad[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "proper rigid"):
            reference.rigid(bad, "bad")
        self.assertAlmostEqual(reference.rotation_errors({"a": np.eye(3)},
                                                          {"a": np.eye(3)}, np.eye(4))["a"], 0)

    def test_nonidentity_frame_composition_and_orientation(self):
        rz90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rz180 = np.diag([-1., -1., 1.])
        np3_from_np5 = np.eye(4)
        np3_from_np5[:3, :3] = rz90
        np3_from_np5[:3, 3] = [1, 0, 0]
        table_from_np5 = np.eye(4)
        table_from_np5[:3, :3] = rz180
        table_from_np5[:3, 3] = [0, 2, 0]
        composed = reference.np3_from_table(np3_from_np5, table_from_np5)
        np.testing.assert_allclose(composed[:3, :3], rz90 @ rz180, atol=1e-12)
        np.testing.assert_allclose(composed[:3, 3], [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(-composed[:3, :3].T @ composed[:3, 3], [0, 1, 0], atol=1e-12)
        alignment = np.eye(4)
        alignment[:3, :3] = 2 * rz90
        self.assertLess(reference.rotation_errors({"a": np.eye(3)},
                                                  {"a": rz90.T}, alignment)["a"], 1e-6)

    def test_real_sample_datasets_when_available(self):
        calibration = (reference.ROOT / ".local-tools/test-data/ycb-cracker-box/"
                       "metadata-reference-frames-001/003_cracker_box/calibration.h5")
        if not calibration.exists() or shutil.which("h5dump") is None:
            self.skipTest("small source metadata or h5dump unavailable")
        h = reference.dataset(calibration, "/H_NP3_from_NP5", (4, 4))
        self.assertEqual(h.shape, (4, 4))
        self.assertGreater(reference.dataset(calibration, "/NP3_rgb_K", (3, 3))[0, 0], 1000)
        reference.rigid(h, "NP3")


if __name__ == "__main__":
    unittest.main()
