"""Analytic point-cloud support/residual tests; no live datasets or native binaries."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.classical_backend import fusion_point_diagnostic as target


class FusionPointDiagnosticTests(unittest.TestCase):
    def test_first_point_depth_and_fixed_pixel_support(self):
        k = np.eye(3)
        points = np.array([[3., 3., 1.], [3.6, 3.6, 1.2], [9., 9., 0.9]])
        selected = np.array([3 + 3 * 9, 5 + 5 * 9, 6 + 6 * 9])
        out = target.projected_cloud_depth(points, k, selected, width=9, height=9)
        self.assertAlmostEqual(out[0], 1.)
        self.assertAlmostEqual(out[1], 1.)  # Chebyshev radius two reaches (3,3).
        self.assertTrue(np.isnan(out[2]))

    def test_paired_transition_denominators_and_shared_accuracy(self):
        observed = np.ones(5)
        control = np.array([1.001, 1.002, np.nan, np.nan, 1.])
        ablation = np.array([1.003, np.nan, 1.004, np.nan, 1.])
        support = np.array([True, True, True, True, False])
        result = target.paired_summary(observed, control, ablation, support)
        self.assertEqual((result["supported"], result["both_hit"], result["control_only"],
                          result["ablation_only"], result["both_missing"]), (4, 1, 1, 1, 1))
        self.assertAlmostEqual(result["shared_control"]["mean_absolute_m"], .001)
        self.assertAlmostEqual(result["shared_ablation"]["mean_absolute_m"], .003)
        self.assertAlmostEqual(result["gained_ablation"]["mean_absolute_m"], .004)
        self.assertEqual(sum(result[key] for key in ("both_hit", "control_only", "ablation_only", "both_missing")),
                         result["supported"])

    def test_openmvs_variable_list_xyz_and_trailing_data(self):
        header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                  b"property float32 x\nproperty float32 y\nproperty float32 z\n"
                  b"property list uint8 uint32 view_indices\n"
                  b"property list uint8 float32 view_weights\nend_header\n")
        payload = struct.pack("<fffBIBf", 1., 2., 3., 1, 7, 1, .5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cloud.ply"
            path.write_bytes(header + payload)
            np.testing.assert_allclose(target.load_openmvs_cloud(path), [[1., 2., 3.]])
            path.write_bytes(header + payload + b"x")
            with self.assertRaisesRegex(ValueError, "trailing"):
                target.load_openmvs_cloud(path)


if __name__ == "__main__":
    unittest.main()
