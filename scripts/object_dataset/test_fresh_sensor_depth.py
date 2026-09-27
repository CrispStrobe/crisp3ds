"""Fail-closed checks for the frozen fresh sensor-depth adapter."""

import copy
import unittest

import numpy as np

from scripts.object_dataset import fresh_sensor_depth as target


class FreshSensorDepthTests(unittest.TestCase):
    def test_camera_similarity_requires_exact_model_and_proper_fit(self):
        model = {"cameras.bin": "a", "images.bin": "b", "points3D.bin": "c"}
        fit = {"schema": "ycb_stock_berkeley_camera_diagnostic_v1",
               "candidate_model_sha256": model,
               "reference_evidence": {"report_sha256":
                   "744b50dfcbf66c0801fa9f36c3a7eac124ba09f3053909c10e7c647d585a723e"},
               "comparison": {"paired_camera_count": 60, "missing_reference_names": [],
                              "extra_candidate_names": [],
                              "center_rms_over_reference_radius": .01,
                              "orientation_p95_degrees": 2,
                              "similarity_source_to_reference": {
                                  "scale": 2, "rotation": np.eye(3).tolist(),
                                  "translation": [1, 2, 3]}}}
        np.testing.assert_array_equal(target.matrix_from_fit(fit, model),
                                      [[2, 0, 0, 1], [0, 2, 0, 2],
                                       [0, 0, 2, 3], [0, 0, 0, 1]])
        with self.assertRaisesRegex(ValueError, "exact source"):
            target.matrix_from_fit(fit, {**model, "images.bin": "changed"})
        bad = copy.deepcopy(fit)
        bad["comparison"]["similarity_source_to_reference"]["rotation"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "preserve orientation"):
            target.matrix_from_fit(bad, model)
        bad = copy.deepcopy(fit)
        bad["comparison"]["center_rms_over_reference_radius"] = .05
        with self.assertRaisesRegex(ValueError, "fit gate"):
            target.matrix_from_fit(bad, model)


if __name__ == "__main__":
    unittest.main()
