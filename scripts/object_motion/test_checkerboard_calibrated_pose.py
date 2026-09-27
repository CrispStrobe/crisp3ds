"""Synthetic-only tests for the capture-calibration-assisted moving-board branch."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.object_motion import checkerboard_calibrated_pose as calibrated


class CalibratedCheckerboardTests(unittest.TestCase):
    def test_distorted_ippe_pose_and_global_180_label_gauge(self):
        k, d = calibrated.EXPECTED_K, calibrated.EXPECTED_D
        rvec = np.array([.2, -.15, .3], dtype=np.float64)
        translation = np.array([.3, -.2, 18.0], dtype=np.float64)
        image = cv2.projectPoints(calibrated.base.OBJECT_CORNERS, rvec, translation,
                                  k, d)[0].reshape(8, 9, 2)
        zero_distorted = cv2.projectPoints(calibrated.base.OBJECT_CORNERS, rvec, translation,
                                           k, np.zeros(5))[0].reshape(8, 9, 2)
        self.assertGreater(float(np.max(np.linalg.norm(image-zero_distorted, axis=2))), .1)
        candidates = calibrated.pose_candidates(image, k, d)
        self.assertEqual(len(candidates), 4)
        first = min((p for p in candidates if p['label_flip_180'] == 0), key=lambda p: p['rms_px'])
        reversed_label = min((p for p in candidates if p['label_flip_180'] == 1), key=lambda p: p['rms_px'])
        self.assertLess(first['rms_px'], 1e-6)
        self.assertAlmostEqual(first['rms_px'], reversed_label['rms_px'], places=6)
        self.assertLess(float(np.linalg.norm(first['translation']-translation)), 1e-5)
        truth_rotation = cv2.Rodrigues(rvec)[0]
        self.assertLess(float(np.linalg.norm(first['rotation']-truth_rotation)), 1e-5)

    def test_calibration_loader_accesses_only_np3_rgb_k_and_d(self):
        with tempfile.TemporaryDirectory() as scratch:
            calibration = Path(scratch) / 'calibration.h5'
            receipt = Path(scratch) / 'receipt.json'
            calibration.write_bytes(b'synthetic calibration')
            cal_hash = calibrated.base.digest(calibration)
            receipt.write_text(json.dumps({'status': 'metadata_only_no_camera_evaluation',
                                           'extraction': {'members': {
                                               '006_mustard_bottle/calibration.h5':
                                               {'bytes': 63544, 'sha256': cal_hash}}}}))
            calls = []

            def fake_dataset(path, key, shape):
                self.assertEqual(path, calibration)
                calls.append((key, shape))
                return calibrated.EXPECTED_K.copy() if key == '/NP3_rgb_K' else calibrated.EXPECTED_D.copy()

            with patch.object(calibrated, 'CALIBRATION', calibration), \
                    patch.object(calibrated, 'CALIBRATION_SHA256', cal_hash), \
                    patch.object(calibrated, 'RECEIPT', receipt), \
                    patch.object(calibrated, 'RECEIPT_SHA256', calibrated.base.digest(receipt)), \
                    patch.object(calibrated, 'dataset', side_effect=fake_dataset):
                k, d = calibrated.read_calibration()
            np.testing.assert_array_equal(k, calibrated.EXPECTED_K)
            np.testing.assert_array_equal(d, calibrated.EXPECTED_D)
            self.assertEqual(calls, [('/NP3_rgb_K', (3, 3)), ('/NP3_rgb_d', (5,))])

    def test_tampered_calibration_rejected_before_h5_dataset_access(self):
        with tempfile.TemporaryDirectory() as scratch:
            calibration = Path(scratch) / 'calibration.h5'
            calibration.write_bytes(b'tampered')
            with patch.object(calibrated, 'CALIBRATION', calibration), \
                    patch.object(calibrated, 'dataset', side_effect=AssertionError('opened H5')):
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    calibrated.read_calibration()

    def test_nonfinite_corners_rejected(self):
        corners = np.zeros((8, 9, 2), dtype=np.float64)
        corners[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            calibrated.pose_candidates(corners, calibrated.EXPECTED_K, calibrated.EXPECTED_D)


if __name__ == '__main__':
    unittest.main()
