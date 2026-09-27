"""Synthetic-only tests: no Berkeley HDF5 or real camera score is read."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.object_motion import checkerboard_pose_reference_eval as reference


def rot_z(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])


class CheckerboardReferenceEvalTests(unittest.TestCase):
    def setUp(self):
        self.names = [f'NP3_{index*6:03}.jpg' for index in range(8)]
        self.source_centers = {name: np.array([math.cos(i*.7)*3, math.sin(i*.7)*2, i*.1])
                               for i, name in enumerate(self.names)}
        self.source_rotations = {name: rot_z(i*9) for i, name in enumerate(self.names)}
        self.scale, self.q = .023, rot_z(63)
        self.offset = np.array([.2, -.1, .7])
        self.target_centers = {name: self.scale*self.q@center + self.offset
                               for name, center in self.source_centers.items()}
        self.target_rotations = {name: rotation@self.q.T
                                 for name, rotation in self.source_rotations.items()}

    def test_exact_positive_proper_sim3_and_shared_orientation_rotation(self):
        result = reference.compare(self.source_centers, self.source_rotations,
                                   self.target_centers, self.target_rotations)
        self.assertEqual(result['matched_count'], 8)
        self.assertAlmostEqual(result['fit']['scale_supplied_units_per_square'], self.scale, places=12)
        self.assertAlmostEqual(result['fit']['rotation_determinant'], 1.0, places=12)
        self.assertLess(result['center_residual_over_supplied_median_radius']['max'], 1e-12)
        self.assertLess(result['leave_one_out_center_residual_over_supplied_median_radius']['max'], 1e-12)
        self.assertLess(result['orientation_residual_degrees']['max'], 2e-6)

    def test_global_180_board_label_gauge_is_absorbed_by_proper_sim3(self):
        gauge = rot_z(180)
        changed_centers = {name: gauge@center for name, center in self.source_centers.items()}
        changed_rotations = {name: rotation@gauge.T for name, rotation in self.source_rotations.items()}
        result = reference.compare(changed_centers, changed_rotations,
                                   self.target_centers, self.target_rotations)
        self.assertLess(result['center_residual_over_supplied_median_radius']['max'], 1e-12)
        self.assertLess(result['orientation_residual_degrees']['max'], 2e-6)

    def test_reject_mismatch_and_degenerate_center_constellation(self):
        with self.assertRaisesRegex(ValueError, 'names'):
            reference.compare(self.source_centers, self.source_rotations,
                              dict(list(self.target_centers.items())[:-1]), self.target_rotations)
        line = {name: [index, 0, 0] for index, name in enumerate(self.names)}
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            reference.fit_similarity(list(line.values()), list(line.values()))

    def test_perturbed_frame_has_nonzero_leave_one_out_error(self):
        perturbed = dict(self.target_centers)
        perturbed[self.names[3]] = perturbed[self.names[3]] + np.array([.012, -.004, .003])
        result = reference.compare(self.source_centers, self.source_rotations,
                                   perturbed, self.target_rotations)
        row = result['per_frame'][self.names[3]]
        self.assertGreater(row['leave_one_out_center_residual_over_radius'],
                           row['center_residual_over_radius'])
        self.assertGreater(row['leave_one_out_center_residual_over_radius'], .01)

    def test_reflection_cannot_be_absorbed_by_proper_sim3(self):
        mirror = np.diag([-1., 1., 1.])
        reflected = {name: mirror@center for name, center in self.target_centers.items()}
        result = reference.compare(self.source_centers, self.source_rotations,
                                   reflected, self.target_rotations)
        self.assertAlmostEqual(result['fit']['rotation_determinant'], 1.0, places=12)
        self.assertGreater(result['center_residual_over_supplied_median_radius']['rms'], .01)

    def test_read_only_preflight_does_not_load_reference_h5_values(self):
        output = reference.EXTERNAL_BASE / 'mustard-checkerboard-camera-eval-synthetic-never-written.json'
        fake_receipt = {'extraction': {'members': {'calibration': {}, **{str(i): {} for i in range(39)}}}}
        with patch.object(reference, 'inspect_inputs', return_value=({}, fake_receipt, dict.fromkeys(self.names))), \
                patch.object(reference, 'disk_space', return_value={'internal_free_bytes': 20*1024**3,
                                                                      'external_free_bytes': 20*1024**3}), \
                patch.object(reference, 'dataset', side_effect=AssertionError('H5 numeric load in preflight')):
            result = reference.preflight(output)[0]
        self.assertFalse(result['writes'])
        self.assertEqual(result['status'], 'ready_without_camera_scoring')

    def test_postflight_rehash_catches_metadata_tampering(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            report = base / 'image-report.json'
            receipt_path = base / 'receipt.json'
            h5 = base / 'metadata' / 'calibration.h5'
            h5.parent.mkdir()
            report.write_bytes(b'image-report')
            receipt_path.write_bytes(b'receipt')
            h5.write_bytes(b'calibration')
            record = {'bytes': h5.stat().st_size, 'sha256': reference.sha256(h5)}
            receipt = {'extraction': {'members': {'006_mustard_bottle/calibration.h5': record}}}
            with patch.object(reference, 'POSE_REPORT', report), \
                    patch.object(reference, 'POSE_REPORT_SHA256', reference.sha256(report)), \
                    patch.object(reference, 'RECEIPT', receipt_path), \
                    patch.object(reference, 'RECEIPT_SHA256', reference.sha256(receipt_path)), \
                    patch.object(reference, 'METADATA_ROOT', base):
                self.assertEqual(reference.verify_postflight(receipt)['verified_h5_members'], 1)
                h5.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'post-score metadata'):
                    reference.verify_postflight(receipt)

    def test_orientation_uses_center_fit_q_without_independent_rotation_fit(self):
        biased = {name: rot_z(20)@rotation for name, rotation in self.target_rotations.items()}
        result = reference.compare(self.source_centers, self.source_rotations,
                                   self.target_centers, biased)
        self.assertAlmostEqual(result['orientation_residual_degrees']['median'], 20, places=5)
        self.assertLess(result['center_residual_over_supplied_median_radius']['max'], 1e-12)


if __name__ == '__main__':
    unittest.main()
