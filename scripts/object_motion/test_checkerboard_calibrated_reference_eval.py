"""Synthetic adapter tests; no real Berkeley pose values or scores are loaded."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.object_motion import checkerboard_calibrated_reference_eval as adapter


def fake_report():
    views = []
    for angle in range(0, 288, 6):
        detected = angle < 234
        row = {'name': f'NP3_{angle:03}.jpg', 'detected': detected}
        if detected:
            row.update({'camera_from_board_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        'camera_from_board_translation_square_units': [0, 0, 10]})
        views.append(row)
    return {'schema': 'mustard_checkerboard_calibrated_assisted_pose_v1',
            'status': 'diagnostic_only', 'software_sha256': adapter.POSE_RUNNER_SHA256,
            'detected_count': 39, 'views': views}


class CalibratedReferenceAdapterTests(unittest.TestCase):
    def test_selected_views_match_exact_train_names_and_reject_duplicates(self):
        selected = adapter.selected_views(fake_report())
        self.assertEqual(len(selected), 39)
        self.assertIn('NP3_006.jpg', selected)
        changed = fake_report()
        changed['views'][1]['name'] = 'NP3_000.jpg'
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            adapter.selected_views(changed)

    def test_hash_only_preflight_never_loads_h5_numeric_pose(self):
        with tempfile.TemporaryDirectory() as scratch:
            base = Path(scratch)
            metadata_root = base / 'metadata-root'
            metadata = metadata_root / 'metadata'
            metadata.mkdir(parents=True)
            archive = metadata_root / '006_mustard_bottle_berkeley_rgbd.tgz'
            archive.write_bytes(b'synthetic archive')
            report_path = base / 'calibrated.json'
            report = fake_report()
            report_path.write_text(json.dumps(report))
            selected = adapter.selected_views(report)
            members = {'006_mustard_bottle/calibration.h5'} | {
                f'006_mustard_bottle/poses/NP5_{int(name[4:7])}_pose.h5' for name in selected}
            inventory = {}
            for name in members:
                path = metadata / Path(name).relative_to(adapter.PREFIX)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                inventory[name] = {'bytes': path.stat().st_size, 'sha256': adapter.shared.sha256(path)}
            receipt_path = metadata_root / 'receipt.json'
            receipt_path.write_text(json.dumps({
                'schema': 'mustard_pose_metadata_acquisition_v1',
                'status': 'metadata_only_no_camera_evaluation',
                'checkerboard_report_sha256': adapter.shared.POSE_REPORT_SHA256,
                'source': {'archive': {'sha256': adapter.shared.sha256(archive)}},
                'extraction': {'members': inventory}}))
            output = base / 'mustard-checkerboard-calibrated-camera-eval-synthetic.json'
            with patch.object(adapter, 'EXTERNAL_BASE', base), \
                    patch.object(adapter, 'METADATA_ROOT', metadata_root), \
                    patch.object(adapter, 'POSE_REPORT', report_path), \
                    patch.object(adapter, 'POSE_REPORT_SHA256', adapter.shared.sha256(report_path)), \
                    patch.object(adapter, 'RECEIPT', receipt_path), \
                    patch.object(adapter, 'RECEIPT_SHA256', adapter.shared.sha256(receipt_path)), \
                    patch.object(adapter, 'ARCHIVE_BYTES', archive.stat().st_size), \
                    patch.object(adapter, 'ARCHIVE_SHA256', adapter.shared.sha256(archive)), \
                    patch.object(adapter, 'SHARED_EVALUATOR_SHA256',
                                 adapter.shared.sha256(Path(adapter.shared.__file__))), \
                    patch.object(adapter, 'disk_space', return_value={'internal_free_bytes': 20*1024**3,
                                                                       'external_free_bytes': 20*1024**3}), \
                    patch.object(adapter.shared, 'dataset', side_effect=AssertionError('numeric H5 load')):
                summary = adapter.preflight(output)[0]
            self.assertFalse(summary['writes'])
            self.assertEqual(summary['matched'], 39)
            self.assertEqual(summary['metadata_member_count'], 40)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
