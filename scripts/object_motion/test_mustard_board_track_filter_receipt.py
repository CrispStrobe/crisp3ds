"""Synthetic-only counts receipt tests; no sealed DB feature blobs are read."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.object_motion import mustard_board_track_filter_receipt as runner
from scripts.object_motion.board_frame_candidate_filter import AcceptedTrack, FilterResult
from scripts.object_motion.board_pose_sparse_contract import Observation


class BoardTrackFilterReceiptTests(unittest.TestCase):
    def test_summary_conserves_tracks_and_leaks_no_points(self):
        observation = Observation('NP3_000.jpg', 42., 43.)
        accepted = AcceptedTrack(123456, (observation,), (12345.6, 23456.7, -2.),
                                 2., 4., .25, .2)
        result = FilterResult(518, (accepted,), {'board_plane_footprint': 517}, 9, -1)
        summary = runner.summarize(result, 48)
        self.assertEqual(summary['accepted_tracks'], 1)
        self.assertEqual(summary['rejected_by_reason']['board_plane_footprint'], 517)
        self.assertEqual(summary['accepted_posed_name_count'], 1)
        self.assertFalse(summary['sparse_minimum_eight_met'])
        serialized = json.dumps(summary)
        self.assertNotIn('12345.6', serialized)
        self.assertNotIn('23456.7', serialized)
        self.assertNotIn('123456', serialized)
        with self.assertRaisesRegex(ValueError, 'conservation'):
            runner.summarize(FilterResult(518, (accepted,), {'board_plane_footprint': 516}, 0, -1), 48)

    def test_worker_uses_one_extraction_and_returns_counts_only(self):
        names = [f'NP3_{angle:03}.jpg' for angle in range(0, 288, 6)]
        views = []
        for index, name in enumerate(names):
            row = {'name': name, 'detected': index < 39}
            if index < 39:
                row.update({'camera_from_board_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            'camera_from_board_translation_square_units': [0, 0, 10]})
            views.append(row)
        pose = {'intrinsics': {'K': [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                               'distortion': [0, 0, 0, 0, 0]}, 'views': views}
        sources = tuple(runner.ImageSource(name, 'a'*64, 'b'*64, 640, 480) for name in names)
        preflight = {'pose_sha256': runner.seal.POSE_SHA256,
                     'track_receipt_sha256': runner.seal.TRACK_RECEIPT_SHA256,
                     'database_sha256': runner.seal.DATABASE_SHA256,
                     'disk_free_bytes': {'internal_free_bytes': 20*1024**3,
                                         'external_free_bytes': 20*1024**3}}
        accepted = AcceptedTrack(7, (), (9999., 8888., -2.), 2., 4., .1, .1)
        filtered = FilterResult(518, (accepted,), {'reprojection': 517}, 0, -1)
        with patch.object(runner, 'preflight', return_value=preflight) as checked, \
                patch.object(runner.seal, '_json', side_effect=[pose, {}, {}]), \
                patch.object(runner, 'sources_from_reports', return_value=sources), \
                patch.object(runner, 'extract_candidate_tracks',
                             return_value=SimpleNamespace(tracks=tuple(range(518)))) as extracted, \
                patch.object(runner, 'filter_candidate_tracks', return_value=filtered) as filtered_call:
            report = runner._worker_receipt(Path('/synthetic/output.json'))
        self.assertEqual(checked.call_count, 2)
        extracted.assert_called_once()
        filtered_call.assert_called_once()
        self.assertEqual(report['counts']['accepted_tracks'], 1)
        self.assertEqual(report['status'], 'diagnostic_only_no_sparse_or_dense_model')
        self.assertNotIn('9999', json.dumps(report))
        self.assertNotIn('8888', json.dumps(report))

    def test_floor_guard_cannot_be_relaxed(self):
        with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3-1)):
            with self.assertRaisesRegex(RuntimeError, '10 GiB'):
                runner.free_space()

    def test_only_exact_fresh_output_path_and_2gib_rss(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            expected = root / 'mustard-board-track-filter-001.json'
            with patch.object(runner, 'DATA', root), patch.object(runner, 'OUTPUT', expected):
                self.assertEqual(runner.validate_output(expected), expected)
                with self.assertRaisesRegex(ValueError, 'exact fresh'):
                    runner.validate_output(root / 'mustard-board-track-filter-002.json')
                expected.write_bytes(b'occupied')
                with self.assertRaisesRegex(ValueError, 'exact fresh'):
                    runner.validate_output(expected)
        self.assertEqual(runner.check_worker_rss(runner.RSS_CAP_KIB), runner.RSS_CAP_KIB)
        with self.assertRaisesRegex(RuntimeError, '2 GiB'):
            runner.check_worker_rss(runner.RSS_CAP_KIB + 1)
        with self.assertRaisesRegex(RuntimeError, '2 GiB'):
            runner.check_worker_rss(None)


if __name__ == '__main__':
    unittest.main()
