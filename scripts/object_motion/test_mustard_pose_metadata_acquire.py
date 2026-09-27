"""Synthetic-only tests for bounded mustard metadata acquisition."""

import io
import hashlib
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.object_motion import mustard_pose_metadata_acquire as acquire


def fake_report():
    views = [{'name': f'NP3_{angle:03}.jpg', 'detected': angle < 234,
              **({'camera_from_board_rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
                 if angle < 234 else {})}
             for angle in range(0, 288, 6)]
    return {'schema': 'mustard_checkerboard_assisted_pose_diagnostic_v1',
            'detected_count': 39, 'views': views}


def write_tar(path, rows):
    with tarfile.open(path, 'w:gz') as archive:
        for name, contents, kind in rows:
            info = tarfile.TarInfo(name)
            if kind == 'file':
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
            elif kind == 'symlink':
                info.type = tarfile.SYMTYPE
                info.linkname = '/etc/passwd'
                archive.addfile(info)


class MetadataAcquireTests(unittest.TestCase):
    def test_download_enforces_byte_and_digest_caps_without_network(self):
        class FakeResponse:
            status = 200
            headers = {'Content-Length': '5', 'ETag': acquire.ETAG}

            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def geturl(self):
                return acquire.URL

            def read(self, _):
                payload, self.payload = self.payload, b''
                return payload

        with tempfile.TemporaryDirectory() as scratch, \
                patch.object(acquire, 'ARCHIVE_BYTES', 5), \
                patch.object(acquire, 'ARCHIVE_SHA256', hashlib.sha256(b'abcde').hexdigest()):
            target = Path(scratch) / 'archive.part'
            with patch.object(acquire.OPENER, 'open', return_value=FakeResponse(b'abcde')):
                result = acquire.download(target, time.monotonic())
            self.assertEqual(result['bytes'], 5)
            self.assertEqual(target.read_bytes(), b'abcde')
            with patch.object(acquire.OPENER, 'open', return_value=FakeResponse(b'abcdef')):
                with self.assertRaisesRegex(ValueError, 'byte cap'):
                    acquire.download(Path(scratch) / 'too-large.part', time.monotonic())
            with patch.object(acquire.OPENER, 'open', return_value=FakeResponse(b'xxxxx')):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    acquire.download(Path(scratch) / 'wrong-hash.part', time.monotonic())

    def test_allowlist_is_39_detected_train_angles_plus_calibration(self):
        selected = acquire.allowlist(fake_report())
        self.assertEqual(len(selected), 40)
        self.assertIn('006_mustard_bottle/poses/NP5_0_pose.h5', selected)
        self.assertNotIn('006_mustard_bottle/poses/NP5_234_pose.h5', selected)
        changed = fake_report()
        changed['views'][1]['name'] = changed['views'][0]['name']
        with self.assertRaises(ValueError):
            acquire.allowlist(changed)

    def test_extract_only_exact_members_and_ignore_traversal_and_photo(self):
        selected = acquire.allowlist(fake_report())
        rows = [(name, name.encode(), 'file') for name in sorted(selected)]
        rows += [('006_mustard_bottle/../../escape', b'bad', 'file'),
                 ('006_mustard_bottle/rgb/NP3_000.jpg', b'photo', 'file')]
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / 'tiny.tgz'
            write_tar(archive, rows)
            metadata = Path(scratch) / 'metadata'
            result = acquire.extract_metadata(archive, metadata, selected)
            self.assertEqual(len(result['members']), 40)
            self.assertFalse((Path(scratch) / 'escape').exists())
            self.assertFalse((metadata / 'rgb').exists())
            self.assertEqual((metadata / 'calibration.h5').read_bytes(),
                             b'006_mustard_bottle/calibration.h5')

    def test_reject_duplicate_and_symlink_allowlisted_member(self):
        selected = acquire.allowlist(fake_report())
        rows = [(name, b'x', 'file') for name in sorted(selected)]
        rows.append(('006_mustard_bottle/calibration.h5', b'y', 'file'))
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / 'duplicate.tgz'
            write_tar(archive, rows)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                acquire.extract_metadata(archive, Path(scratch) / 'out', selected)
        rows = [(name, b'x', 'file') for name in sorted(selected)
                if name != '006_mustard_bottle/calibration.h5']
        rows.append(('006_mustard_bottle/calibration.h5', b'', 'symlink'))
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / 'symlink.tgz'
            write_tar(archive, rows)
            with self.assertRaisesRegex(ValueError, 'invalid'):
                acquire.extract_metadata(archive, Path(scratch) / 'out', selected)

    def test_reject_missing_or_oversized_metadata(self):
        selected = acquire.allowlist(fake_report())
        rows = [(name, b'x', 'file') for name in sorted(selected)][:-1]
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / 'missing.tgz'
            write_tar(archive, rows)
            with self.assertRaisesRegex(ValueError, 'missing'):
                acquire.extract_metadata(archive, Path(scratch) / 'out', selected)
        rows = [(name, b'0123456789', 'file') for name in sorted(selected)]
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / 'oversized.tgz'
            write_tar(archive, rows)
            with patch.object(acquire, 'METADATA_CAP', 100):
                with self.assertRaisesRegex(ValueError, 'oversized'):
                    acquire.extract_metadata(archive, Path(scratch) / 'out', selected)


if __name__ == '__main__':
    unittest.main()
