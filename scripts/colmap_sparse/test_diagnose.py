"""Tests for the mapper replay boundary and SQLite immutability check."""

import json
from pathlib import Path
import tempfile
import unittest

import pycolmap

from scripts.colmap_sparse import diagnose


class DiagnoseTests(unittest.TestCase):
    def test_mapper_options_match_saved_baseline_except_snapshots(self):
        baseline = json.loads((diagnose.SOURCE_RUN / "provenance.json").read_text())
        options = diagnose.mapper_options(pycolmap, baseline, Path("/unused/snapshots"))
        self.assertEqual(options.snapshot_images_freq, 1)
        self.assertEqual(options.snapshot_path, "/unused/snapshots")
        self.assertEqual(options.num_threads, 2)
        self.assertEqual(options.mapper.num_threads, 2)

    def test_sqlite_header_counter_change_allowed_only_when_mirrored(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.db"
            copy = Path(tmp) / "copy.db"
            data = bytearray(128)
            data[:16] = b"SQLite format 3\x00"
            data[27] = data[95] = 4
            source.write_bytes(data)
            data[27] = data[95] = 5
            copy.write_bytes(data)
            self.assertEqual(diagnose.sqlite_header_changes(source, copy), [27, 95])
            self.assertEqual(diagnose.sqlite_content_sha256(source),
                             diagnose.sqlite_content_sha256(copy))
            data[95] = 6
            copy.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "mirror differ"):
                diagnose.sqlite_header_changes(source, copy)

    def test_sqlite_content_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.db"
            copy = Path(tmp) / "copy.db"
            data = bytearray(128)
            data[:16] = b"SQLite format 3\x00"
            source.write_bytes(data)
            data[120] = 1
            copy.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "outside SQLite header"):
                diagnose.sqlite_header_changes(source, copy)
            self.assertNotEqual(diagnose.sqlite_content_sha256(source),
                                diagnose.sqlite_content_sha256(copy))


if __name__ == "__main__":
    unittest.main()
