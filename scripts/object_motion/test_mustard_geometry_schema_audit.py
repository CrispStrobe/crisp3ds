"""Tiny synthetic schema tests; sealed TRAIN database is never opened."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.object_motion.mustard_geometry_schema_audit import (
    MAX_IMAGE_ID, SchemaAuditError, audit_database, digest,
    summarize_scalar_rows,
)


class GeometrySchemaAuditTest(unittest.TestCase):
    def test_aggregates_scalar_violations_and_limits_examples(self):
        rows = [(MAX_IMAGE_ID + i + 2, 1, 2, None, 2) for i in range(7)]
        rows.insert(0, (MAX_IMAGE_ID + 9, 2, 2, 16, 2))
        rows.append((2 * MAX_IMAGE_ID + 3, 0, 2, 0, 0))
        report = summarize_scalar_rows(rows)
        self.assertEqual(report["geometry_rows"], 9)
        self.assertEqual(report["offending_count"], 7)
        self.assertEqual(len(report["offending_first_five"]), 5)
        self.assertEqual(report["blob_length_validity_counts"], {"exact": 2, "null": 7})
        self.assertEqual(report["config_counts"], {"0": 1, "2": 8})
        self.assertNotIn("data", report)

    def test_read_only_sqlite_hash_before_after_and_sidecar_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "fixture.db"
            with closing(sqlite3.connect(database)) as db:
                db.execute("CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB,config INTEGER)")
                db.execute("INSERT INTO two_view_geometries VALUES(?,?,?,?,?)",
                           (MAX_IMAGE_ID + 2, 1, 2, b"12345678", 2))
                db.execute("INSERT INTO two_view_geometries VALUES(?,?,?,?,?)",
                           (MAX_IMAGE_ID + 3, 0, 2, None, 0))
                db.commit()
            before = digest(database)
            with patch("scripts.object_motion.mustard_geometry_schema_audit.disk_floor",
                       return_value={"internal": 20 * 1024**3, "external": 20 * 1024**3}):
                report = audit_database(database, before)
                self.assertEqual(report["offending_count"], 1)
                self.assertEqual(report["offending_first_five"][0]["blob_length"], None)
                self.assertEqual(report["selected_columns"],
                                 ["pair_id", "rows", "cols", "length(data)", "config"])
                self.assertEqual(digest(database), before)
                self.assertFalse(Path(str(database) + "-wal").exists())
                with self.assertRaisesRegex(SchemaAuditError, "SHA"):
                    audit_database(database, "0" * 64)
                Path(str(database) + "-wal").touch()
                with self.assertRaisesRegex(SchemaAuditError, "sidecar"):
                    audit_database(database, before)


if __name__ == "__main__":
    unittest.main()
