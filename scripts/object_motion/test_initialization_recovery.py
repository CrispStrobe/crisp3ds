"""Frozen-seed and intrinsic plausibility contracts without native mapping."""

import json
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.object_motion import initialization_recovery as recovery


class InitializationRecoveryTests(unittest.TestCase):
    def test_cache_logical_reader_opens_sqlite_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "fixture.db"
            import sqlite3
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("CREATE TABLE cameras(id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)")
                connection.execute("INSERT INTO cameras VALUES(1,2,1280,1024,?,0)", (b"camera",))
                connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
                connection.execute("INSERT INTO images VALUES(1,'photo.jpg')")
                for name in recovery.TABLES[2:]:
                    connection.execute(f"CREATE TABLE {name}(id INTEGER PRIMARY KEY, value BLOB)")
                    connection.execute(f"INSERT INTO {name} VALUES(1,?)", (b"same",))
            before = recovery.bounded.digest(database)
            result = recovery.read_cache_logical(database)
            self.assertEqual(before, recovery.bounded.digest(database))
            self.assertEqual(result["tables"]["matches"]["rows"], 1)

    def test_cache_logical_reader_closes_connection_on_error(self):
        class FailingConnection:
            closed = False

            def execute(self, *_args):
                raise RuntimeError("injected query failure")

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "fixture.db"
            database.touch()
            connection = FailingConnection()
            with patch.object(recovery.sqlite3, "connect", return_value=connection) as connect:
                with self.assertRaisesRegex(RuntimeError, "injected query failure"):
                    recovery.read_cache_logical(database)
            self.assertTrue(connection.closed)
            self.assertIn("mode=ro&immutable=1", connect.call_args.args[0])

    def test_intrinsics_gate_accepts_predeclared_plausible_camera(self):
        self.assertTrue(recovery.plausible_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                                     [1080, 640, 512, -.03]))

    def test_intrinsics_gate_rejects_observed_bad_and_nonfinite(self):
        self.assertFalse(recovery.plausible_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                                      [2995.0027, 640, 512, -6.99465]))
        self.assertFalse(recovery.plausible_intrinsics("SIMPLE_RADIAL", 1280, 1024,
                                                      [float("nan"), 640, 512, 0]))
        self.assertFalse(recovery.plausible_intrinsics("OPENCV", 1280, 1024,
                                                      [1080, 640, 512, 0]))
        self.assertFalse(recovery.plausible_intrinsics("SIMPLE_RADIAL", 640, 480,
                                                      [1080, 320, 240, 0]))

    def test_frozen_seed_and_image_only_diagnostic_required(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "diagnostic.json"
            audit = Path(folder) / "audit.json"
            evidence = {"schema": "ycb_initialization_read_only_v1",
                        "source_databases_sha256": {"foreground": recovery.HISTORICAL_DATABASE_SHA256},
                        "top_candidates": [{"pair": list(recovery.SEED_NAMES)}],
                        "reference_metadata_used_for_ranking": False}
            report.write_text(json.dumps(evidence))
            audit.write_text(json.dumps({"schema": "ycb_cache_mutation_audit_v1",
                                         "current_source_database_sha256": recovery.REBOUND_DATABASE_SHA256,
                                         "historical_source_database_sha256": recovery.HISTORICAL_DATABASE_SHA256,
                                         "photo_manifest_sha256": "b" * 64}))
            with patch.object(recovery, "AUDIT", audit), \
                 patch.object(recovery.ablation, "source_identity", return_value={
                     "source_database_sha256": recovery.REBOUND_DATABASE_SHA256,
                     "source_manifest_sha256": "b" * 64}), \
                 patch.object(recovery.bounded, "digest", return_value=recovery.DIAGNOSTIC_SHA256):
                self.assertEqual(recovery.verify_frozen_inputs("source", "prepared", report)["source_database_sha256"], recovery.REBOUND_DATABASE_SHA256)
                evidence["top_candidates"][0]["pair"] = ["NP3_000.jpg", "NP3_006.jpg"]
                report.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "frozen seed"):
                    recovery.verify_frozen_inputs("source", "prepared", report)
                evidence["top_candidates"][0]["pair"] = list(recovery.SEED_NAMES)
                evidence["reference_metadata_used_for_ranking"] = True
                report.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(ValueError, "frozen seed"):
                    recovery.verify_frozen_inputs("source", "prepared", report)


if __name__ == "__main__":
    unittest.main()
