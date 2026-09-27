"""Hermetic cyclic-graph SQLite copy and fixed mapper-policy contracts."""

from __future__ import annotations

import importlib.util
from contextlib import closing
import shutil
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.classical_backend import local_graph_sfm as local
from scripts.classical_backend.mustard_stage import TRAIN_NAMES


SCRATCH = Path(__file__).resolve().parents[2] / ".local-tools" / "tmp"


def fixture_database(path: Path) -> tuple[list[str], set[int]]:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript("""
            CREATE TABLE cameras(camera_id INTEGER, model INTEGER, params BLOB);
            CREATE TABLE images(image_id INTEGER, name TEXT, camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, data BLOB);
            CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, data BLOB);
            CREATE TABLE matches(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
        """)
        connection.execute("INSERT INTO cameras VALUES(1,2,?)", (b"camera",))
        for i, name in enumerate(TRAIN_NAMES, 1):
            connection.execute("INSERT INTO images VALUES(?,?,1)", (i, name))
            connection.execute("INSERT INTO keypoints VALUES(?,?,?)", (i, 2, bytes([i])))
            connection.execute("INSERT INTO descriptors VALUES(?,?,?)", (i, 2, bytes([255-i])))
        ordered = local.sorted_images(connection, list(TRAIN_NAMES))
        allowed = local.allowed_pairs(ordered)
        removed_positive = set(sorted(allowed)[::10][:19])
        removed_raw = set(sorted(allowed)[1::10][:18])
        for i in range(1, 49):
            for j in range(i+1, 49):
                key = local.pair_id(i, j)
                match_rows = 0 if key in removed_raw else (40 if key in allowed else 3)
                verified_rows = 0 if key in removed_positive else (30 if key in allowed else 2)
                connection.execute("INSERT INTO matches VALUES(?,?,2,?)",
                                   (key, match_rows, f"match-{i}-{j}".encode()))
                connection.execute("INSERT INTO two_view_geometries VALUES(?,?,2,?)",
                                   (key, verified_rows, f"verified-{i}-{j}".encode()))
        connection.commit()
    return list(TRAIN_NAMES), removed_positive


class LocalGraphTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cyclic_selected_order_wrap_and_initial_pair(self):
        images = [(i, name) for i, name in enumerate(TRAIN_NAMES, 1)]
        allowed = local.allowed_pairs(images)
        self.assertEqual(len(allowed), 192)
        self.assertIn(local.pair_id(1, 48), allowed)
        self.assertIn(local.pair_id(3, 6), allowed)
        self.assertNotIn(local.pair_id(1, 20), allowed)
        self.assertEqual(images[2][1], "NP3_012.jpg")
        self.assertEqual(images[5][1], "NP3_036.jpg")

    def test_immutable_backup_prunes_both_pair_tables_only(self):
        source = self.root / "source.db"
        names, _ = fixture_database(source)
        before = local.digest(source)
        with patch.object(local, "SOURCE_DB_SHA", before):
            result = local.copy_filter_database(source, self.root / "filtered.db", names)
        self.assertEqual(local.digest(source), before)
        self.assertEqual(result["removed_match_rows"], 936)
        self.assertEqual(result["removed_verified_rows"], 936)
        self.assertEqual(result["after"]["positive_verified_components"], [48])
        self.assertEqual(result["after"]["positive_allowed_verified_rows"], 173)
        with closing(local.immutable_connection(source)) as original, closing(
                sqlite3.connect(self.root / "filtered.db")) as copied:
            for table in local.PROTECTED_TABLES:
                self.assertEqual(local.table_rows(original, table), local.table_rows(copied, table))
            self.assertEqual(len(local.table_rows(copied, "matches")), 192)
            self.assertEqual(len(local.table_rows(copied, "two_view_geometries")), 192)
            self.assertEqual({row[0] for row in local.table_rows(copied, "matches")},
                             {row[0] for row in local.table_rows(copied, "two_view_geometries")})

    def test_immutable_source_seal_and_missing_edge_fail_before_copy(self):
        source = self.root / "source.db"
        names, _ = fixture_database(source)
        with self.assertRaisesRegex(ValueError, "SHA differs"):
            local.copy_filter_database(source, self.root / "not-created.db", names)
        self.assertFalse((self.root / "not-created.db").exists())
        with closing(sqlite3.connect(source)) as connection:
            connection.execute("DELETE FROM two_view_geometries WHERE pair_id=?",
                               (local.pair_id(3, 6),))
            connection.commit()
        with patch.object(local, "SOURCE_DB_SHA", local.digest(source)):
            with self.assertRaisesRegex(ValueError, "lacks a required"):
                local.copy_filter_database(source, self.root / "also-not-created.db", names)
        self.assertFalse((self.root / "also-not-created.db").exists())

    def test_no_model_and_registration_only_selection_are_explicit(self):
        empty = local.reconstruction_summary([])
        self.assertEqual(empty["reconstruction_status"], "no_model")
        self.assertFalse(empty["registration_gate"]["met"])
        sparse = local.reconstruction_summary([{"index": 1, "registered": 33, "points3D": 100},
                                               {"index": 0, "registered": 12, "points3D": 500}])
        self.assertEqual(sparse["reconstruction_status"], "insufficient_registration")
        enough = local.reconstruction_summary([{"index": 2, "registered": 40, "points3D": 200},
                                               {"index": 1, "registered": 40, "points3D": 200}])
        self.assertEqual(enough["selected_model"]["index"], 1)
        self.assertTrue(enough["registration_gate"]["met"])
        self.assertIn("not_camera_or_shape_acceptance", enough["reconstruction_status"])

    def test_logical_fingerprint_allows_header_only_write_not_row_mutation(self):
        source = self.root / "source.db"
        names, _ = fixture_database(source)
        destination = self.root / "filtered.db"
        with patch.object(local, "SOURCE_DB_SHA", local.digest(source)):
            record = local.copy_filter_database(source, destination, names)
        before_file_sha = record["filtered_database_sha256"]
        before_logical_sha = record["filtered_database_logical_sha256"]
        # Reproduce SQLite's harmless paired header change-counter update.
        data = bytearray(destination.read_bytes())
        for offset in (24, 92):
            value = struct.unpack_from(">I", data, offset)[0]
            struct.pack_into(">I", data, offset, value + 1)
        destination.write_bytes(data)
        self.assertNotEqual(local.digest(destination), before_file_sha)
        self.assertEqual(local.logical_database_sha(destination), before_logical_sha)
        self.assertTrue(local.database_postflight(
            destination, before_file_sha, before_logical_sha)["logical_unchanged"])
        with closing(sqlite3.connect(destination)) as connection:
            connection.execute("UPDATE cameras SET model=3 WHERE camera_id=1")
            connection.commit()
        self.assertNotEqual(local.logical_database_sha(destination), before_logical_sha)

    def test_postflight_rejects_match_and_schema_mutations(self):
        source = self.root / "source.db"
        names, _ = fixture_database(source)
        clean = self.root / "filtered.db"
        with patch.object(local, "SOURCE_DB_SHA", local.digest(source)):
            record = local.copy_filter_database(source, clean, names)
        for label, sql in (
                ("match", "UPDATE matches SET data=x'01' WHERE pair_id=(SELECT MIN(pair_id) FROM matches)"),
                ("schema", "CREATE INDEX modified_matches ON matches(rows)")):
            with self.subTest(label=label):
                changed = self.root / f"{label}.db"
                shutil.copyfile(clean, changed)
                with closing(sqlite3.connect(changed)) as connection:
                    connection.execute(sql)
                    connection.commit()
                audit = local.database_postflight(
                    changed, record["filtered_database_sha256"],
                    record["filtered_database_logical_sha256"])
                self.assertFalse(audit["logical_unchanged"])

    def test_postflight_rejects_wal_shm_and_journal_sidecars(self):
        source = self.root / "source.db"
        names, _ = fixture_database(source)
        clean = self.root / "filtered.db"
        with patch.object(local, "SOURCE_DB_SHA", local.digest(source)):
            record = local.copy_filter_database(source, clean, names)
        for suffix in ("-wal", "-shm", "-journal"):
            with self.subTest(suffix=suffix):
                sidecar = Path(str(clean) + suffix)
                sidecar.write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "postflight sidecar"):
                    local.database_postflight(
                        clean, record["filtered_database_sha256"],
                        record["filtered_database_logical_sha256"])
                sidecar.unlink()

    @unittest.skipUnless(importlib.util.find_spec("pycolmap"), "optional native PyCOLMAP")
    def test_fixed_intrinsics_automatic_mapper_options(self):
        import pycolmap
        options = local.mapper_options()
        self.assertEqual(options.init_image_id1, -1)
        self.assertEqual(options.init_image_id2, -1)
        self.assertTrue(options.multiple_models)
        self.assertEqual((options.max_num_models, options.min_model_size), (5, 10))
        self.assertEqual((options.num_threads, options.mapper.num_threads), (2, 2))
        self.assertFalse(options.ba_refine_focal_length)
        self.assertFalse(options.ba_refine_principal_point)
        self.assertFalse(options.ba_refine_extra_params)
        self.assertFalse(options.mapper.abs_pose_refine_focal_length)
        self.assertFalse(options.mapper.abs_pose_refine_extra_params)


if __name__ == "__main__":
    unittest.main()
