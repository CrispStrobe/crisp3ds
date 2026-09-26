import sqlite3
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.colmap_sparse.feature_split import split_features
from scripts.colmap_sparse.heldout_v2 import (
    SEED, mutual_heldout_matches, prepare_database, source_camera,
)
from scripts.sparse_verify.verify import spatial_split_overlap


class HeldoutV2Test(unittest.TestCase):
    def fixture(self, root):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,
                                 height INTEGER,params BLOB,prior_focal_length INTEGER);
            CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER);
            CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE matches(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
            CREATE TABLE pose_priors(image_id INTEGER);
        """)
        conn.execute("INSERT INTO cameras VALUES(1,0,768,512,?,0)",
                     (struct.pack("<ddd", 921.6, 384, 256),))
        conn.execute("INSERT INTO matches VALUES(1,1,2,?)", (struct.pack("<II", 0, 1),))
        conn.execute("INSERT INTO two_view_geometries VALUES(1,1,2,?)", (b"bad",))
        rows = np.array([[0, 0, 1, 0, 0, 1], [.02, 0, 0, -1, 1, 0],
                         [2, 0, 1, 0, 0, 1], [4, 0, 1, 0, 0, 1],
                         [6, 0, 1, 0, 0, 1], [8, 0, 1, 0, 0, 1]], np.float32)
        desc = np.arange(6*128, dtype=np.uint8).reshape(6, 128)
        audit_images = []
        for i in range(10):
            name = f"{i:02}.png"
            (root/name).write_bytes(bytes([i]))
            conn.execute("INSERT INTO images VALUES(?,?,1)", (i+1, name))
            conn.execute("INSERT INTO keypoints VALUES(?,?,?,?)",
                         (i+1, 6, 6, rows.tobytes()))
            conn.execute("INSERT INTO descriptors VALUES(?,?,?,?)",
                         (i+1, 6, 128, desc.tobytes()))
            audit_images.append(dict(name=name,
                                     heldout_ids=list(split_features(rows, name, SEED).heldout_ids)))
        conn.commit()
        return conn, dict(images=audit_images), rows, desc

    def test_delete_old_matches_and_reindex_training_rows(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            conn, audit, rows, desc = self.fixture(Path(temp))
            camera, records, full_descriptors, preparation = prepare_database(conn, audit, Path(temp))
            self.assertEqual(camera["model"], "SIMPLE_PINHOLE")
            self.assertEqual(preparation["copied_match_rows_deleted"], 1)
            self.assertEqual(preparation["copied_two_view_geometries_deleted"], 1)
            self.assertEqual(preparation["remaining_match_rows_before_matching"], 0)
            self.assertEqual(preparation["remaining_two_view_geometries_before_matching"], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0], 0)
            self.assertEqual(len(records), 10)
            for record in records:
                self.assertEqual(len(record["heldout_ids"])+len(record["compact_to_original"]), 6)
                self.assertEqual(spatial_split_overlap(rows, set(record["heldout_ids"])), 0)
                iid = record["image_id"]
                db_kp = np.frombuffer(conn.execute("SELECT data FROM keypoints WHERE image_id=?",
                                                   (iid,)).fetchone()[0], dtype=np.float32).reshape(-1, 6)
                db_desc = np.frombuffer(conn.execute("SELECT data FROM descriptors WHERE image_id=?",
                                                     (iid,)).fetchone()[0], dtype=np.uint8).reshape(-1, 128)
                self.assertTrue(np.array_equal(db_kp, rows[list(record["compact_to_original"])]))
                self.assertTrue(np.array_equal(db_desc, desc[list(record["compact_to_original"])]))
                self.assertTrue(np.array_equal(full_descriptors[record["name"]], desc))
            conn.close()

    def test_source_camera_rejects_prior(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            conn, _, _, _ = self.fixture(Path(temp))
            conn.execute("INSERT INTO pose_priors VALUES(1)")
            with self.assertRaises(ValueError):
                source_camera(conn)
            conn.close()

    def test_heldout_match_requires_ratio_both_directions(self):
        # A's first descriptor uniquely chooses B's first, but B's first has
        # a nearly equal second choice in A. Reverse-ratio failure rejects it.
        a = np.array([[0], [11], [100]], dtype=np.uint8)
        b = np.array([[5], [100], [200]], dtype=np.uint8)
        pairs = mutual_heldout_matches(a, (0, 1, 2), b, (0, 1, 2))
        self.assertNotIn((0, 0), pairs)


if __name__ == "__main__":
    unittest.main()
