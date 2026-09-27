"""Analytic exact-PNG membership and COLMAP keypoint DB tests."""

import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from scripts.object_motion import mustard_sparse_support as audit


class SparseSupportTests(TestCase):
    def test_floor_pixel_membership_and_out_of_bounds(self):
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 2] = 255
        xy = np.asarray([[2.01, 1.99], [2.99, 1.01], [1.99, 1.99],
                         [-0.01, 1], [4, 2], [2, 3]], dtype=np.float32)
        self.assertEqual(audit.sample_membership(xy, mask),
                         {"denominator": 6, "inside_coarse_pose_mask": 2,
                          "outside_coarse_pose_mask": 1, "outside_image_grid": 3})
        with self.assertRaises(ValueError):
            audit.sample_membership([[float("nan"), 1]], mask)

    def test_readonly_db_keypoints_and_blob_guards(self):
        with TemporaryDirectory() as directory:
            dbpath = Path(directory) / "database.db"
            with closing(sqlite3.connect(dbpath)) as db:
                db.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT)")
                db.execute("CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB)")
                db.executemany("INSERT INTO images VALUES(?,?)", [(1, "a.jpg"), (2, "b.jpg")])
                xy = np.asarray([[2.2, 1.2, 1, 0], [0.5, 0.5, 1, 0]], dtype="<f4")
                db.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (1, 2, 4, xy.tobytes()))
                db.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (2, 0, 2, b""))
                db.commit()
            masks = {name: np.zeros((3, 4), np.uint8) for name in ("a.jpg", "b.jpg")}
            masks["a.jpg"][1, 2] = 255
            report = audit.database_keypoint_support(dbpath, ["a.jpg", "b.jpg"], masks)
            self.assertEqual(report["total"]["denominator"], 2)
            self.assertEqual(report["total"]["inside_coarse_pose_mask"], 1)
            self.assertEqual(report["total"]["outside_coarse_pose_mask"], 1)
            self.assertEqual(report["by_image"]["b.jpg"]["denominator"], 0)
            with closing(sqlite3.connect(dbpath)) as db:
                db.execute("UPDATE keypoints SET data=? WHERE image_id=1", (b"truncated",))
                db.commit()
            with self.assertRaisesRegex(ValueError, "blob"):
                audit.database_keypoint_support(dbpath, ["a.jpg", "b.jpg"], masks)
            with closing(sqlite3.connect(dbpath)) as db:
                db.execute("UPDATE keypoints SET data=? WHERE image_id=1", (xy.tobytes(),))
                db.commit()
            (Path(directory) / "database.db-wal").write_bytes(b"live")
            with self.assertRaisesRegex(ValueError, "sidecar-free"):
                audit.database_keypoint_support(dbpath, ["a.jpg", "b.jpg"], masks)

    def test_source_sidecar_zero_wal_only(self):
        with TemporaryDirectory() as directory:
            dbpath = Path(directory) / "database.db"
            dbpath.write_bytes(b"db")
            wal = Path(directory) / "database.db-wal"
            shm = Path(directory) / "database.db-shm"
            wal.write_bytes(b"")
            shm.write_bytes(b"header")
            first = audit.source_sidecars(dbpath)
            self.assertEqual(first["-wal"]["bytes"], 0)
            shm.write_bytes(b"changed")
            self.assertNotEqual(first, audit.source_sidecars(dbpath))
            wal.write_bytes(b"uncommitted")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                audit.source_sidecars(dbpath)
