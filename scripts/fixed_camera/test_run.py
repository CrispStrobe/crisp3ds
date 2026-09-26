"""Synthetic geometry and compact-index checks for fixed-camera tracks."""

import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from scripts.fixed_camera.run import build_tracks, read_verified_edges, reconstruct
from scripts.sparse_verify import verify as checker


def pose(name, center, angle=0):
    c, s = math.cos(angle), math.sin(angle)
    r = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    t = checker.mv(r, [-v for v in center])
    return {"name": name, "camera_id": 1, "R": r, "t": t, "center": center}


def projected(im, xyz, camera):
    local = [v + t for v, t in zip(checker.mv(im["R"], xyz), im["t"])]
    return [camera["params"][0] * local[0] / local[2] + camera["params"][1],
            camera["params"][0] * local[1] / local[2] + camera["params"][2]]


class FixedCameraTests(unittest.TestCase):
    def setUp(self):
        self.cam = {1: {"model": "SIMPLE_PINHOLE", "width": 800, "height": 600,
                        "params": [700, 400, 300]}}
        self.ims = {"a": pose("a", [0, 0, 0]),
                    "b": pose("b", [1, 0, 0], 0.17),
                    "c": pose("c", [0, 1, 0], -0.11)}

    def test_exact_nonidentity_pose(self):
        xyz = [0.2, -0.1, 5]
        features = {name: [projected(im, xyz, self.cam[1])]
                    for name, im in self.ims.items()}
        point, reason = reconstruct((("a", 0), ("b", 0), ("c", 0)), self.ims, self.cam, features)
        self.assertIsNone(reason)
        self.assertLess(max(abs(a-b) for a, b in zip(point["xyz"], xyz)), 1e-9)
        self.assertLess(point["max_reprojection_px"], 1e-8)

    def test_noisy_outlier_rejected(self):
        xyz = [0.2, -0.1, 5]
        features = {name: [projected(im, xyz, self.cam[1])]
                    for name, im in self.ims.items()}
        features["c"][0][0] += 35
        _, reason = reconstruct((("a", 0), ("b", 0), ("c", 0)), self.ims, self.cam, features)
        self.assertEqual(reason, "high_reprojection")

    def test_negative_depth_and_parallel(self):
        behind = [0.2, 0.1, -5]
        features = {name: [projected(im, behind, self.cam[1])]
                    for name, im in self.ims.items()}
        _, reason = reconstruct((("a", 0), ("b", 0)), self.ims, self.cam, features)
        self.assertEqual(reason, "nonpositive_depth")
        same = {"a": self.ims["a"], "b": pose("b", [0, 0, 0], 0.17)}
        front = [0, 0, 5]
        features = {name: [projected(im, front, self.cam[1])] for name, im in same.items()}
        _, reason = reconstruct((("a", 0), ("b", 0)), same, self.cam, features)
        self.assertEqual(reason, "low_parallax")

    def test_conflict_rejects_whole_component(self):
        edges = [(('a', 1), ('b', 1)), (('b', 1), ('c', 1)), (('c', 1), ('a', 2))]
        tracks, rejected, count = build_tracks(edges)
        self.assertEqual((tracks, rejected, count), ([], {"same_image_conflict": 1}, 1))

    def test_geometry_compact_to_original_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "geometry.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
                conn.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?)",
                             (1*2147483647+2, 1, 2, struct.pack('<II', 1, 0)))
            manifest = {"images": [{"image_id": 1, "name": "a", "compact_to_original": [9, 4]},
                                   {"image_id": 2, "name": "b", "compact_to_original": [7, 3]}]}
            edges, pairs = read_verified_edges(db, manifest)
            self.assertEqual((edges, pairs), ([(('a', 4), ('b', 7))], 1))


if __name__ == "__main__":
    unittest.main()
