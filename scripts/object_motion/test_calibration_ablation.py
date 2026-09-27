"""Cached-feature and oracle-camera ablation contracts without native SfM."""

import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.object_motion import calibration_ablation as ablation


class CalibrationAblationTests(unittest.TestCase):
    def test_serial_verifier_mock_contract_and_closes_on_error(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy absent")
        raw = np.array([[2, 1], [0, 3]], dtype=np.uint32)
        points = {1: np.array([[1, 2, 7], [3, 4, 7], [5, 6, 7]], dtype=np.float32),
                  2: np.array([[2, 3, 7], [4, 5, 7], [6, 7, 7], [8, 9, 7]], dtype=np.float32)}
        events = []

        class Database:
            def open(self, path):
                events.append(("open", path))

            def read_image(self, image_id):
                return types.SimpleNamespace(camera_id=image_id)

            def read_camera(self, camera_id):
                return camera_id

            def read_keypoints(self, image_id):
                return points[image_id]

            def read_matches(self, left, right):
                return raw

            def write_two_view_geometry(self, left, right, geometry):
                events.append(("write", left, right, geometry))

            def close(self):
                events.append(("close",))

        def estimator(camera1, xy1, camera2, xy2, matches, options):
            self.assertEqual((camera1, camera2), (1, 2))
            self.assertEqual((xy1.dtype.name, xy2.dtype.name), ("float64", "float64"))
            self.assertEqual(xy1.tolist(), [[1, 2], [3, 4], [5, 6]])
            self.assertEqual(xy2.tolist(), [[2, 3], [4, 5], [6, 7], [8, 9]])
            self.assertIs(matches, raw)
            self.assertEqual(matches.tolist(), [[2, 1], [0, 3]])
            return "geometry"

        fake = types.SimpleNamespace(Database=Database, estimate_two_view_geometry=estimator)
        with patch.dict(sys.modules, {"pycolmap": fake}):
            ablation.serial_reverify("db", [("a", "b")], {"a": 1, "b": 2}, object())
        self.assertEqual(events, [("open", "db"), ("write", 1, 2, "geometry"), ("close",)])
        events.clear()
        fake.estimate_two_view_geometry = lambda *args: (_ for _ in ()).throw(RuntimeError("native failure"))
        with patch.dict(sys.modules, {"pycolmap": fake}):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                ablation.serial_reverify("db", [("a", "b")], {"a": 1, "b": 2}, object())
        self.assertEqual(events, [("open", "db"), ("close",)])
        self.assertEqual(raw.tolist(), [[2, 1], [0, 3]])

    def test_table_digests_detect_mutation_and_are_scalar_sha(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
            connection.execute("CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
            connection.execute("CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
            connection.execute("INSERT INTO keypoints VALUES(1,1,2,?)", (b"abcd",))
            connection.execute("INSERT INTO descriptors VALUES(1,1,2,?)", (b"xy",))
            connection.execute("INSERT INTO matches VALUES(7,1,2,?)", (b"match",))
            first = ablation.feature_digest(connection)
            self.assertEqual(len(ablation.combined_feature_sha256(first)), 64)
            match = ablation.raw_match_digest(connection)
            self.assertEqual(len(match["sha256"]), 64)
            connection.execute("UPDATE keypoints SET data=?", (b"abce",))
            self.assertNotEqual(ablation.combined_feature_sha256(first),
                                ablation.combined_feature_sha256(ablation.feature_digest(connection)))

    def test_pair_decode_and_selected_seed(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            connection.execute("CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER)")
            for index, angle in enumerate(range(0, 360, 6), 1):
                connection.execute("INSERT INTO images VALUES(?,?)", (index, f"NP3_{angle:03}.jpg"))
            for left in range(1, 5):
                for right in range(left + 1, 61):
                    connection.execute("INSERT INTO matches VALUES(?,1)",
                                       (left * ablation.MAX_IMAGE_ID + right,))
            pairs, ids = ablation.pair_names(connection)
            self.assertEqual(len(pairs), 230)
            self.assertEqual(ids["NP3_192.jpg"], 33)
            self.assertEqual(pairs[0], ("NP3_000.jpg", "NP3_006.jpg"))

    def test_supplied_full_opencv_preserves_k3_when_fixture_available(self):
        if not ablation.CALIBRATION.exists() or shutil.which("h5dump") is None:
            self.skipTest("Berkeley calibration metadata or h5dump absent")
        shifted = ablation.full_opencv_params(ablation.CALIBRATION, 0.5)
        unshifted = ablation.full_opencv_params(ablation.CALIBRATION, 0.0)
        self.assertEqual(len(shifted), 12)
        self.assertAlmostEqual(shifted[2] - unshifted[2], 0.5)
        self.assertAlmostEqual(shifted[3] - unshifted[3], 0.5)
        self.assertLess(shifted[8], -0.17)  # k3 is not silently dropped.
        self.assertEqual(shifted[9:], [0, 0, 0])

    def test_serial_pair_preserves_raw_matches_when_local_fixture_available(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("pinned local PyCOLMAP not installed")
        source = ablation.SOURCE / "database.db"
        if not source.is_file():
            self.skipTest("sealed foreground feature database absent")
        temp_root = ablation.ROOT / ".local-tools/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            copied = ablation.Path(folder) / "copied.db"
            shutil.copyfile(source, copied)
            with sqlite3.connect(copied) as connection:
                raw_before = ablation.raw_match_digest(connection)
                ids = {name: image_id for image_id, name in
                       connection.execute("SELECT image_id,name FROM images")}
                connection.execute("DELETE FROM two_view_geometries")
            pair = ("NP3_000.jpg", "NP3_006.jpg")
            original_estimator = pycolmap.estimate_two_view_geometry
            captured = []
            def checked_estimator(camera1, xy1, camera2, xy2, matches, options):
                captured.append((xy1.dtype.name, xy2.dtype.name, matches.dtype.name,
                                 xy1.shape[1], xy2.shape[1], matches.shape[1]))
                return original_estimator(camera1, xy1, camera2, xy2, matches, options)
            with patch.object(pycolmap, "estimate_two_view_geometry", side_effect=checked_estimator):
                ablation.serial_reverify(copied, [pair], ids, pycolmap.TwoViewGeometryOptions())
            self.assertEqual(captured, [("float64", "float64", "uint32", 2, 2, 2)])
            with sqlite3.connect(copied) as connection:
                self.assertEqual(ablation.raw_match_digest(connection), raw_before)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0], 1)
            db = pycolmap.Database()
            db.open(str(copied))
            try:
                raw = db.read_matches(ids[pair[0]], ids[pair[1]])
                geometry = db.read_two_view_geometry(ids[pair[0]], ids[pair[1]])
                self.assertEqual(raw.shape[1], 2)
                self.assertGreater(len(geometry.inlier_matches), 0)
            finally:
                db.close()



if __name__ == "__main__":
    unittest.main()
