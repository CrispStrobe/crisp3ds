"""Foreground ROI filtering keeps heldout IDs frozen and clears pair caches."""

import copy
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.colmap_sparse import foreground as fg
from scripts.foreground_roi import polygon as roi


def square(limit):
    return [[0, 0], [limit, 0], [limit, limit], [0, limit]]


class ForegroundTests(unittest.TestCase):
    def test_polygon_boundary_and_subset_are_frozen(self):
        polygon = square(6)
        self.assertTrue(roi.in_roi((polygon,), (3, 3)))
        self.assertTrue(roi.in_roi((polygon,), (6, 3)))
        self.assertFalse(roi.in_roi((polygon,), (7, 3)))
        manifest = {"images": [
            {"name": "a", "features": [[1, 1], [9, 9]], "heldout_ids": [0, 1]},
            {"name": "b", "features": [[2, 2], [8, 8]], "heldout_ids": [0, 1]}],
            "heldout_pairs": [
                {"image1": "a", "feature1": 0, "image2": "b", "feature2": 0},
                {"image1": "a", "feature1": 1, "image2": "b", "feature2": 0}]}
        before = copy.deepcopy(manifest)
        self.assertEqual(fg.object_subset(manifest, {"a": (polygon,), "b": (polygon,)}), [0])
        self.assertEqual(manifest, before)

    def test_roi_schema_requires_all_images_and_in_bounds_vertices(self):
        data = {"schema": "foreground_roi_v1", "provenance": {"method": "manual",
                "description": roi.PROVENANCE}, "images": [{"name": "a", "width": 768,
                "height": 512, "original_sha256": "a"*64, "polygons": [square(6)]}]}
        expected = {"a": {"width": 768, "height": 512, "sha256": "a"*64}}
        self.assertEqual(roi.validate_contract(data, expected)["a"], (tuple(map(tuple, square(6))),))
        with self.assertRaisesRegex(ValueError, "every frozen image"):
            roi.validate_contract(data, {**expected, "b": expected["a"]})
        data["images"][0]["polygons"][0][2][0] = 800
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            roi.validate_contract(data, expected)

    def test_compaction_clears_cache_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source_path, copy_path = Path(temp) / "source.db", Path(temp) / "copy.db"
            source = sqlite3.connect(source_path)
            source.executescript("""CREATE TABLE cameras(camera_id INTEGER,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER);
                CREATE TABLE pose_priors(image_id INTEGER);
                CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
                CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);
                CREATE TABLE matches(pair_id INTEGER);
                CREATE TABLE two_view_geometries(pair_id INTEGER);""")
            source.execute("INSERT INTO cameras VALUES(1,0,768,512,?,0)",
                           (struct.pack("<ddd", 921.6, 384, 256),))
            source.execute("INSERT INTO matches VALUES(12)")
            source.execute("INSERT INTO two_view_geometries VALUES(12)")
            keypoints = np.array([[1, 1, 1, 0, 0, 1], [5, 5, 1, 0, 0, 1],
                                  [9, 9, 1, 0, 0, 1]], np.float32)
            descriptors = np.arange(3*128, dtype=np.uint8).reshape(3, 128)
            source.execute("INSERT INTO keypoints VALUES(1,3,6,?)", (keypoints.tobytes(),))
            source.execute("INSERT INTO descriptors VALUES(1,3,128,?)", (descriptors.tobytes(),))
            source.commit()
            dest = sqlite3.connect(copy_path)
            source.backup(dest)
            item = {"name": "a", "image_id": 1, "heldout_ids": [3],
                    "features": keypoints.tolist()+[[2, 2, 1, 0, 0, 1]],
                    "compact_to_original": [0, 1, 2]}
            manifest = {"images": [item]}
            before = json.dumps(manifest, sort_keys=True)
            records, prep = fg.filter_database(dest, manifest, {"a": (square(6),)})
            self.assertEqual(records[0]["compact_to_original"], [0, 1])
            self.assertEqual(prep["training_rows_after"], 2)
            self.assertEqual(dest.execute("SELECT rows FROM keypoints").fetchone()[0], 2)
            self.assertEqual(dest.execute("SELECT rows FROM descriptors").fetchone()[0], 2)
            self.assertEqual(dest.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0)
            self.assertEqual(dest.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0], 0)
            self.assertEqual(source.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1)
            self.assertEqual(source.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0], 1)
            self.assertEqual(source.execute("SELECT rows FROM keypoints").fetchone()[0], 3)
            self.assertEqual(json.dumps(manifest, sort_keys=True), before)
            dest.close(); source.close()

    def test_score_counts_finite_and_missing_in_fixed_population(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "cameras.txt").write_text("placeholder")
            manifest = {"images": [{"name": name, "features": [[1, 1]]}
                                   for name in ("a", "b", "c")],
                        "heldout_pairs": [
                            {"image1": "a", "feature1": 0, "image2": "b", "feature2": 0},
                            {"image1": "a", "feature1": 0, "image2": "b", "feature2": 0},
                            {"image1": "a", "feature1": 0, "image2": "c", "feature2": 0}]}
            images = {1: {"name": "a", "camera_id": 1},
                      2: {"name": "b", "camera_id": 1}}
            with mock.patch.object(fg.checker, "cameras", return_value={1: {}}), \
                 mock.patch.object(fg.checker, "images", return_value=images), \
                 mock.patch.object(fg.checker, "pair_sampson", side_effect=[0.5, None]):
                score = fg.score_pairs(manifest, [0, 1, 2], model)
            self.assertEqual(score["total_pairs"], 3)
            self.assertEqual(score["scored_pairs"], 2)
            self.assertEqual(score["finite_pairs"], 1)
            self.assertEqual(score["unscored_pairs"], 1)
            self.assertEqual(score["greater_than_4px_or_missing"], 2)


if __name__ == "__main__":
    unittest.main()
