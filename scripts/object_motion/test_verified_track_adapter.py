"""Synthetic SQLite graph tests; no sealed TRAIN database is opened."""

from contextlib import closing
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.object_motion.verified_track_adapter import (
    MAX_COMPONENT_NODES, MAX_IMAGE_ID, ImageSource, TrackAdapterError,
    _Components, extract_candidate_tracks,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifiedTrackAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.database = base / "fixture.db"
        self.images, self.masks = base / "rgb", base / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        self.sources = []
        for image_id in range(1, 5):
            name = f"train_{image_id:03d}.jpg"
            photo, mask = self.images / name, self.masks / (name + ".png")
            Image.new("RGB", (64, 48), (image_id, 20, 30)).save(photo)
            Image.new("L", (64, 48), 255).save(mask)
            self.sources.append(ImageSource(name, digest(photo), digest(mask), 64, 48))
        with closing(sqlite3.connect(self.database)) as db:
            db.executescript("""
                CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB, config INTEGER);
            """)
            for image_id, source in enumerate(self.sources, 1):
                db.execute("INSERT INTO images VALUES(?,?)", (image_id, source.name))
                xy = np.array([[10.5 + image_id, 12.5], [20.5 + image_id, 22.5]], dtype="<f4")
                db.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (image_id, 2, 2, xy.tobytes()))
            db.commit()

    def edges(self, rows):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("DELETE FROM two_view_geometries")
            for (a, b), pairs in sorted(rows.items()):
                blob = np.asarray(pairs, dtype="<u4").reshape((-1, 2)).tobytes()
                db.execute("INSERT INTO two_view_geometries VALUES(?,?,?,?,?)",
                           (a * MAX_IMAGE_ID + b, len(pairs), 2, blob, 2))
            db.commit()

    def extract(self, sources=None):
        return extract_candidate_tracks(self.database, digest(self.database), self.images,
                                        self.masks, sources or self.sources, required_images=4)

    def test_clean_tracks_are_deterministic_and_read_only(self):
        self.edges({(1, 2): [(1, 1), (0, 0)], (2, 3): [(1, 1), (0, 0)]})
        before = digest(self.database)
        result = self.extract()
        self.assertEqual(digest(self.database), before)
        self.assertEqual(len(result.tracks), 2)
        self.assertEqual(result.verified_edges, 4)
        self.assertEqual(result.mask_supported_edges, 4)
        self.assertEqual([len(track.observations) for track in result.tracks], [3, 3])
        self.assertEqual(result.tracks[0].observations[0].x, 11.0)
        self.assertFalse(Path(str(self.database) + "-wal").exists())

    def test_ambiguous_alias_component_is_rejected_whole(self):
        self.edges({(1, 2): [(0, 0), (1, 1)], (2, 3): [(0, 0), (1, 1)],
                    (1, 3): [(1, 0)]})
        result = self.extract()
        self.assertEqual(result.tracks, ())
        self.assertEqual(result.rejected_conflicting_components, 1)

    def test_off_mask_edges_are_excluded_and_hashes_fail_closed(self):
        self.edges({(1, 2): [(0, 0)], (2, 3): [(0, 0)]})
        name = self.sources[1].name
        mask = self.masks / (name + ".png")
        pixels = np.full((48, 64), 255, np.uint8)
        pixels[12, 12] = 0  # image 2, feature 0: DB (12.5,12.5) -> OpenCV (12,12)
        Image.fromarray(pixels).save(mask)
        sources = list(self.sources)
        sources[1] = replace(sources[1], mask_sha256=digest(mask))
        result = self.extract(sources)
        self.assertEqual(result.tracks, ())
        self.assertEqual(result.mask_supported_edges, 0)
        with self.assertRaisesRegex(TrackAdapterError, "hash"):
            self.extract()

    def test_index_and_duplicate_edge_fail_closed(self):
        self.edges({(1, 2): [(2, 0)]})
        with self.assertRaisesRegex(TrackAdapterError, "feature-index"):
            self.extract()
        self.edges({(1, 2): [(0, 0), (0, 0)]})
        with self.assertRaisesRegex(TrackAdapterError, "feature-index"):
            self.extract()

    def test_giant_component_is_poisoned_even_if_later_extended(self):
        graph = _Components()
        for index in range(MAX_COMPONENT_NODES):
            graph.union((index + 1, 0), (index + 2, 0))
        root = graph.find((1, 0))
        self.assertTrue(graph.oversize[root])
        graph.union((MAX_COMPONENT_NODES + 1, 0), (MAX_COMPONENT_NODES + 2, 0))
        self.assertTrue(graph.oversize[graph.find((1, 0))])


if __name__ == "__main__":
    unittest.main()
