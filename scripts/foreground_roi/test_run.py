import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.foreground_roi.polygon import (
    PROVENANCE, contains_point, load_contract, validate_contract, validate_polygon,
)
from scripts.foreground_roi.run import cycle_support, heldout_geometry, pair_support, point_support


class PolygonContractTest(unittest.TestCase):
    def test_pixel_centres_edges_and_outside(self):
        poly = validate_polygon([[0, 0], [10, 0], [10, 10], [0, 10]], 10, 10)
        for point in ((.5, .5), (0, 5), (10, 10), (5, 0), (5, 5)):
            self.assertTrue(contains_point(poly, point))
        for point in ((-.5, 5), (10.5, 5), (5, 10.5), (math.nan, 5)):
            self.assertFalse(contains_point(poly, point))

    def test_self_intersection_nonfinite_and_bounds_rejected(self):
        invalid = [
            [[0, 0], [10, 10], [0, 10], [10, 0]],
            [[0, 0], [10, 0], [math.nan, 10]],
            [[0, 0], [10, 0], [11, 10]],
            [[0, 0], [10, 0], [0, -1]],
            [[0, 0], [10, 0], [0, 0]],
            [[0, 0], [10, 0], [5, 0], [5, 10], [0, 10]],
        ]
        for polygon in invalid:
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                validate_polygon(polygon, 10, 10)

    def test_version_provenance_and_png_hash(self):
        digest = "a"*64
        expected = {"a.png": dict(width=10, height=10, sha256=digest)}
        payload = dict(schema="foreground_roi_v1",
                       provenance=dict(method="manual", description=PROVENANCE),
                       images=[dict(name="a.png", width=10, height=10,
                                    original_sha256=digest,
                                    polygons=[[[0, 0], [10, 0], [0, 10]]])])
        self.assertIn("a.png", validate_contract(payload, expected))
        payload["images"][0]["original_sha256"] = "b"*64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_contract(payload, expected)
        payload["images"][0]["original_sha256"] = digest
        payload["provenance"]["method"] = "automatic"
        with self.assertRaisesRegex(ValueError, "manual provenance"):
            validate_contract(payload, expected)

    def test_json_nan_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            path = Path(temp)/"roi.json"
            path.write_text('{"schema":"foreground_roi_v1","images":[],"x":NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite JSON"):
                load_contract(path, {})


class SupportTest(unittest.TestCase):
    def test_pair_and_cycle_support_counts(self):
        masks = {"a": (True, False), "b": (True, False), "c": (False,)}
        pairs = [dict(image1="a", feature1=0, image2="b", feature2=0),
                 dict(image1="a", feature1=0, image2="b", feature2=1),
                 dict(image1="a", feature1=1, image2="b", feature2=1)]
        result = pair_support(pairs, masks)
        self.assertEqual((result["both_inside"], result["exactly_one_inside"],
                          result["neither_inside"]), (1, 1, 1))
        cycles = [dict(id=0, refs=[dict(image="a", feature_id=0),
                                       dict(image="b", feature_id=0),
                                       dict(image="c", feature_id=0)],
                       lexical=dict(status="scored", third_reprojection_px=5))]
        result = cycle_support(cycles, masks)
        self.assertEqual(result["all_three_inside"]["cycle_count"], 0)
        self.assertEqual(result["other"]["cycle_count"], 1)

    def test_point_track_support(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            root = Path(temp)
            (root/"images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 a.png\n1 1 1 2 2 2\n"
                "2 1 0 0 0 -1 0 0 1 b.png\n3 1 1 4 2 2\n")
            (root/"points3D.txt").write_text(
                "1 0 0 5 255 255 255 0 1 0 2 0\n"
                "2 1 0 5 255 255 255 0 1 1 2 1\n")
            manifest = [dict(name="a.png", compact_to_original=[0, 1]),
                        dict(name="b.png", compact_to_original=[0, 1])]
            masks = {"a.png": (True, True), "b.png": (True, False)}
            summary, points = point_support(root, manifest, masks)
            self.assertEqual(summary["point_count"], 2)
            self.assertEqual(summary["all_inside"], 1)
            self.assertEqual(summary["any_inside"], 2)
            self.assertEqual(summary["majority_inside"], 1)
            self.assertEqual(summary["exactly_half_inside"], 1)
            self.assertEqual(summary["inside_track_links"], 3)
            self.assertEqual(len(points), 2)

    def test_roi_pair_geometry_counts_missing_and_outlier(self):
        with tempfile.TemporaryDirectory(dir=".local-tools/tmp") as temp:
            root = Path(temp)
            (root/"cameras.txt").write_text("1 SIMPLE_PINHOLE 100 100 100 50 50\n")
            (root/"images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 a.png\n\n"
                "2 1 0 0 0 -1 0 0 1 b.png\n\n")
            images_manifest = [dict(name="a.png", features=[[50, 50], [50, 50]]),
                               dict(name="b.png", features=[[30, 50], [30, 65]]),
                               dict(name="c.png", features=[[50, 50]])]
            masks = {"a.png": (True, True), "b.png": (True, True), "c.png": (True,)}
            matches = [dict(image1="a.png", feature1=0, image2="b.png", feature2=0),
                       dict(image1="a.png", feature1=1, image2="b.png", feature2=1),
                       dict(image1="a.png", feature1=0, image2="c.png", feature2=0)]
            bucket = heldout_geometry(matches, masks, images_manifest, root)["both_inside"]
            self.assertEqual(bucket["raw_count"], 3)
            self.assertEqual(bucket["unregistered_count"], 1)
            self.assertEqual(bucket["scored_geometry"]["finite_count"], 2)
            self.assertEqual(bucket["scored_geometry"]["within_4px"], 1)
            self.assertEqual(bucket["missing_inclusive_over_4px_or_unavailable_count"], 2)


if __name__ == "__main__":
    unittest.main()
