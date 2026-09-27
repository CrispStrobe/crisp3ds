import math
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.object_dataset import sparse_sfm_metrics as metrics


class FakeImage:
    def __init__(self, name, center_x, observations):
        self.name = name
        self.center_x = center_x
        self.points2D = [SimpleNamespace(xy=xy, point3D_id=point_id)
                         for xy, point_id in observations]

    def project_point(self, xyz):
        x, y, z = xyz
        if z <= 0:
            return None
        return ((x - self.center_x) / z * 100 + 50, y / z * 100 + 50)


def point(xyz, *observations):
    return SimpleNamespace(xyz=xyz, track=SimpleNamespace(elements=[
        SimpleNamespace(image_id=image_id, point2D_idx=index)
        for image_id, index in observations]))


class SparseSFMMetricsTests(unittest.TestCase):
    def test_disconnected_verified_graph_includes_isolates(self):
        names = ["A", "B", "C", "D"]
        result = metrics.graph_components(names, [("A", "B"), ("B", "A")])
        self.assertEqual(result["component_sizes"], [2, 1, 1])
        self.assertEqual(result["verified_pair_edges"], 1)
        self.assertEqual(result["components"], [["A", "B"], ["C"], ["D"]])
        with self.assertRaisesRegex(ValueError, "outside selected"):
            metrics.graph_components(names, [("A", "X")])

    def test_sqlite_colmap_pair_decoding_and_zero_row_isolate(self):
        temp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            database = Path(directory) / "database.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
                connection.executemany("INSERT INTO images VALUES (?,?)", [(1, "A"), (2, "B"), (3, "C")])
                connection.executemany("INSERT INTO two_view_geometries VALUES (?,?)", [
                    (metrics.PAIR_BASE + 2, 10), (2 * metrics.PAIR_BASE + 3, 0)])
            graph = metrics.verified_graph(database, ["A", "B", "C"])
            self.assertEqual(graph["component_sizes"], [2, 1])
            self.assertEqual(graph["verified_pair_edges"], 1)

    def test_known_translated_cameras_tracks_and_reprojection_denominators(self):
        images = {
            1: FakeImage("A", 0, [((50, 50), 1), ((100, 50), 2)]),
            2: FakeImage("B", 1, [((3, 50), 1), ((50, 50), 2)]),
            3: FakeImage("C", 2, [((0, 50), 2)]),
        }
        model = SimpleNamespace(images=images, reg_image_ids=lambda: [1, 2, 3], points3D={
            1: point((0, 0, 2), (1, 0), (2, 0)),
            2: point((1, 0, 2), (1, 1), (2, 1), (3, 0)),
        })
        result = metrics.summarize_model(model, ["A", "B", "C", "D"])
        self.assertEqual(result["registered_count"], 3)
        self.assertEqual(result["selected_count"], 4)
        self.assertEqual(result["track_length"]["point_denominator"], 2)
        self.assertEqual(result["track_length"]["median"], 2.5)
        self.assertEqual(result["track_length"]["at_least_three_fraction"], 0.5)
        self.assertEqual(result["triangulated_observations_by_image"], {"A": 2, "B": 2, "C": 1})
        error = result["reprojection_l2_pixels"]
        self.assertEqual(error["track_observation_denominator"], 5)
        self.assertEqual(error["finite_count"], 5)
        self.assertEqual(error["invalid_count"], 0)
        self.assertAlmostEqual(error["mean"], 0.6)
        self.assertAlmostEqual(error["p95"], 2.4)

    def test_missing_projection_count_is_not_invented_zero(self):
        images = {1: FakeImage("A", 0, [((50, 50), 1)]),
                  2: FakeImage("B", 0, [((50, 50), 1)])}
        model = SimpleNamespace(images=images, reg_image_ids=lambda: [1, 2], points3D={
            1: point((0, 0, -2), (1, 0), (2, 0))})
        error = metrics.summarize_model(model, ["A", "B"])["reprojection_l2_pixels"]
        self.assertEqual(error["track_observation_denominator"], 2)
        self.assertEqual(error["invalid_count"], 2)
        self.assertEqual(error["finite_count"], 0)
        self.assertIsNone(error["mean"])

    def test_percentile_contract_and_invalid_track(self):
        self.assertIsNone(metrics.percentile([], 0.95))
        self.assertTrue(math.isclose(metrics.percentile([0, 10], 0.95), 9.5))
        image = FakeImage("A", 0, [((50, 50), 1)])
        bad = SimpleNamespace(images={1: image}, reg_image_ids=lambda: [1], points3D={
            1: point((0, 0, 2), (1, 0), (1, 0))})
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            metrics.summarize_model(bad, ["A"])

    def test_broken_2d_3d_backlink_rejected(self):
        images = {1: FakeImage("A", 0, [((50, 50), 99)]),
                  2: FakeImage("B", 0, [((50, 50), 1)])}
        model = SimpleNamespace(images=images, reg_image_ids=lambda: [1, 2], points3D={
            1: point((0, 0, 2), (1, 0), (2, 0))})
        with self.assertRaisesRegex(ValueError, "backlink"):
            metrics.summarize_model(model, ["A", "B"])

    @unittest.skipUnless(importlib.util.find_spec("pycolmap"), "optional native PyCOLMAP smoke")
    def test_native_pycolmap_known_two_camera_projection(self):
        import numpy as np
        import pycolmap

        reconstruction = pycolmap.Reconstruction()
        reconstruction.add_camera(pycolmap.Camera.create(
            1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 100, 100))
        for image_id, tx, xy in ((1, 0.0, (50.0, 50.0)), (2, -1.0, (0.0, 50.0))):
            image = pycolmap.Image(
                name=f"NP3_{(image_id - 1) * 6:03}.jpg",
                keypoints=np.asarray([xy], dtype=np.float64),
                cam_from_world=pycolmap.Rigid3d(
                    pycolmap.Rotation3d(), np.asarray([tx, 0.0, 0.0])),
                camera_id=1, id=image_id)
            reconstruction.add_image(image)
            reconstruction.register_image(image_id)
        track = pycolmap.Track([pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0)])
        reconstruction.add_point3D(np.asarray([0.0, 0.0, 2.0]), track)
        report = metrics.summarize_model(reconstruction, ["NP3_000.jpg", "NP3_006.jpg"])
        self.assertEqual(report["registered_count"], 2)
        self.assertEqual(report["reprojection_l2_pixels"]["finite_count"], 2)
        self.assertEqual(report["reprojection_l2_pixels"]["mean"], 0.0)

    @unittest.skipUnless(importlib.util.find_spec("pycolmap"), "optional native PyCOLMAP smoke")
    def test_native_hashbound_end_to_end_small_model(self):
        import numpy as np
        import pycolmap

        temp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            base = Path(directory)
            run = base / "run"
            model_dir = run / "sparse/0"
            model_dir.mkdir(parents=True)
            names = [f"NP3_{angle:03}.jpg" for angle in range(0, 288, 6)]
            package_path = base / "package.json"
            package_path.write_text(json.dumps({
                "schema": "ycb_object_evaluation_package_v1", "object_id": "006_mustard_bottle",
                "training_inputs": [{"path": "photos/" + name, "sha256": f"{i:064x}"}
                                    for i, name in enumerate(names)]}))
            reconstruction = pycolmap.Reconstruction()
            reconstruction.add_camera(pycolmap.Camera.create(
                1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 100.0, 100, 100))
            for image_id, tx, xy in ((1, 0.0, (50.0, 50.0)),
                                     (2, -1.0, (0.0, 50.0)),
                                     (3, -2.0, (0.0, 50.0))):
                image = pycolmap.Image(
                    name=names[image_id - 1],
                    keypoints=np.asarray([xy], dtype=np.float64),
                    cam_from_world=pycolmap.Rigid3d(
                        pycolmap.Rotation3d(), np.asarray([tx, 0.0, 0.0])),
                    camera_id=1, id=image_id)
                reconstruction.add_image(image)
                reconstruction.register_image(image_id)
            reconstruction.add_point3D(np.asarray([0.0, 0.0, 2.0]),
                                       pycolmap.Track([pycolmap.TrackElement(1, 0),
                                                      pycolmap.TrackElement(2, 0)]))
            reconstruction.write_binary(str(model_dir))
            with sqlite3.connect(run / "database.db") as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
                connection.executemany("INSERT INTO images VALUES (?,?)",
                                       [(i + 1, name) for i, name in enumerate(names)])
                connection.execute("INSERT INTO two_view_geometries VALUES (?,?)",
                                   (metrics.PAIR_BASE + 2, 10))
            (run / "sfm.json").write_text(json.dumps({
                "registered_images": 3, "sparse_points": 1,
                "registered_names": names[:3], "candidate_models": [{"index": 0}]}))
            (run / "result.json").write_text(json.dumps({
                "schema": "classical_backend_v1", "status": "failed",
                "failure": "registered image fraction below threshold",
                "sfm_source": {"kind": "internal_image_only_pycolmap"},
                "inputs": [{"name": name, "sha256": f"{i:064x}"}
                           for i, name in enumerate(names)],
                "stages": [{"name": name, "status": "complete"}
                           for name in ("features", "matching", "sfm")]}))
            output = base / "metrics.json"
            with patch.object(metrics, "PACKAGE_SHA", metrics.sha256(package_path)), \
                 patch.object(metrics, "MIN_FREE", 0):
                report = metrics.evaluate(run, package_path, output)
            self.assertEqual(report["model"]["registered_count"], 3)
            self.assertEqual(report["matching_graph"]["component_count"], 47)
            self.assertEqual(report["model"]["reprojection_l2_pixels"]["finite_count"], 2)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
