import math
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
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
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
                connection.executemany("INSERT INTO images VALUES (?,?)", [(1, "A"), (2, "B"), (3, "C")])
                connection.executemany("INSERT INTO two_view_geometries VALUES (?,?)", [
                    (metrics.PAIR_BASE + 2, 10), (2 * metrics.PAIR_BASE + 3, 0)])
                connection.commit()
            graph = metrics.verified_graph(database, ["A", "B", "C"])
            self.assertEqual(graph["component_sizes"], [2, 1])
            self.assertEqual(graph["verified_pair_edges"], 1)

    def test_verified_graph_closes_connection_on_success_and_failure(self):
        temp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            database = Path(directory) / "database.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
                connection.execute("INSERT INTO images VALUES (1, 'A')")
                connection.commit()
            connect = sqlite3.connect
            opened = []

            def tracked_connect(*args, **kwargs):
                connection = connect(*args, **kwargs)
                opened.append(connection)
                return connection

            with patch.object(metrics.sqlite3, "connect", side_effect=tracked_connect):
                self.assertEqual(metrics.verified_graph(database, ["A"])["component_count"], 1)
                with self.assertRaisesRegex(ValueError, "inventory differs"):
                    metrics.verified_graph(database, ["B"])
            self.assertEqual(len(opened), 2)
            for connection in opened:
                with self.assertRaises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

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
        with self.assertRaisesRegex(ValueError, "duplicate track observation"):
            metrics.summarize_model(bad, ["A"])

    def test_distinct_2d_observations_in_same_image_are_counted_not_hidden(self):
        images = {1: FakeImage("A", 0, [((50, 50), 1), ((51, 50), 1)]),
                  2: FakeImage("B", 0, [((50, 50), 1)])}
        model = SimpleNamespace(images=images, reg_image_ids=lambda: [1, 2], points3D={
            1: point((0, 0, 2), (1, 0), (1, 1), (2, 0))})
        report = metrics.summarize_model(model, ["A", "B"])
        self.assertEqual(report["repeated_image_tracks"], 1)
        self.assertEqual(report["repeated_image_observations"], 1)
        self.assertEqual(report["track_length"]["at_least_three_count"], 1)
        self.assertEqual(report["distinct_image_track_length"]["at_least_three_count"], 0)
        self.assertEqual(report["reprojection_l2_pixels"]["track_observation_denominator"], 3)

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
            with closing(sqlite3.connect(run / "database.db")) as connection:
                connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
                connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
                connection.executemany("INSERT INTO images VALUES (?,?)",
                                       [(i + 1, name) for i, name in enumerate(names)])
                connection.execute("INSERT INTO two_view_geometries VALUES (?,?)",
                                   (metrics.PAIR_BASE + 2, 10))
                connection.commit()
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


class FailedSparseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[2] / ".local-tools/tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / "run"
        (self.run / "models").mkdir(parents=True)
        self.producer_source = self.base / "historical-run.py"
        self.producer_source.write_bytes(
            (metrics.ROOT / "scripts/classical_backend/run.py").read_bytes())
        self.snapshot = self.base / "database-snapshot.db"
        self.package = self.base / "package.json"
        self.names = [f"NP3_{index * 6:03}.jpg" for index in range(48)]
        self.rows = [{"path": "photos/" + name, "sha256": f"{index:064x}"}
                     for index, name in enumerate(self.names)]
        self.package.write_text(json.dumps({"schema": "ycb_object_evaluation_package_v1",
                                            "object_id": "006_mustard_bottle",
                                            "training_inputs": self.rows}))
        with closing(sqlite3.connect(self.run / "database.db")) as connection:
            connection.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
            connection.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
            connection.executemany("INSERT INTO images VALUES (?,?)",
                                   [(index + 1, name) for index, name in enumerate(self.names)])
            connection.execute("INSERT INTO two_view_geometries VALUES (?,?)",
                               (metrics.PAIR_BASE + 2, 15))
            connection.commit()
        self.snapshot.write_bytes((self.run / "database.db").read_bytes())
        source = {"kind": "internal_image_only_pycolmap", "random_seed": 20260927,
                  "camera_model": "SIMPLE_RADIAL", "camera_mode": "SINGLE",
                  "matching": "sequential"}
        (self.run / "pycolmap-options.json").write_text(json.dumps({
            "version": "3.11.1", "seed": 20260927, "camera_model": "SIMPLE_RADIAL",
            "camera_mode": "SINGLE", "matching_strategy": "sequential"}))
        self.result = {"schema": "classical_backend_v1", "status": "failed",
                       "failure": "sfm failed: exit 1", "sfm_source": source,
                       "inputs": [{"name": name, "sha256": row["sha256"]}
                                  for name, row in zip(self.names, self.rows)],
                       "stages": [{"name": name, "status": "failed" if name == "sfm" else "complete",
                                   "exit_code": 1 if name == "sfm" else 0,
                                   "log": str(self.run / f"{name}.log")}
                                  for name in ("features", "matching", "sfm")],
                       "changed_source_images": [], "changed_pose_masks": [],
                       "changed_binaries": [], "software": {"runner_sha256":
                           metrics.sha256(self.producer_source)}}
        for name in ("features", "matching"):
            (self.run / f"{name}.log").write_text(name + " complete\n")
        self.no_model_log = "ValueError: SfM produced no model\n"
        (self.run / "sfm.log").write_text(self.no_model_log)
        self.save_result()

    def save_result(self):
        (self.run / "result.json").write_text(json.dumps(self.result))

    def evaluate(self, output_name="metrics.json"):
        with patch.object(metrics, "PACKAGE_SHA", metrics.sha256(self.package)), \
             patch.object(metrics, "MIN_FREE", 0):
            return metrics.evaluate_failed_sfm(self.run, self.package, self.base / output_name,
                                               self.snapshot, self.producer_source)

    def test_no_model_graph_only_and_no_invented_geometry(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "failed_no_model")
        self.assertEqual(report["matching_graph"]["component_sizes"], [2] + [1] * 46)
        self.assertIsNone(report["model"])
        self.assertEqual(report["model_role"], "unavailable_no_model")
        self.assertFalse(report["accepted_sparse_export"])
        self.assertIn(str((self.run / "sfm.log").resolve()), report["source_sha256"])
        self.assertEqual(report["source_sha256"][str(self.snapshot.resolve())],
                         report["source_sha256"][str((self.run / "database.db").resolve())])

    def test_snapshot_or_historical_producer_drift_rejected(self):
        self.snapshot.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "database snapshot"):
            self.evaluate()
        self.snapshot.write_bytes((self.run / "database.db").read_bytes())
        self.producer_source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "producer"):
            self.evaluate()

    def test_nonempty_source_wal_or_snapshot_sidecar_rejected(self):
        (self.run / "database.db-wal").write_bytes(b"uncheckpointed")
        with self.assertRaisesRegex(ValueError, "database snapshot"):
            self.evaluate()
        (self.run / "database.db-wal").unlink()
        (self.base / "database-snapshot.db-shm").write_bytes(b"sidecar")
        with self.assertRaisesRegex(ValueError, "database snapshot"):
            self.evaluate()

    def test_no_model_rejects_wrong_failure_or_hidden_export(self):
        self.result["stages"][-1]["status"] = "complete"
        self.save_result()
        with self.assertRaisesRegex(ValueError, "producer"):
            self.evaluate()
        self.result["stages"][-1]["status"] = "failed"
        self.save_result()
        (self.run / "sparse/0").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "accepted sparse"):
            self.evaluate()

    def test_rejected_two_camera_candidate_is_not_success(self):
        (self.run / "sfm.log").write_text(
            "ValueError: SfM model has fewer than 3 cameras or no points\n")
        (self.run / "sfm.json").write_text(json.dumps({
            "registered_images": 2, "sparse_points": 3,
            "registered_names": self.names[:2], "selected_model_index": 0,
            "candidate_models": [{"index": 0, "registered_images": 2, "sparse_points": 3}]}))
        model_dir = self.run / "models/0"
        model_dir.mkdir()
        for name in metrics.MODEL_FILES:
            (model_dir / name).write_bytes(name.encode())
        fake = SimpleNamespace(Reconstruction=lambda path: object())
        model_report = {"registered_count": 2, "sparse_points": 3,
                        "registered_names": sorted(self.names[:2])}
        with patch.dict(sys.modules, {"pycolmap": fake}), \
             patch.object(metrics, "summarize_model", return_value=model_report):
            report = self.evaluate()
        self.assertEqual(report["status"], "failed_degenerate_model")
        self.assertEqual(report["model_role"], "rejected_candidate_model")
        self.assertFalse(report["accepted_sparse_export"])
        self.assertEqual(report["model"]["registered_count"], 2)


if __name__ == "__main__":
    unittest.main()
