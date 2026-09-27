"""Small contract checks for the stock image-only Sceaux control."""

import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.upstream_control import image_only


class Option:
    def __init__(self, **values):
        self.__dict__.update(values)

    def todict(self):
        return dict(self.__dict__)


class ImageOnlyTests(unittest.TestCase):
    def test_only_thread_limits_change_numeric_defaults(self):
        fake = types.SimpleNamespace(
            ImageReaderOptions=lambda: Option(camera_model="SIMPLE_RADIAL", camera_params="",
                                               default_focal_length_factor=1.2),
            SiftExtractionOptions=lambda: Option(num_threads=-1, max_num_features=8192,
                                                 max_image_size=3200),
            SiftMatchingOptions=lambda: Option(num_threads=-1, max_ratio=0.8),
            ExhaustiveMatchingOptions=lambda: Option(block_size=50),
            TwoViewGeometryOptions=lambda: Option(max_error=4.0),
            IncrementalPipelineOptions=lambda: Option(
                num_threads=-1, mapper=Option(num_threads=-1),
                multiple_models=True, max_num_models=50, min_model_size=10,
                init_image_id1=-1, init_image_id2=-1),
            CameraMode=types.SimpleNamespace(AUTO=types.SimpleNamespace(name="AUTO")),
            Device=types.SimpleNamespace(cpu=types.SimpleNamespace(name="cpu")),
        )
        reader, extract, matching, _, _, mapping = image_only.options(fake)
        self.assertEqual(reader.camera_params, "")
        self.assertEqual((extract.max_num_features, extract.max_image_size), (8192, 3200))
        self.assertEqual((extract.num_threads, matching.num_threads,
                          mapping.num_threads, mapping.mapper.num_threads), (2, 2, 2, 2))
        self.assertEqual((mapping.multiple_models, mapping.max_num_models,
                          mapping.min_model_size, mapping.init_image_id1), (True, 50, 10, -1))

    def test_only_manifest_images_reach_bounded_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sample = base / "sample"
            images = sample / "images"
            images.mkdir(parents=True)
            rows = []
            for name in image_only.IMAGE_NAMES:
                path = images / name
                path.write_bytes(name.encode())
                rows.append({"path": f"images/{name}", "bytes": path.stat().st_size,
                             "sha256": image_only.digest(path)})
            (sample / "scene.mvs").write_bytes(b"upstream cameras")
            (sample / "scene_dense_mesh.ply").write_bytes(b"upstream mesh")
            output = base / "result"
            python = base / "python"
            python.write_bytes(b"mock interpreter")
            calls = []

            def fake_stage(run, name, command, deadline, max_bytes, max_log,
                           max_rss, extra_reserve_paths):
                calls.append((name, command, max_bytes, max_log, max_rss,
                              extra_reserve_paths))
                database = run / "database.db"
                with sqlite3.connect(database) as connection:
                    for table in ("images", "cameras", "keypoints", "descriptors",
                                  "matches", "two_view_geometries"):
                        connection.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER)")
                (run / "effective-options.json").write_text(json.dumps({"camera_mode": "AUTO"}))
                if name == "mapping":
                    (run / "models.json").write_text(json.dumps([{
                        "index": 0, "registered_images": 11, "sparse_points": 10,
                        "duplicate_image_tracks": 0, "nonfinite_points": 0,
                        "nonfinite_cameras": 0, "nonfinite_poses": 0}]))
                return {"name": name, "status": "complete"}

            manifest = {"commit": "pinned", "files": rows}
            with patch.object(image_only, "verify_sample", return_value=(manifest, "manifest-hash")), \
                 patch.object(image_only, "stage", side_effect=fake_stage), \
                 patch.object(image_only, "toolchain", return_value={"version": "3.11.1"}), \
                 patch.object(image_only.shutil, "disk_usage", return_value=types.SimpleNamespace(free=20 << 30)):
                report = image_only.run(sample, output, python)

            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["positive_control_pass"])
            self.assertEqual([call[0] for call in calls], ["features", "matching", "mapping"])
            self.assertTrue(all(call[2:5] == (512 << 20, 16 << 20, 4 << 30)
                                and call[5] == (image_only.ROOT,) for call in calls))
            self.assertEqual(set(path.name for path in (output / "images").iterdir()),
                             set(image_only.IMAGE_NAMES))
            self.assertFalse((output / "scene.mvs").exists())
            self.assertFalse((output / "scene_dense_mesh.ply").exists())
            self.assertTrue(all("--worker" in call[1] and
                                "scripts.upstream_control.image_only" in call[1]
                                for call in calls))

    def test_refuses_existing_output_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaises(FileExistsError):
                image_only.run(path, path)


if __name__ == "__main__":
    unittest.main()
