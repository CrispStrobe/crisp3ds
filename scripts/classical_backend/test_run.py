import sys
import argparse
from contextlib import closing
import json
import hashlib
import contextlib
import io
import sqlite3
from types import SimpleNamespace
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
import zipfile

from scripts.classical_backend import run
from scripts.classical_backend import fetch_openmvs
from scripts.classical_backend import finish
from scripts.classical_backend import finish_rough
from scripts.classical_backend import masked_dense


class SelectionAndFailureTests(unittest.TestCase):
    def test_initial_pair_sqlite_handle_closes_on_success_and_query_error(self):
        class Connection:
            def __init__(self, fail):
                self.fail = fail
                self.closed = False

            def execute(self, *_):
                if self.fail:
                    raise sqlite3.DatabaseError("query failed")
                return SimpleNamespace(fetchall=lambda: [(7, "a.jpg"), (3, "b.jpg")])

            def close(self):
                self.closed = True

        for fail in (False, True):
            connection = Connection(fail)
            with mock.patch.object(run.sqlite3, "connect", return_value=connection):
                if fail:
                    with self.assertRaises(sqlite3.DatabaseError):
                        run.init_pair_ids(Path("dummy.db"), ["a.jpg", "b.jpg"], ["a.jpg", "b.jpg"])
                else:
                    self.assertEqual(run.init_pair_ids(Path("dummy.db"), ["a.jpg", "b.jpg"],
                                                        ["a.jpg", "b.jpg"]), (7, 3))
            self.assertTrue(connection.closed)

    def test_finish_rough_rejects_symlink_and_unbounded_settings_before_source_read(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            link = root / "source-link"
            link.symlink_to(source, target_is_directory=True)
            args = argparse.Namespace(source=link, output=root / "fresh", binary_dir=root,
                                      max_threads=2, max_gib=1, timeout_minutes=10,
                                      max_rss_gib=10, max_log_mib=32)
            with self.assertRaisesRegex(ValueError, "symlink"):
                finish_rough.finish(args)
            args.source = source
            args.max_gib = float("nan")
            with self.assertRaisesRegex(ValueError, "bounded"):
                finish_rough.finish(args)

    def test_small_view_default_mapper_minimum_clamps_but_explicit_oversize_rejects(self):
        self.assertEqual(run.effective_sfm_min_model_size(None, 3), 3)
        self.assertEqual(run.effective_sfm_min_model_size(None, 80), 10)
        with self.assertRaisesRegex(ValueError, "selected photo count"):
            run.effective_sfm_min_model_size(10, 3)
        argv = ["classical", "--images", "photos", "--output", "out", "--max-views", "3"]
        with (mock.patch.object(sys, "argv", argv),
              mock.patch.object(run, "run", return_value={"status": "sparse_complete"}) as execute,
              mock.patch("builtins.print")):
            self.assertEqual(run.main(), 0)
        self.assertIsNone(execute.call_args.args[0].sfm_min_model_size)
        with (mock.patch.object(sys, "argv", argv + ["--sfm-min-model-size", "10"]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
            run.main()

    def test_generic_sfm_retry_policy_rejects_two_view_dead_end(self):
        class Options:
            def __init__(self):
                self.mapper = SimpleNamespace()
        fake_pycolmap = SimpleNamespace(IncrementalPipelineOptions=Options)
        options = run.sfm_options(fake_pycolmap, 5, 10, None)
        self.assertTrue(options.multiple_models)
        self.assertEqual(options.max_num_models, 5)
        self.assertEqual(options.min_model_size, 10)
        self.assertEqual(options.mapper.num_threads, 2)
        assisted = run.sfm_options(fake_pycolmap, 1, 2, (9, 4))
        self.assertFalse(assisted.multiple_models)
        self.assertEqual((assisted.init_image_id1, assisted.init_image_id2), (9, 4))

    def test_masked_dense_rejects_wrong_native_stem_even_with_valid_hashes(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            base = Path(tmp)
            source = base / "source"
            masks = base / "masks"
            masks.mkdir()
            (source / "dense" / "images").mkdir(parents=True)
            for model_dir in (source / "sparse" / "0", source / "dense" / "sparse"):
                model_dir.mkdir(parents=True)
                for name in ("cameras.bin", "images.bin", "points3D.bin"):
                    (model_dir / name).write_bytes(name.encode())
            (source / "dense" / "images" / "photo.jpg").write_bytes(b"image")
            (masks / "photo.mask.png").write_bytes(b"mask")
            stages = [{"name": name, "status": "complete"} for name in
                      ("reuse_sfm", "undistort", "import")]
            (source / "result.json").write_text(json.dumps({"schema": "classical_backend_v1",
                                                               "stages": stages, "inputs": [{"name": "photo.jpg"}]}))
            report = {"schema": "classical_dense_masks_v1", "status": "complete",
                      "source_model": str(source / "sparse" / "0"),
                      "source_model_sha256": masked_dense.model_hashes(source / "sparse" / "0"),
                      "undistorted_model": str(source / "dense" / "sparse"),
                      "undistorted_model_sha256": masked_dense.model_hashes(source / "dense" / "sparse"),
                      "max_pose_matrix_difference": 0, "ignore_mask_label": 0,
                      "native_filename_basis": "image stem",
                      "images": [{"name": "photo.jpg", "native_mask_name": "photo.mask.png",
                                  "undistorted_image_sha256": run.digest(source / "dense" / "images" / "photo.jpg"),
                                  "mask_sha256": run.digest(masks / "photo.mask.png")}]}
            masked_dense.validate_inputs(source, masks, report)
            report["images"][0]["native_mask_name"] = "photo.jpg.mask.png"
            with self.assertRaisesRegex(ValueError, "filename differs"):
                masked_dense.validate_inputs(source, masks, report)

    def test_initial_pair_uses_names_not_threaded_insertion_order(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            database = Path(tmp) / "database.db"
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
                    connection.executemany("INSERT INTO images VALUES (?, ?)",
                                           [(1, "c.jpg"), (9, "a.jpg"), (4, "b.jpg")])
            self.assertEqual(run.init_pair_ids(database, ["a.jpg", "b.jpg", "c.jpg"],
                                               ["a.jpg", "b.jpg"]), (9, 4))
            with self.assertRaisesRegex(ValueError, "two distinct"):
                run.init_pair_ids(database, ["a.jpg", "b.jpg"], ["a.jpg", "a.jpg"])
            with self.assertRaisesRegex(ValueError, "missing"):
                run.init_pair_ids(database, ["a.jpg", "d.jpg"], ["a.jpg", "d.jpg"])

    def test_selection_rejects_duplicate_or_escaping_names(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (root / name).write_bytes(b"photo")
            names = root / "list.txt"
            names.write_text("a.jpg\na.jpg\nb.jpg\n")
            with self.assertRaisesRegex(ValueError, "unique basenames"):
                run.photos(root, names, 3)
            names.write_text("a.jpg\n../escape.jpg\nc.jpg\n")
            with self.assertRaisesRegex(ValueError, "unique basenames"):
                run.photos(root, names, 3)

    def test_native_executable_path_uses_windows_suffix(self):
        base = Path("tools")
        self.assertEqual(run.tool_path(base, "TextureMesh", windows=False), base / "TextureMesh")
        self.assertEqual(run.tool_path(base, "TextureMesh", windows=True), base / "TextureMesh.exe")

    def test_filtered_finish_rechecks_photo_mask_and_model_bytes(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            camera = model / "cameras.bin"
            photo = root / "image.jpg"
            mask = root / "image.jpg.png"
            camera.write_bytes(b"camera")
            photo.write_bytes(b"photo")
            mask.write_bytes(b"mask1")
            manifest = {"images": [{"name": "image.jpg", "path": str(photo),
                                    "pose_support_mask": str(mask)}]}
            filtering = {"model_dir": str(model), "model_files_sha256": {"cameras.bin": run.digest(camera)},
                         "image_sha256": {"image.jpg": run.digest(photo)},
                         "mask_sha256": {"image.jpg": run.digest(mask)}}
            finish.verify_filter_files(filtering, manifest)
            mask.write_bytes(b"mask2")
            with self.assertRaisesRegex(ValueError, "mask changed"):
                finish.verify_filter_files(filtering, manifest)

    def test_stage_timeout_is_recorded_and_process_stopped(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            with self.assertRaises(run.StageError) as raised:
                run.stage(root, "timeout", [sys.executable, "-c", "import time; time.sleep(30)"],
                          time.monotonic() + 0.1, 1 << 20, 1 << 20, 1 << 30)
            result = raised.exception.result
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure"], "deadline exceeded")
            self.assertTrue((root / "timeout.log").is_file())

    def test_folder_size_tolerates_native_atomic_replacement(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            (root / "stable.dmap").write_bytes(b"1234")
            (root / "vanishing.dmap").write_bytes(b"old")
            original = Path.stat
            def transient_stat(path, *args, **kwargs):
                if path.name == "vanishing.dmap":
                    raise FileNotFoundError(path)
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "stat", transient_stat):
                self.assertEqual(run.folder_bytes(root), 4)

    def test_artifact_header_rejects_empty_geometry(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            path = Path(tmp) / "empty.ply"
            path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\n"
                             b"element face 0\nend_header\n" + b" " * 40)
            with self.assertRaises(ValueError):
                run.checked_ply(path, "face")

    def test_pinned_fetch_rejects_same_size_binary_tamper(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            destination = root / "tools"
            destination.mkdir()
            archive = destination / "OpenMVS_macOS_arm64-v2.4.0.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("InterfaceCOLMAP", b"real binary")
            with (mock.patch.object(fetch_openmvs, "DEST", destination),
                  mock.patch.object(fetch_openmvs, "TMP", root),
                  mock.patch.object(fetch_openmvs, "SIZE", archive.stat().st_size),
                  mock.patch.object(fetch_openmvs, "SHA256", fetch_openmvs.sha256(archive)),
                  mock.patch.object(fetch_openmvs, "SELECTED", ("InterfaceCOLMAP",))):
                fetch_openmvs.fetch()
                target = destination / "bin" / "InterfaceCOLMAP"
                self.assertEqual(target.read_bytes(), b"real binary")
                target.write_bytes(b"fake binary")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    fetch_openmvs.fetch()

    def test_pinned_fetch_rejects_oversized_member(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            destination = root / "tools"
            destination.mkdir()
            archive = destination / "OpenMVS_macOS_arm64-v2.4.0.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.writestr("InterfaceCOLMAP", b"x" * 32_000_001)
            with (mock.patch.object(fetch_openmvs, "DEST", destination),
                  mock.patch.object(fetch_openmvs, "TMP", root),
                  mock.patch.object(fetch_openmvs, "SIZE", archive.stat().st_size),
                  mock.patch.object(fetch_openmvs, "SHA256", fetch_openmvs.sha256(archive)),
                  mock.patch.object(fetch_openmvs, "SELECTED", ("InterfaceCOLMAP",))):
                with self.assertRaisesRegex(ValueError, "unexpected archive member"):
                    fetch_openmvs.fetch()

    def test_shared_sfm_photo_hash_comparison_reaches_stage(self):
        """Regression: a photo Path must not shadow the producer provenance record."""
        with tempfile.TemporaryDirectory(dir=run.ROOT / ".local-tools/tmp") as tmp:
            root = Path(tmp)
            images = root / "photos"
            images.mkdir()
            photo_hashes = {}
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                path = images / name
                path.write_bytes(name.encode())
                photo_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = root / "models" / "0"
            model.mkdir(parents=True)
            model_hashes = {}
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                path = model / name
                path.write_bytes(name.encode())
                model_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            mask_hashes = {name: "mask-" + name for name in photo_hashes}
            manifest.write_text(json.dumps({"images": [{"name": name, "sha256": photo_hashes[name],
                                                         "mask_sha256": mask_hashes[name]}
                                                        for name in photo_hashes]}))
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps({"schema": "object_motion_run_v1", "arm": "raw",
                                              "input_manifest_sha256": run.digest(manifest),
                                              "image_hashes": photo_hashes, "mask_hashes": mask_hashes}))
            summary = root / "summary.json"
            summary.write_text(json.dumps({"arm": "raw", "model_count": 1,
                                           "model_dir": str(model.resolve()),
                                           "model_files_sha256": model_hashes,
                                           "producer_provenance_sha256": run.digest(provenance)}))
            args = argparse.Namespace(images=images, output=root / "out", image_list=None,
                                      pose_mask_dir=None,
                                      sparse_model=model, sparse_provenance=provenance,
                                      sparse_manifest=manifest, sparse_result=summary,
                                      binary_dir=root, python=Path(sys.executable), max_views=3,
                                      max_image_size=1200, max_threads=2, camera_model="SIMPLE_RADIAL",
                                      matching="exhaustive", sequential_overlap=8,
                                      sift_max_features=1800, sift_max_image_size=1200, seed=0,
                                      init_image_pair=None,
                                      sfm_max_models=5, sfm_min_model_size=None, stop_after_sfm=False,
                                      min_registered_fraction=0.7, max_gib=0.1,
                                      max_rss_gib=1, max_log_mib=1, timeout_minutes=1)
            failure = {"name": "reuse_sfm", "status": "failed", "exit_code": 99,
                       "failure": "intentional unit stop"}
            with (mock.patch.object(run, "check_toolchain", return_value={"openmvs_binaries": {}}),
                  mock.patch.object(run, "stage", side_effect=run.StageError(failure)) as called_stage):
                result = run.run(args)
            command = called_stage.call_args.args[2]
            self.assertEqual(command[command.index("--sfm-min-model-size") + 1], "3")
            self.assertEqual(result["sfm_source"]["kind"], "external_raw_image_only_verified_model")
            self.assertEqual(len(result["inputs"]), 3)
            self.assertEqual(result["stages"][-1]["name"], "reuse_sfm")
            self.assertEqual(result["failure"], "reuse_sfm failed: intentional unit stop")


if __name__ == "__main__":
    unittest.main()
