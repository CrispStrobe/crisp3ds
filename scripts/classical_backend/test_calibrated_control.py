import json
import shutil
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts.classical_backend import calibrated_control as control
from scripts.classical_backend.run import ROOT


def fixture_dmap(path: Path, K: np.ndarray, R: np.ndarray, C: np.ndarray,
                 image_size=(641, 512), depth_size=(641, 512)):
    name = b"dense/images/NP3_090.jpg"
    prefix = (control.HEADER.pack(0x5244, 1, 0, *image_size, *depth_size, 0.1, 10.0) +
              struct.pack("<H", len(name)) + name + struct.pack("<II", 1, 16) +
              np.asarray(K, dtype="<f8").tobytes() + np.asarray(R, dtype="<f8").tobytes() +
              np.asarray(C, dtype="<f8").tobytes())
    path.write_bytes(prefix + bytes(depth_size[0] * depth_size[1] * 4))


class CalibratedCameraTests(unittest.TestCase):
    def test_pinned_native_minimum_causes_full_resolution_boundary(self):
        for width, actual_level, target in ((1277, 0, 1277),
                                             (1280, 1, 640), (1282, 1, 641)):
            row = control.native_depth_size(width, 1015, 2, 640, 1600)
            self.assertEqual((row["actual_level"], row["predicted_max_resolution"]),
                             (actual_level, target))
        guarded = control.native_depth_size(1277, 1015, 2, 600, 1600)
        self.assertEqual((guarded["actual_level"], guarded["predicted_max_resolution"]),
                         (1, 638))

    def test_native_pixel_and_disk_preflight_rejects_full_or_over_budget(self):
        sizes = {f"view{i:02}.jpg": (1277, 1015) for i in range(60)}
        with self.assertRaisesRegex(ValueError, "full-resolution"):
            control.native_depth_preflight(sizes, 113_500_000, minimum=640)
        profile = control.native_depth_preflight(sizes, 113_500_000, minimum=600)
        self.assertEqual({row["actual_level"] for row in profile["images"].values()}, {1})
        self.assertLess(profile["predicted_two_generation_peak_bytes"], control.MAX_OUTPUT)
        with self.assertRaisesRegex(ValueError, "exceeds output"):
            control.native_depth_preflight(sizes, 200_000_000, minimum=600)

    def test_local_berkeley_h5_is_read_with_nonzero_k3_and_plus_half_principal(self):
        calibration = (ROOT / "build-opencv/object-motion/ycb-camera-reference-003/"
                       "metadata/calibration.h5")
        if not calibration.exists() or shutil.which("h5dump") is None:
            self.skipTest("verified local Berkeley calibration and h5dump unavailable")
        values = control.calibration_params(calibration)
        self.assertEqual(len(values), 12)
        self.assertAlmostEqual(values[2], 621.8445331259582, places=10)
        self.assertAlmostEqual(values[3], 481.08324224028723, places=10)
        self.assertAlmostEqual(values[8], -0.17716948411083983, places=10)
        self.assertEqual(values[9:], [0.0, 0.0, 0.0])

    def test_feature_pixel_origin_plus_half_matches_independent_opencv_projection(self):
        calibration = (ROOT / "build-opencv/object-motion/ycb-camera-reference-003/"
                       "metadata/calibration.h5")
        try:
            import cv2
            import pycolmap
        except ImportError:
            self.skipTest("OpenCV/PyCOLMAP local calibration test dependencies unavailable")
        if not calibration.exists() or shutil.which("h5dump") is None:
            self.skipTest("verified local Berkeley calibration and h5dump unavailable")
        params = control.calibration_params(calibration)
        camera = pycolmap.Camera(model="FULL_OPENCV", width=1280, height=1024, params=params)
        rays = np.asarray([[-0.55, -0.4, 1], [-0.31, 0.33, 1], [0, 0, 1],
                           [0.43, -0.28, 1], [0.6, 0.4, 1]], dtype=np.float64)
        actual = np.asarray(camera.img_from_cam(rays))
        original_K = np.asarray([[params[0], 0, params[2] - 0.5],
                                 [0, params[1], params[3] - 0.5], [0, 0, 1]])
        expected, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                        original_K, np.asarray(params[4:]))
        np.testing.assert_allclose(actual, expected[:, 0, :] + 0.5, atol=1e-9)

    def test_nonfinite_expected_depth_camera_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid PINHOLE"):
            control.expected_depth_k(np.asarray([1000.0, 1000.0, float("nan"), 512.0]),
                                     (1280, 1024), (640, 512))

    def test_identical_depth_cameras_reproject_to_zero_relative_error(self):
        camera = {"K": np.asarray([[5.0, 0, 2.0], [0, 5.0, 2.0], [0, 0, 1.0]]),
                  "R": np.eye(3), "C": np.zeros(3), "depth_size": (5, 5)}
        depth = np.full((5, 5), 2.0, dtype=np.float32)
        result = control.reprojection_pair(camera, depth, camera, depth)
        self.assertEqual(result["sampled"], 25)
        self.assertEqual(result["compared"], 25)
        self.assertEqual(result["within_2_percent"], 25)
        self.assertEqual(result["median_relative_depth_error"], 0.0)

    def test_producer_rejects_calibration_camera_and_pose_mismatches(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as tmp:
            folder = Path(tmp)
            photos, masks, model = (folder / "photos", folder / "masks", folder / "model")
            for path in (photos, masks, model):
                path.mkdir()
            sample = folder / "sample.jpg"
            Image.new("RGB", (1280, 1024), (100, 120, 140)).save(sample)
            entries, photo_hashes = [], {}
            for index in range(60):
                name = f"NP3_{index * 6:03}.jpg"
                shutil.copyfile(sample, photos / name)
                (masks / (name + ".png")).write_bytes(b"mask")
                photo_hashes[name] = control.digest(photos / name)
                entries.append({"name": name, "sha256": photo_hashes[name],
                                "mask_sha256": control.digest(masks / (name + ".png"))})
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"images": entries}))
            calibration = folder / "calibration.h5"
            calibration.write_bytes(b"calibration")
            for name in control.MODEL_FILES:
                (model / name).write_bytes(name.encode())
            params = [1000.0, 1000.0, 640.5, 512.5, -0.02, 0.1, 0.003, 0.002,
                      -0.17, 0.0, 0.0, 0.0]
            camera = types.SimpleNamespace(model=types.SimpleNamespace(name="FULL_OPENCV"),
                                           params=np.asarray(params), width=1280, height=1024)
            pose = types.SimpleNamespace(rotation=types.SimpleNamespace(matrix=lambda: np.eye(3)),
                                         translation=np.zeros(3))
            images = {index: types.SimpleNamespace(name=entry["name"], has_pose=True,
                                                   camera_id=1, cam_from_world=pose)
                      for index, entry in enumerate(entries)}
            reconstruction = types.SimpleNamespace(images=images, cameras={1: camera},
                                                   points3D={1: types.SimpleNamespace(xyz=np.ones(3))},
                                                   num_points3D=lambda: 1)
            fake_pycolmap = types.SimpleNamespace(Reconstruction=lambda _: reconstruction)
            report = {"schema": control.MODEL_SCHEMA, "lane": "calibrated_intrinsics_only",
                      "provenance_class": "oracle-assisted intrinsics; image-estimated poses",
                      "status": "complete", "pixel_center_convention": control.PIXEL_CONVENTION,
                      "model_dir": str(model), "model_files_sha256": control.model_hashes(model),
                      "calibration_h5_sha256": control.digest(calibration),
                      "full_opencv_params": params, "source_image_hashes": photo_hashes,
                      "registered": 60, "points3D": 1, "reverification_options": {"method": "test"},
                      **{key: "a" * 64 for key in
                         ("source_database_sha256", "cached_features_sha256_before",
                          "cached_features_sha256_after", "raw_matches_sha256_before",
                          "raw_matches_sha256_after", "verified_geometry_sha256_after")}}
            with (mock.patch.dict(sys.modules, {"pycolmap": fake_pycolmap}),
                  mock.patch.object(control, "calibration_params", return_value=params)):
                self.assertEqual(control.validate_producer(report, model, photos, masks,
                                                           manifest, calibration)["registered"], 60)
                changed = report.copy()
                changed["full_opencv_params"] = params.copy()
                changed["full_opencv_params"][8] = 0  # cannot drop nonzero k3
                with self.assertRaisesRegex(ValueError, "calibration HDF5"):
                    control.validate_producer(changed, model, photos, masks, manifest, calibration)
                missing = images.pop(59)
                with self.assertRaisesRegex(ValueError, "registration"):
                    control.validate_producer(report, model, photos, masks, manifest, calibration)
                images[59] = missing
                camera.width = 1279
                with self.assertRaisesRegex(ValueError, "FULL_OPENCV intrinsics"):
                    control.validate_producer(report, model, photos, masks, manifest, calibration)
                camera.width = 1280
                pose.translation = np.asarray([float("nan"), 0, 0])
                with self.assertRaisesRegex(ValueError, "invalid image poses"):
                    control.validate_producer(report, model, photos, masks, manifest, calibration)

    def test_exact_pinned_colmap_to_openmvs_half_pixel_then_scale(self):
        params = np.asarray([1077.5837619820666, 1077.5837619820666, 641.0, 512.0])
        K = control.expected_depth_k(params, (1282, 1024), (641, 512))
        self.assertAlmostEqual(K[0, 0], 538.7918809910333, places=9)
        self.assertAlmostEqual(K[1, 1], 538.7918809910333, places=9)
        self.assertEqual(K[0, 2], 320.0)
        self.assertEqual(K[1, 2], 255.5)
        self.assertNotEqual(K[0, 2], params[2] * 0.5)

    def test_dmap_camera_fixture_enforces_exact_k_rotation_and_center(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as tmp:
            folder = Path(tmp)
            params = np.asarray([1077.5837619820666, 1077.5837619820666, 641.0, 512.0])
            K = control.expected_depth_k(params, (1282, 1024), (641, 512))
            R = np.eye(3)
            C = np.asarray([1.0, 2.0, 3.0])
            path = folder / "depth0016.dmap"
            fixture_dmap(path, K, R, C)
            entry = control.read_dmap_camera(path)
            self.assertEqual(entry["name"], "NP3_090.jpg")
            np.testing.assert_array_equal(entry["K"], K)
            image = types.SimpleNamespace(name="NP3_090.jpg", has_pose=True, camera_id=1,
                                          cam_from_world=types.SimpleNamespace(
                                              rotation=types.SimpleNamespace(matrix=lambda: R),
                                              translation=-C))
            camera = types.SimpleNamespace(model=types.SimpleNamespace(name="PINHOLE"),
                                           params=params, width=1282, height=1024)
            model = types.SimpleNamespace(images={1: image}, cameras={1: camera})
            fake_pycolmap = types.SimpleNamespace(Reconstruction=lambda _: model)
            with mock.patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                result = control.validate_dmap_cameras(folder, folder)
            self.assertEqual(result["checked_views"], 1)
            self.assertEqual(result["max_abs_difference"], {"K": 0.0, "R": 0.0, "C": 0.0})
            shifted = K.copy()
            shifted[0, 2] += 0.01
            fixture_dmap(path, shifted, R, C)
            with mock.patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
                with self.assertRaisesRegex(ValueError, "exact COLMAP resize"):
                    control.validate_dmap_cameras(folder, folder)

    def test_truncated_dmap_and_incomplete_reports_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as tmp:
            folder = Path(tmp)
            path = folder / "depth0001.dmap"
            path.write_bytes(b"DR")
            with self.assertRaisesRegex(ValueError, "truncated"):
                control.read_dmap_camera(path)
            report = folder / "report.json"
            report.write_text(json.dumps({"schema": "classical_dense_masks_v1", "status": "complete",
                                          "images": [{"name": "one.jpg"}]}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                control.mask_count(report, 2)
            report.write_text(json.dumps({"schema": "classical_dmap_camera_check_v1",
                                          "status": "complete", "checked_views": 2}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                control.checked_dmap_count(report, 3)


if __name__ == "__main__":
    unittest.main()
