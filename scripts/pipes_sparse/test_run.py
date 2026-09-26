"""Small geometry and input-boundary checks for the preregistered baseline."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.pipes_sparse import run
from scripts.fixed_camera.run import build_tracks, reconstruct
from scripts.sparse_verify.verify import project


class PipesSparseTests(unittest.TestCase):
    def test_gcd_backend_pool_disabled_and_actual_serial(self):
        with patch.object(run.cv2, "setNumThreads") as setting, patch.object(
                run.cv2, "getNumThreads", return_value=1):
            self.assertEqual(run.configure_opencv_threads(), 1)
            setting.assert_called_once_with(0)
        with patch.object(run.cv2, "setNumThreads"), patch.object(
                run.cv2, "getNumThreads", return_value=8):
            with self.assertRaises(ValueError):
                run.configure_opencv_threads()

    def test_actual_anisotropic_resize_and_pixel_edge_frame(self):
        source = {"model": "PINHOLE", "width": 6220, "height": 4141,
                  "params": [3430.27, 3429.23, 3119.2, 2057.75]}
        resized, (sx, sy) = run.resize_camera(source)
        self.assertEqual((resized["width"], resized["height"]), (1024, 682))
        self.assertNotEqual(sx, sy)
        xyz = [0.2, -0.3, 3.7]
        before, after = project(source, xyz), project(resized, xyz)
        self.assertAlmostEqual(after[0], before[0] * sx, places=10)
        self.assertAlmostEqual(after[1], before[1] * sy, places=10)
        kp = cv2.KeyPoint(after[0] - .5, after[1] - .5, 4)
        np.testing.assert_allclose(run.edge_xy(kp), after, atol=3e-5)

    def test_strict_bidirectional_ratio_rejects_ambiguous_reverse(self):
        a = np.array([[0., 0.], [1., 0.], [10., 0.]], dtype=np.float32)
        b = np.array([[.45, 0.], [10., 0.], [30., 0.]], dtype=np.float32)
        self.assertEqual(run.mutual_ratio(a, b), [(2, 1)])

    def test_conflict_component_and_fixed_ray_reconstruction(self):
        edges = [(('a', 0), ('b', 0)), (('b', 0), ('a', 1)),
                 (('a', 2), ('b', 2))]
        tracks, rejected, total = build_tracks(edges)
        self.assertEqual((total, rejected), (2, {"same_image_conflict": 1}))
        self.assertEqual(tracks, [(('a', 2), ('b', 2))])
        cam = {0: {"model": "PINHOLE", "width": 1024, "height": 682,
                   "params": [600., 601., 512., 341.]}}
        images = {}
        features = {}
        xyz = np.array([.2, -.1, 4.])
        for name, center_x in (("a", 0.), ("b", 1.)):
            t = [-center_x, 0., 0.]
            images[name] = {"camera_id": 0, "R": np.eye(3).tolist(),
                            "t": t, "center": [center_x, 0., 0.]}
            features[name] = [project(cam[0], (xyz + np.asarray(t)).tolist())]
        point, reason = reconstruct((("a", 0), ("b", 0)), images, cam, features)
        self.assertIsNone(reason)
        np.testing.assert_allclose(point["xyz"], xyz, atol=1e-9)

    def test_manifest_is_only_allowlisted_image_input(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "pipes/images/dslr_images_undistorted").mkdir(parents=True)
            camera = {"model": "PINHOLE", "width": 8, "height": 6,
                      "params": [5., 5., 4., 3.]}
            selected = []
            for ident, name in enumerate(run.NAMES):
                path = data / "pipes/images/dslr_images_undistorted" / name
                path.write_bytes(b"image-fixture")
                selected.append({"id": ident, "name": name,
                                 "path": f"pipes/images/dslr_images_undistorted/{name}",
                                 "size_bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "camera_id": 0, "camera": camera,
                                 "qvec": [1., 0., 0., 0.], "tvec": [0., 0., 0.],
                                 "pose_convention": "world_to_camera"})
            (data / "scan1.ply").write_bytes(b"ground-truth-must-not-be-read")
            (data / "points3D.txt").write_bytes(b"supplied-SfM-points-must-not-be-read")
            all_images = selected + [{"name": f"DSC_{n:04d}.JPG"} for n in range(638, 648)]
            digests = {r["path"]: r["sha256"] for r in selected}
            (data / "prepare-metadata.json").write_text(json.dumps({"status": "validated",
                "images": all_images, "file_sha256": digests,
                "scan": "scan1.ply", "points": "points3D.txt"}))
            views, _ = run.selected_views(data)
            self.assertEqual([v["name"] for v in views], list(run.NAMES))
            selected[0]["path"] = "scan1.ply"
            (data / "prepare-metadata.json").write_text(json.dumps({"status": "validated",
                "images": all_images, "file_sha256": digests}))
            with self.assertRaises(ValueError):
                run.selected_views(data)


if __name__ == "__main__":
    unittest.main()
