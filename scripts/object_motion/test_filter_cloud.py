"""Frozen point-support filter tests; no reference scan input."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend import geometry
from scripts.classical_backend.test_ply import fixture
from scripts.object_motion.filter_cloud import (filter_header, raw_vertices,
    project_simple_radial, retain_by_support, select_records, support_for_camera)

ROOT = Path(__file__).resolve().parents[2]


class FilterCloudTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".local-tools/tmp").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_byte_preservation_and_fresh_output(self):
        cloud = self.base / "native.ply"
        original = fixture(face=False)
        cloud.write_bytes(original)
        header, records, xyz = raw_vertices(cloud)
        self.assertEqual(len(xyz), 3)
        out = self.base / "selected.ply"
        self.assertEqual(select_records(header, records, [True, False, True], out), 2)
        self.assertEqual(out.read_bytes(), filter_header(header, 3, 2) + records[0] + records[2])
        self.assertEqual(geometry.counts(out), (2, 0))
        self.assertEqual(hashlib.sha256(cloud.read_bytes()).hexdigest(), hashlib.sha256(original).hexdigest())
        with self.assertRaises(FileExistsError):
            select_records(header, records, [True, False, True], out)

    def test_frozen_threshold_and_behind_camera(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is in the local PyCOLMAP environment")
        self.assertEqual(retain_by_support([0, 47, 48, 60]).tolist(), [False, False, True, True])
        with self.assertRaises(ValueError):
            retain_by_support([61])
        points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 1], [0, 0.2, 1]], dtype=float)
        matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        support = support_for_camera(points, matrix, [100, 50, 50, 0],
                                     [(50, 40, 60)], 100, 100)
        self.assertEqual(support.tolist(), [True, False, False, False])

    def test_saved_pycolmap_projection_agreement_when_available(self):
        try:
            import numpy as np
            import pycolmap
        except ImportError:
            self.skipTest("PyCOLMAP/NumPy are in the local environment")
        model_path = ROOT / "build-opencv/object-motion/foreground/models/0"
        manifest_path = ROOT / "build-opencv/object-motion/prepare-001/manifest.json"
        if not model_path.exists() or not manifest_path.exists():
            self.skipTest("local image-only model not prepared")
        model = pycolmap.Reconstruction(str(model_path))
        manifest = json.loads(manifest_path.read_text())
        image = next(im for im in model.images.values() if im.name == "NP3_000.jpg")
        camera = model.cameras[image.camera_id]
        points = np.array([point.xyz for _, point in sorted(model.points3D.items())[:100]])
        projected_u, projected_v, z = project_simple_radial(points, image.cam_from_world.matrix(), camera.params)
        result = support_for_camera(points, image.cam_from_world.matrix(), camera.params,
                                    manifest["images"][0]["pose_support_scanlines"],
                                    camera.width, camera.height)
        scanlines = {row: (lo, hi) for row, lo, hi in manifest["images"][0]["pose_support_scanlines"]}
        expected = []
        for index, point in enumerate(points):
            camera_xyz = image.cam_from_world * point
            self.assertAlmostEqual(z[index], camera_xyz[2], places=9)
            if camera_xyz[2] <= 1e-9:
                expected.append(False)
                continue
            uv = image.project_point(point)
            if uv is None:
                expected.append(False)
                continue
            u, v = uv
            self.assertAlmostEqual(float(u), float(projected_u[index]), places=7)
            self.assertAlmostEqual(float(v), float(projected_v[index]), places=7)
            row = int(np.floor(v))
            lo, hi = scanlines.get(row, (np.inf, -np.inf))
            expected.append(bool(0 <= u < camera.width and 0 <= v < camera.height and lo <= u <= hi))
        self.assertEqual(result.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
