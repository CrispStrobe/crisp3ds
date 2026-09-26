"""Diagnostics tests with the pinned PyCOLMAP/NumPy environment."""

import json
from pathlib import Path
from types import SimpleNamespace as Obj
import unittest

from scripts.object_motion.diagnose import diagnose
from scripts.object_motion.check_interface import compare


class Transform:
    def inverse(self):
        import numpy as np
        return Obj(translation=np.array([1.0, 2.0, 3.0]))


class DiagnoseTest(unittest.TestCase):
    def test_integer_num_points_property_and_track(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is only installed in the local PyCOLMAP environment")
        point2d = Obj(xy=np.array([15.0, 201.0]), has_point3D=lambda: True)
        image = Obj(name="NP3_000.jpg", camera_id=1, num_points3D=1,
                    image_id=1, points2D=[point2d], cam_from_world=Transform())
        camera = Obj(model=Obj(name="SIMPLE_RADIAL"), width=1280, height=1024,
                     params=np.array([1000.0, 640.0, 512.0, 0.01]))
        point = Obj(track=Obj(elements=[Obj(image_id=1, point2D_idx=0),
                                        Obj(image_id=1, point2D_idx=0)]))
        model = Obj(reg_image_ids=lambda: [1], images={1: image}, cameras={1: camera},
                    points3D={1: point})
        manifest = {"images": [{"name": "NP3_000.jpg", "angle_degrees": 0,
                                "pose_support_scanlines": [[200, 10, 20], [201, 10, 20]]}]}
        report = diagnose(model, manifest)
        self.assertEqual(report["registered"], 1)
        self.assertEqual(report["tracks"]["all_inside"], 1)
        self.assertTrue(report["cameras"][0]["focal_plausible_0_3_to_3_widths"])

    def test_existing_raw_model_when_available(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("PyCOLMAP is only installed in the local environment")
        root = Path(__file__).resolve().parents[2]
        model_dir = root / "build-opencv/object-motion/raw/models/0"
        manifest_path = root / "build-opencv/object-motion/prepare-001/manifest.json"
        if not model_dir.exists() or not manifest_path.exists():
            self.skipTest("local real sparse fixture not prepared")
        model = pycolmap.Reconstruction(str(model_dir))
        report = diagnose(model, json.loads(manifest_path.read_text()))
        self.assertEqual(report["registered"], model.num_reg_images())
        self.assertEqual(sum(report["tracks"].values()), model.num_points3D())

    def test_existing_foreground_orbit_when_available(self):
        try:
            import pycolmap
        except ImportError:
            self.skipTest("PyCOLMAP is only installed in the local environment")
        root = Path(__file__).resolve().parents[2]
        model_dir = root / "build-opencv/object-motion/foreground/models/0"
        manifest_path = root / "build-opencv/object-motion/prepare-001/manifest.json"
        if not model_dir.exists() or not manifest_path.exists():
            self.skipTest("local foreground model not prepared")
        report = diagnose(pycolmap.Reconstruction(str(model_dir)), json.loads(manifest_path.read_text()))
        self.assertEqual(report["registered"], 60)
        orbit = report["camera_center_orbit_diagnostic"]
        self.assertEqual(orbit["steps_in_dominant_direction"], 59)
        self.assertGreater(abs(orbit["angular_span_degrees"]), 340)

    def test_existing_openmvs_camera_roundtrip_when_available(self):
        try:
            import pycolmap  # noqa: F401
        except ImportError:
            self.skipTest("PyCOLMAP is only installed in the local environment")
        root = Path(__file__).resolve().parents[2]
        source = root / "build-opencv/classical-ycb-foreground-002/dense/sparse"
        exported = root / "build-opencv/object-motion/roundtrip-001/export/sparse"
        if not source.exists() or not exported.exists():
            self.skipTest("local OpenMVS reverse export not prepared")
        report = compare(source, exported)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["metrics"]["tested_projections"], 3000)


if __name__ == "__main__":
    unittest.main()
