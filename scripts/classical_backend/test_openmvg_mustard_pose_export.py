"""Synthetic conversion gate tests; never runs the OpenMVG converter."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_pose_export as export


def sample_model() -> dict:
    names = [f"NP3_{i:03}.jpg" for i in range(48)]
    views = [{"key": i, "value": {"ptr_wrapper": {"data": {
        "filename": name, "id_view": i, "id_pose": i}}}}
             for i, name in enumerate(names)]
    poses = [{"key": i, "value": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   "center": [float(i), 0, 0]}} for i in range(46)]
    return {"views": views, "intrinsics": [{"key": 0, "value": {}}],
            "extrinsics": poses, "structure": [], "control_points": []}


class MustardPoseExportTest(unittest.TestCase):
    def test_fixed_command_exports_only_named_camera_fields(self):
        output = Path("/synthetic")
        argv = export.command(output)
        self.assertEqual(argv[-3:], ["-V", "-I", "-E"])
        self.assertEqual(argv[argv.index("-i") + 1], str(export.INPUT))
        self.assertEqual(argv[argv.index("-o") + 1], str(output / "sfm_camera_poses.json"))
        self.assertNotIn("-S", argv)
        self.assertNotIn("-C", argv)

    def test_json_requires_46_named_finite_poses(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poses.json"
            model = sample_model()
            names = [entry["value"]["ptr_wrapper"]["data"]["filename"] for entry in model["views"]]
            path.write_text(json.dumps(model))
            checked = export.inspect_json(path, names)
            self.assertEqual((checked["views"], checked["poses"]), (48, 46))
            self.assertEqual(checked["missing_names"], names[46:])
            model["extrinsics"][0]["value"]["rotation"][0][0] = 2
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(RuntimeError, "not orthonormal"):
                export.inspect_json(path, names)

    def test_new_output_and_full_disk_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "already"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                export.preflight(existing)
            with patch.object(export.build.fifth.four.fork.base, "tree_bytes", return_value=0), \
                 patch.object(export.build.fifth.four.fork.base, "free_bytes",
                              side_effect=[export.FLOOR, export.FLOOR]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    export.capacity(root / "new", prospective=True)


if __name__ == "__main__":
    unittest.main()
