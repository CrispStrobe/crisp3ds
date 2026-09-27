"""Synthetic-only bare-mesh and neutral-preview producer checks."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.classical_backend import cracker_visual_hull as hull
from scripts.classical_backend.cracker_visual_hull import write_neutral_preview, write_ply
from scripts.classical_backend.visual_hull_core import exposed_cube_mesh
from scripts.object_dataset import evaluate


class CrackerVisualHullSyntheticTest(unittest.TestCase):
    def test_structured_empty_and_boundary_abstentions(self):
        empty = np.zeros((5, 5, 5), bool)
        self.assertEqual(hull.abstention_state(empty),
                         {"occupied_voxels": 0, "boundary_touched": False, "abstention_reason": "empty"})
        empty[0, 2, 2] = True
        self.assertEqual(hull.abstention_state(empty),
                         {"occupied_voxels": 1, "boundary_touched": True,
                          "abstention_reason": "outer_grid_touch"})
        empty[0, 2, 2] = False
        empty[2, 2, 2] = True
        self.assertIsNone(hull.abstention_state(empty)["abstention_reason"])

    def test_incomplete_or_altered_source_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            (source / "masks").mkdir(parents=True)
            result = {"schema": "classical_fresh_masked_complete_v1", "status": "complete",
                      "sources_unchanged": True}
            report = {"schema": "classical_dense_masks_v1", "status": "complete",
                      "semantics": "photo-derived coarse pose support, not a silhouette or ground truth",
                      "ignore_mask_label": 0, "undistorted_model_sha256": hull.MODEL_SHA256,
                      "images": []}
            result_path, report_path = source / "result.json", source / "masks" / "report.json"
            result_path.write_text(json.dumps(result))
            report_path.write_text(json.dumps(report))
            result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
            report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
            output = Path(temporary) / "output"
            with patch.object(hull, "SOURCE", source), patch.object(hull, "RESULT_SHA256", result_sha), \
                    patch.object(hull, "MASK_REPORT_SHA256", report_sha), \
                    patch.object(hull, "DATA", output.parent), patch.object(hull, "OUTPUT", output), \
                    patch.object(hull, "disk_floor", return_value={}), \
                    patch.object(sys, "argv", ["hull", "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "incomplete or malformed sealed 60-view"):
                    hull.main()
                self.assertFalse(output.exists())
            with patch.object(hull, "SOURCE", source), patch.object(hull, "RESULT_SHA256", "0" * 64), \
                    patch.object(hull, "DATA", output.parent), patch.object(hull, "OUTPUT", output), \
                    patch.object(hull, "disk_floor", return_value={}), \
                    patch.object(sys, "argv", ["hull", "--output", str(output)]):
                with self.assertRaisesRegex(ValueError, "sealed JSON differs"):
                    hull.main()
                self.assertFalse(output.exists())

    def test_bare_mesh_roundtrip_and_candidate_only_preview(self):
        occupancy = np.zeros((5, 5, 5), bool)
        occupancy[2, 2, 2] = True
        vertices, faces = exposed_cube_mesh(occupancy, np.zeros(3), 5)
        with tempfile.TemporaryDirectory() as temporary:
            mesh = Path(temporary) / "hull.ply"
            preview = Path(temporary) / "neutral.png"
            write_ply(mesh, vertices, faces)
            info, read_vertices, read_faces = evaluate.inspect_ply(mesh, geometry=True)
            self.assertEqual(info["nontriangle_faces"], 0)
            np.testing.assert_allclose(read_vertices, vertices)
            np.testing.assert_array_equal(read_faces, faces)
            write_neutral_preview(preview, vertices, faces, np.zeros(3), 5)
            with Image.open(preview) as image:
                self.assertEqual(image.size, (768, 256))
                self.assertEqual(image.mode, "RGB")
                self.assertNotEqual(image.getpixel((128, 128)), (255, 255, 255))
            with self.assertRaises(FileExistsError):
                write_ply(mesh, vertices, faces)


if __name__ == "__main__":
    unittest.main()
