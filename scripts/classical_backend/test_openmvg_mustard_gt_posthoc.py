"""Synthetic-only tests; never opens Berkeley H5 datasets or runs GT scoring."""
from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_gt_posthoc as posthoc
from scripts.classical_backend.test_openmvg_mustard_json_pose_validator import fixture


class PosthocGateTest(unittest.TestCase):
    def test_id_bound_common_46_names_not_array_order(self):
        data = fixture()
        data["views"].reverse()
        data["extrinsics"].reverse()
        centers, rotations = posthoc.named_cameras(data)
        self.assertEqual(set(centers), set(posthoc.EXPECTED_NAMES))
        self.assertEqual(set(rotations), set(posthoc.EXPECTED_NAMES))
        self.assertNotIn("NP3_042.jpg", centers)
        self.assertNotIn("NP3_048.jpg", centers)

    def test_exact_archive_inventory_and_staged_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "tiny.tgz"
            members = {posthoc.PREFIX + f"poses/NP5_{angle}_pose.h5"
                       for angle in (0, 300, 306, 312, 318, 330, 336, 342, 348)}
            with tarfile.open(archive, "w:gz") as stream:
                for name in sorted(members):
                    payload = b"x" * 2296
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    stream.addfile(info, io.BytesIO(payload))
            self.assertEqual(set(posthoc.archive_inventory(archive, members)), members)
            output = root / "fresh"
            (output / "metadata").mkdir(parents=True)
            with patch.object(posthoc, "ARCHIVE", archive), \
                 patch.object(posthoc, "capacity", return_value={}):
                staged = posthoc.stage_missing(output, posthoc.archive_inventory(archive, members))
            self.assertEqual(len(staged), 9)
            posthoc.verify_staged(output, staged)
            (output / "metadata/NP5_0_pose.h5").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "staged NP5 metadata hash differs"):
                posthoc.verify_staged(output, staged)
            with self.assertRaisesRegex(ValueError, "lacks common-46"):
                posthoc.archive_inventory(archive, members | {posthoc.PREFIX + "poses/NP5_54_pose.h5"})

    def test_fresh_output_and_11gib_plus_output_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh reviewed external root"):
                posthoc.preflight(output)
            with patch.object(posthoc.shutil, "disk_usage",
                              side_effect=[type("Usage", (), {"free": posthoc.FLOOR})(),
                                           type("Usage", (), {"free": posthoc.FLOOR})()]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    posthoc.capacity(Path(temporary) / "new", prospective=True)

    def test_shared_sim3_source_is_pinned(self):
        with patch.object(posthoc, "COMPARE_SOURCE_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "source seal differs"):
                posthoc.algorithm_runtime()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "shared evaluator needs numpy")
    def test_two_proper_sim3_fits_and_signed_paired_deltas(self):
        centers, rotations, target, target_rotations = {}, {}, {}, {}
        for name in posthoc.EXPECTED_NAMES:
            angle = math.radians(int(name[4:7]))
            x, y, z = math.cos(angle), math.sin(angle), 0.1 * math.sin(2 * angle)
            centers[name] = [x, y, z]
            rotations[name] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            target[name] = [2 * x + 3, 2 * y - 4, 2 * z + 1]
            target_rotations[name] = rotations[name]
        fixed = {"centers": {name: list(value) for name, value in centers.items()},
                 "rotations": rotations}
        fixed["centers"][posthoc.EXPECTED_NAMES[0]][0] += 0.2
        result = posthoc.compare_arms({"centers": centers, "rotations": rotations},
                                      fixed, target, target_rotations)
        self.assertEqual(result["arms"]["adjust_all"]["matched_count"], 46)
        self.assertLess(result["arms"]["adjust_all"]["center_residual_over_supplied_median_radius"]["max"], 1e-8)
        self.assertGreater(result["arms"]["fixed_none"]["center_residual_over_supplied_median_radius"]["max"], 0)
        self.assertIn("scale_berkeley_pose_units_per_openmvg_unit", result["arms"]["fixed_none"]["fit"])
        self.assertNotIn("scale_supplied_units_per_square", result["arms"]["fixed_none"]["fit"])
        self.assertEqual(len(result["paired_per_frame_deltas"]), 46)


if __name__ == "__main__":
    unittest.main()
