"""Synthetic-only three-arm 44-name posthoc gates; never opens real GT H5."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_mustard_three_arm_gt_posthoc as three
from scripts.classical_backend.test_openmvg_mustard_json_pose_validator import fixture


class ThreeArmPosthocTest(unittest.TestCase):
    def test_global_id_bound_linkage_and_exact_44_intersection(self):
        data = fixture()
        data["extrinsics"] = []
        for row in data["views"]:
            view = row["value"]["ptr_wrapper"]["data"]
            if view["filename"] in ("NP3_252.jpg", "NP3_276.jpg"):
                continue
            angle = math.radians(int(view["filename"][4:7]))
            data["extrinsics"].append({"key": view["id_pose"], "value": {
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center": [math.cos(angle), math.sin(angle), 0.2]}})
        data["extrinsics"].reverse()
        centers, rotations = three.named_global(data)
        self.assertEqual(len(centers), 46)
        self.assertEqual(set(centers) & set(three.two.EXPECTED_NAMES), set(three.COMMON))
        self.assertEqual(set(centers), set(rotations))
        self.assertNotIn("NP3_252.jpg", centers)
        self.assertNotIn("NP3_276.jpg", centers)

    def test_fresh_output_and_11gib_plus_one_mib_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "already"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh reviewed external root"):
                three.preflight(existing)
            with patch.object(three.shutil, "disk_usage",
                              side_effect=[type("Usage", (), {"free": three.FLOOR})(),
                                           type("Usage", (), {"free": three.FLOOR})()]):
                with self.assertRaisesRegex(RuntimeError, "disk/output cap"):
                    three.capacity(Path(temporary) / "new", prospective=True)

    def test_prior_metadata_receipt_and_staged_hashes_are_bound_without_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = root / "prior"
            metadata = root / "source/metadata"
            prior_meta = prior / "metadata"
            metadata.mkdir(parents=True)
            prior_meta.mkdir(parents=True)
            required = {three.two.PREFIX + "calibration.h5"} | {
                three.two.PREFIX + f"poses/NP5_{int(name[4:7])}_pose.h5"
                for name in three.two.EXPECTED_NAMES}
            staged_names = {three.two.PREFIX + f"poses/NP5_{angle}_pose.h5"
                            for angle in (0, 300, 306, 312, 318, 330, 336, 342, 348)}
            extracted_names = required - staged_names
            self.assertEqual((len(required), len(extracted_names), len(staged_names)), (47, 38, 9))
            found = {}
            staged = {}
            for member in sorted(extracted_names):
                path = metadata / Path(member).relative_to(three.two.PREFIX)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(member.encode())
                found[member] = {"bytes": path.stat().st_size, "sha256": three.two.sha(path)}
            for angle in (42, 48):
                member = three.two.PREFIX + f"poses/NP5_{angle}_pose.h5"
                path = metadata / Path(member).relative_to(three.two.PREFIX)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(member.encode())
                found[member] = {"bytes": path.stat().st_size, "sha256": three.two.sha(path)}
            for member in sorted(staged_names):
                path = prior_meta / Path(member).name
                path.write_bytes(member.encode())
                staged[member] = {"bytes": path.stat().st_size, "sha256": three.two.sha(path)}
            self.assertEqual(len(found), 40)
            metadata_receipt = root / "source/receipt.json"
            metadata_receipt.write_text(json.dumps({"schema": "mustard_pose_metadata_acquisition_v1",
                                                    "status": "metadata_only_no_camera_evaluation",
                                                    "extraction": {"members": found}}))
            report = prior / "report.json"
            report.write_text(json.dumps({"schema": "openmvg_mustard_two_arm_gt_posthoc_result_v1",
                                          "status": "posthoc_46_view_diagnostic_only"}))
            report_sha = three.two.sha(report)
            inventory = {"report.json": {"bytes": report.stat().st_size, "sha256": report_sha}}
            inventory.update({"metadata/" + Path(member).name: record for member, record in staged.items()})
            prior_receipt = prior / "receipt.json"
            prior_receipt.write_text(json.dumps({
                "schema": "openmvg_mustard_two_arm_gt_posthoc_v1",
                "status": "posthoc_46_view_diagnostic_only", "report_sha256": report_sha,
                "report_bytes": report.stat().st_size,
                "names": {"common_names": list(three.two.EXPECTED_NAMES)},
                "adjust_all": {"json_sha256": three.poses.APPROVED_JSON_SHA256},
                "fixed_none": {"json_sha256": three.two.FIXED_EXPORT_JSON_SHA},
                "metadata": {"receipt_sha256": three.two.sha(metadata_receipt),
                             "archive_sha256": three.two.ARCHIVE_SHA,
                             "extracted_members": {name: found[name] for name in extracted_names},
                             "archive_only_members": {name: {"bytes": 2296} for name in staged_names}},
                "staged_metadata": staged, "artifact_inventory": inventory}))
            with patch.object(three, "PRIOR", prior), \
                 patch.object(three, "PRIOR_RECEIPT_SHA", three.two.sha(prior_receipt)), \
                 patch.object(three, "PRIOR_REPORT_SHA", report_sha), \
                 patch.object(three.two, "METADATA", metadata), \
                 patch.object(three.two, "METADATA_RECEIPT", metadata_receipt), \
                 patch.object(three.two, "METADATA_RECEIPT_SHA", three.two.sha(metadata_receipt)):
                checked = three.audit_reference()
                self.assertEqual(checked["used_h5_count"], 45)
                (prior_meta / "NP5_0_pose.h5").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "Berkeley metadata seal differs|staged NP5 seal differs"):
                    three.audit_reference()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "shared Sim3 evaluator needs NumPy")
    def test_reference_reader_uses_reviewed_output_and_mocked_h5dump(self):
        import numpy as np
        from scripts.object_motion import ycb_camera_reference as ycb
        members = {three.two.PREFIX + "calibration.h5": {"path": "/synthetic/calibration.h5"}}
        members.update({three.two.PREFIX + f"poses/NP5_{int(name[4:7])}_pose.h5":
                        {"path": "/synthetic/pose.h5"} for name in three.COMMON})
        def fake_dataset(_path, key, _shape):
            if key == "/NP3_rgb_K":
                return np.diag([1000.0, 1000.0, 1.0])
            if key == "/NP3_rgb_d":
                return np.zeros(5)
            return np.eye(4)
        reviewed_output = Path("/synthetic/reviewed-output")
        with patch.object(ycb, "dataset", side_effect=fake_dataset) as h5dump, \
             patch.object(three, "capacity", return_value={}) as disk:
            centers, rotations, _ = three.reference_cameras(
                {"used_h5": members}, three.time.monotonic(), reviewed_output)
        self.assertEqual((len(centers), len(rotations)), (44, 44))
        self.assertEqual(h5dump.call_count, 47)  # 3 calibration datasets + 44 poses
        self.assertEqual(disk.call_count, 44)
        disk.assert_any_call(reviewed_output)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "shared Sim3 evaluator needs NumPy")
    def test_three_independent_44_fit_and_paired_deltas(self):
        centers, rotations, target, target_rotations = {}, {}, {}, {}
        for name in three.COMMON:
            angle = math.radians(int(name[4:7]))
            x, y, z = math.cos(angle), math.sin(angle), 0.1 * math.sin(2 * angle)
            centers[name] = [x, y, z]
            rotations[name] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            target[name] = [3 * x + 2, 3 * y - 1, 3 * z + 5]
            target_rotations[name] = rotations[name]
        arms = {label: {"centers": {name: list(value) for name, value in centers.items()},
                        "rotations": rotations} for label in
                ("adjust_all", "fixed_none", "global_adjust_all")}
        arms["fixed_none"]["centers"][three.COMMON[0]][0] += 0.1
        arms["global_adjust_all"]["centers"][three.COMMON[1]][0] += 0.2
        result = three.compare_three(arms, target, target_rotations)
        self.assertEqual({arm["matched_count"] for arm in result["arms"].values()}, {44})
        self.assertEqual(len(result["paired_per_frame_deltas"]), 3)
        self.assertTrue(all(len(rows) == 44 for rows in result["paired_per_frame_deltas"].values()))
        self.assertNotIn("scale_supplied_units_per_square", result["arms"]["global_adjust_all"]["fit"])
        self.assertLess(result["arms"]["adjust_all"]["center_residual_over_supplied_median_radius"]["max"], 1e-8)


if __name__ == "__main__":
    unittest.main()
