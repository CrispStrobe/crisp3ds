"""Read-only Berkeley evidence and synthetic partial-orbit camera checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.object_motion import stock_camera_compare as diagnostic


class StockCameraCompareTests(unittest.TestCase):
    def synthetic_evidence(self, root):
        metadata = root / "metadata"
        members = {}
        for member in sorted(diagnostic.rig.MEMBERS):
            path = metadata / member.removeprefix(diagnostic.rig.PREFIX)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(member.encode())
            members[member] = {"bytes": path.stat().st_size,
                               "sha256": diagnostic.colmap.sha256(path)}
        report = root / "report.json"
        report.write_text(json.dumps({
            "schema": "ycb_berkeley_camera_oracle_v1",
            "source_archive_sha256": diagnostic.rig.ARCHIVE_SHA256,
            "google_mesh_cross_frame_transform_known": False,
            "metadata": {"members": members,
                         "total_uncompressed_bytes": sum(row["bytes"] for row in members.values())},
        }))
        return report, metadata, diagnostic.colmap.sha256(report)

    def test_synthetic_metadata_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            report, metadata, digest = self.synthetic_evidence(Path(temp))
            evidence = diagnostic.verified_metadata(report, metadata, digest)
            self.assertEqual(len(evidence["metadata_sha256"]), 61)
            self.assertEqual(evidence["report_sha256"], digest)
            first = metadata / "calibration.h5"
            first.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
                diagnostic.verified_metadata(report, metadata, digest)

    def test_partial_named_orbit_reports_coverage_and_orientation(self):
        names = [f"NP3_{angle:03}.jpg" for angle in range(0, 360, 6)]
        angle = np.deg2rad(27)
        c, s = np.cos(angle), np.sin(angle)
        world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        target_centers = {name: np.array([np.cos(np.deg2rad(i * 6)),
                                           np.sin(np.deg2rad(i * 6)), .01 * i])
                          for i, name in enumerate(names)}
        target_rotations = {name: np.eye(3) for name in names}
        selected = names[::10]
        source = {name: (world.T @ (target_centers[name] - [2., -3., 1.]) / 2,
                         world)
                  for name in selected}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "diagnostic.json"
            with patch.object(diagnostic.colmap, "read_candidate", return_value=(source, {}, {})), \
                 patch.object(diagnostic, "verified_metadata", return_value={"fixture": True}), \
                 patch.object(diagnostic.rig, "reference_cameras",
                              return_value=(target_centers, target_rotations, None, None)):
                result = diagnostic.evaluate(Path(temp), output)
            self.assertEqual(result["coverage"]["registered"], 6)
            self.assertEqual(len(result["coverage"]["missing_names"]), 54)
            self.assertLess(result["comparison"]["center_rms_over_reference_radius"], 1e-10)
            self.assertLess(result["comparison"]["orientation_p95_degrees"], 1e-4)
            self.assertFalse(result["metric_mesh_accuracy_claim_allowed"])
            self.assertEqual(json.loads(output.read_text())["schema"], result["schema"])

    def test_short_and_foreign_models_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "diagnostic.json"
            pose = (np.zeros(3), np.eye(3))
            for names in (["NP3_000.jpg", "NP3_006.jpg", "NP3_012.jpg"],
                          ["NP3_000.jpg", "NP3_006.jpg", "NP3_012.jpg", "other.jpg"]):
                with patch.object(diagnostic.colmap, "read_candidate",
                                  return_value=({name: pose for name in names}, {}, {})), \
                     patch.object(diagnostic, "verified_metadata", return_value={"fixture": True}):
                    with self.assertRaisesRegex(ValueError, "4–60 distinct selected"):
                        diagnostic.evaluate(Path(temp), output)
                self.assertFalse(output.exists())

    def test_reference_report_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            altered, metadata, digest = self.synthetic_evidence(Path(temp))
            altered.write_bytes(altered.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "report hash mismatch"):
                diagnostic.verified_metadata(altered, metadata, digest)


if __name__ == "__main__":
    unittest.main()
