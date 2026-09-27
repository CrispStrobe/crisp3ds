"""Limits and gate checks for the independent level-1 native fallback."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import fresh_masked_level1 as target


class FreshMaskedLevel1Tests(unittest.TestCase):
    def test_level1_native_size_and_two_generation_budget(self):
        sizes = {name: (1282, 1024) for name in target.base.EXPECTED_NAMES}
        profile = target.depth_preflight(sizes, 120 << 20)
        self.assertEqual(profile["requested_level"], 1)
        self.assertEqual(profile["hard_output_cap_bytes"], 3 << 29)
        self.assertEqual({row["actual_level"] for row in profile["images"].values()}, {1})
        self.assertEqual({tuple(row["pixel_budget_size"])
                          for row in profile["images"].values()}, {(641, 512)})
        self.assertLess(profile["predicted_two_generation_peak_bytes"], target.CAP)
        with self.assertRaisesRegex(ValueError, "peak exceeds"):
            target.depth_preflight(sizes, 700 << 20)

    def test_minimum_cannot_silently_restore_level0(self):
        sizes = {name: (1198, 958) for name in target.base.EXPECTED_NAMES}
        with self.assertRaisesRegex(ValueError, "undo requested level-1"):
            target.depth_preflight(sizes, 0)

    def test_reviewed_source_gate_precedes_any_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "fresh-level1"
            args = argparse.Namespace(
                output=output, model=root / "model", images=root / "images",
                pose_masks=root / "masks", manifest=root / "manifest.json",
                source_report=root / "producer.json", camera_gate=root / "gate.json",
                source_report_sha256="a" * 64, camera_gate_sha256="b" * 64,
                model_sha256=["c" * 64] * 3, binary_dir=root / "bin")
            with patch.object(target.base, "input_binding",
                              side_effect=ValueError("camera gate rejected")) as gate:
                with self.assertRaisesRegex(ValueError, "camera gate rejected"):
                    target.run(args)
            gate.assert_called_once()
            self.assertFalse(output.exists())

    def test_frozen_limits_and_distinct_profile(self):
        self.assertEqual(target.PROFILE, "fresh_masked_level1_bounded_10m")
        self.assertEqual(target.CAP, 3 << 29)
        self.assertEqual(target.RSS_CAP, 6 << 30)
        self.assertEqual(target.TIMEOUT_SECONDS, 600)
        self.assertEqual((target.LEVEL, target.MIN_RESOLUTION,
                          target.MAX_RESOLUTION, target.GEOMETRIC_ITERS), (1, 600, 1600, 2))


if __name__ == "__main__":
    unittest.main()
