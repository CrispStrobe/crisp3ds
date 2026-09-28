"""Synthetic fail-closed tests for the post-hoc rough rescue receipt."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_bunny_high_rough_rescue as rescue


class RoughRescueTest(unittest.TestCase):
    def test_requires_exact_original_failure_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "result.json").write_text(json.dumps({
                "schema": "openmvg_bunny_high_cached73_fusion_rough_v1",
                "status": "failed", "failure": "native crash", "source_unchanged": True,
                "real_wall_seconds_total": 30, "output_bytes": 100}))
            with patch.object(rescue, "SOURCE", source):
                with self.assertRaisesRegex(ValueError, "exact post-native"):
                    rescue.validate(source)

    def test_mismatched_source_path_blocks_before_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with self.assertRaisesRegex(ValueError, "unexpected rough source"):
                rescue.validate(source)


if __name__ == "__main__":
    unittest.main()
