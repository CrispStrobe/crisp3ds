from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.classical_backend.fresh005_prefusion_diagnostic import diagnose, inspect_map
from scripts.classical_backend.run import digest


class PrefusionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _map(self, path: Path, name: str = "NP3_000.jpg", confidence=(0.0, 1.0, 2.0, 3.0)):
        header = struct.pack("<HBBIIIIff", 0x5244, 7, 0, 2, 2, 2, 2, 1.0, 4.0)
        image = ("dense/images/" + name).encode()
        payload = (np.array([0, 1, 2, 3], dtype="<f4").tobytes() +
                   np.zeros((4, 3), dtype="<f4").tobytes() +
                   np.array(confidence, dtype="<f4").tobytes())
        path.write_bytes(header + struct.pack("<H", len(image)) + image +
                         struct.pack("<I", 1) + struct.pack("<I", 1) +
                         np.zeros(21, dtype="<f8").tobytes() + payload)

    def test_depth_and_confidence_offsets(self):
        path = self.root / "depth0001.dmap"
        self._map(path)
        result = inspect_map(path, "NP3_000.jpg", digest(path))
        self.assertEqual(result["positive_depth_pixels"], 3)
        self.assertEqual(result["positive_depth_zero_confidence"], 0)
        self.assertEqual(result["positive_depth_quantiles"][1], 2.0)
        self.assertEqual(result["positive_depth_confidence_quantiles"][1], 2.0)

    def test_rejects_wrong_channel_layout_and_hash(self):
        path = self.root / "depth0001.dmap"
        self._map(path)
        with self.assertRaisesRegex(ValueError, "differs from sealed"):
            inspect_map(path, "NP3_000.jpg", "0" * 64)
        data = bytearray(path.read_bytes())
        data[2] = 3
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "channels"):
            inspect_map(path, "NP3_000.jpg", digest(path))

    def test_report_seal_is_required_before_scanning(self):
        (self.root / "result.json").write_text(json.dumps({"status": "complete"}))
        (self.root / "dmap-camera-check.json").write_text("{}")
        (self.root / "dense.ply").write_text("x")
        (self.root / "masks").mkdir()
        (self.root / "masks/report.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "result seal differs"):
            diagnose(self.root)


if __name__ == "__main__":
    unittest.main()
