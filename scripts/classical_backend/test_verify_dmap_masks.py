import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.classical_backend import verify_dmap_masks
from scripts.classical_backend.run import ROOT


class DmapTests(unittest.TestCase):
    def test_reads_v240_depth_payload_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as temp:
            path = Path(temp) / "depth0001.dmap"
            name = b"dense/images/photo.jpg"
            header = verify_dmap_masks.HEADER.pack(0x5244, 1, 0, 2, 2, 2, 2, 0.1, 5.0)
            payload = (header + struct.pack("<H", len(name)) + name +
                       struct.pack("<II", 1, 1) + b"\0" * (21 * 8) +
                       np.array([1, 0, 2, 0], dtype="<f4").tobytes())
            path.write_bytes(payload)
            image, depth = verify_dmap_masks.load_depth(path)
            self.assertEqual(image, "dense/images/photo.jpg")
            self.assertEqual(depth.shape, (2, 2))
            self.assertEqual(int(np.count_nonzero(depth)), 2)
            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                verify_dmap_masks.load_depth(path)


if __name__ == "__main__":
    unittest.main()
