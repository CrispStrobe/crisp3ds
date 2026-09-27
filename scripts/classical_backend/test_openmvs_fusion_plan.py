from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.classical_backend.openmvs_fusion_plan import (
    ARM_CAP, BUFFER, EXPECTED_OPTIONS, build_plan,
)
from scripts.classical_backend.run import RESERVE, digest, tool_path


class FusionPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.binary_dir = self.root / "bin"
        (self.source / "dense/images").mkdir(parents=True)
        (self.source / "masks").mkdir()
        self.binary_dir.mkdir()
        self.binary = tool_path(self.binary_dir, "DensifyPointCloud")
        self.binary.write_bytes(b"test binary")
        (self.source / "scene.mvs").write_bytes(b"scene")
        for i in range(60):
            stem = f"NP3_{i:03d}"
            (self.source / "dense/images" / f"{stem}.jpg").write_bytes(b"image" + bytes([i]))
            (self.source / "masks" / f"{stem}.mask.png").write_bytes(b"mask" + bytes([i]))
            (self.source / f"depth{i+1:04d}.dmap").write_bytes(b"dmap" + bytes([i]))
        self.report = {
            "schema": "classical_masked_dense_v1", "status": "complete", "mask_count": 60,
            "native_options": EXPECTED_OPTIONS,
            "binary_hashes": {"DensifyPointCloud": digest(self.binary)},
            "stages": [{"name": "densify", "status": "complete"}],
        }
        self._save_report()

    def _save_report(self):
        self.result = self.source / "result.json"
        self.result.write_text(json.dumps(self.report))
        self.result_hash = digest(self.result)

    def _plan(self, **kwargs):
        return build_plan(self.source, self.root / "control", self.root / "ablation",
                          self.binary_dir, expected_result_sha256=self.result_hash,
                          free_bytes=RESERVE + 2 * ARM_CAP + BUFFER, **kwargs)

    def test_same_cached_inputs_and_only_fusion_filter_differs(self):
        plan = self._plan()
        self.assertEqual(plan["execution"], "not_authorized_or_run")
        self.assertEqual(len(plan["input_sha256"]), 181)
        self.assertEqual(plan["sole_option_difference"], {"--fusion-filter": [2, 1]})
        for command in plan["commands"].values():
            self.assertEqual(command[command.index("--geometric-iters") + 1], "0")
            self.assertEqual(command[command.index("--max-threads") + 1], "2")
            self.assertEqual(Path(command[command.index("--mask-path") + 1]).name, "masks")

    def test_rejects_tampered_result_or_binary(self):
        with self.assertRaisesRegex(ValueError, "reviewed 008 evidence"):
            build_plan(self.source, self.root / "control", self.root / "ablation",
                       self.binary_dir, expected_result_sha256="0" * 64,
                       free_bytes=RESERVE + 2 * ARM_CAP + BUFFER)
        self.binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "native binary differs"):
            self._plan()

    def test_rejects_missing_depth_map_and_tampered_mask_name(self):
        (self.source / "depth0060.dmap").unlink()
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            self._plan()
        (self.source / "depth0060.dmap").write_bytes(b"restored")
        (self.source / "masks/NP3_059.mask.png").rename(self.source / "masks/bad.mask.png")
        with self.assertRaisesRegex(ValueError, "mask/image names"):
            self._plan()

    def test_rejects_linked_input_and_insufficient_disk(self):
        target = self.source / "dense/images/NP3_000.jpg"
        target.unlink()
        target.symlink_to(self.source / "dense/images/NP3_001.jpg")
        with self.assertRaisesRegex(ValueError, "linked source file"):
            self._plan()
        target.unlink()
        target.write_bytes(b"image")
        with self.assertRaisesRegex(ValueError, "insufficient free space"):
            build_plan(self.source, self.root / "control", self.root / "ablation",
                       self.binary_dir, expected_result_sha256=self.result_hash,
                       free_bytes=RESERVE + 2 * ARM_CAP + BUFFER - 1)

    def test_rejects_existing_output(self):
        (self.root / "control").mkdir()
        with self.assertRaisesRegex(ValueError, "output must be fresh"):
            self._plan()


if __name__ == "__main__":
    unittest.main()
