"""Synthetic-only fail-closed tests for cached-73 OpenMVS rough fusion."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvg_bunny_high_cache_fusion as control


class BunnyCachedFusionTest(unittest.TestCase):
    def test_commands_disable_geometric_pass_and_stop_at_rough_mesh(self):
        stages = control.commands(Path("/synthetic"),
                                  {"DensifyPointCloud": "/bin/dense",
                                   "ReconstructMesh": "/bin/mesh"})
        self.assertEqual([name for name, _, _ in stages], ["densify", "mesh"])
        dense = stages[0][1]
        self.assertEqual(dense[dense.index("--geometric-iters") + 1], "0")
        self.assertEqual(dense[dense.index("--resolution-level") + 1], "3")

    def test_fusion_checkpoint_needs_native_fusion_without_reestimation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depth = root / "depth0000.dmap"
            depth.write_bytes(b"cached")
            (root / "densify.log").write_bytes(b"initializing\n")
            initial = {depth.name: (depth.stat().st_size, depth.stat().st_mtime_ns)}
            self.assertFalse(control.cache_fusion_checkpoint(root, initial, 110, 100))
            with self.assertRaisesRegex(RuntimeError, "90 real seconds"):
                control.cache_fusion_checkpoint(root, initial, 191, 100)
            (root / "densify.log").write_bytes(b"Dense fused depth-maps")
            self.assertTrue(control.cache_fusion_checkpoint(root, initial, 110, 100))
            (root / "densify.log").write_bytes(b"Estimated depth-maps\nDense fused depth-maps")
            with self.assertRaisesRegex(RuntimeError, "estimated depth maps"):
                control.cache_fusion_checkpoint(root, initial, 110, 100)
            (root / "densify.log").write_bytes(b"Dense fused depth-maps")
            depth.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "rewritten"):
                control.cache_fusion_checkpoint(root, initial, 110, 100)

    def test_existing_output_blocks_before_source_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "exists"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                control.preflight(output)

    def test_clone_detects_physical_copy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.write_bytes(b"synthetic data")
            free = 20 << 30

            def copy_fallback(args, **kwargs):
                shutil.copyfile(args[-2], args[-1])

            usage = shutil._ntuple_diskusage(100 << 30, 1 << 30, free)
            usage_after = shutil._ntuple_diskusage(100 << 30, 1 << 30, free - (8 << 20))
            with patch.object(control.subprocess, "run", side_effect=copy_fallback), \
                 patch.object(control.shutil, "disk_usage", side_effect=[usage, usage_after]):
                with self.assertRaisesRegex(RuntimeError, "physical-copy fallback"):
                    control.clone_one(source, target,
                                      {"bytes": source.stat().st_size,
                                       "sha256": control.sha(source)}, free)

    def test_failure_receipt_preserves_capacity_and_source_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            checked = {"schema": "openmvg_bunny_high_cached73_fusion_rough_v1",
                       "output": str(output),
                       "sealed_source": {"images": {}, "dmaps": {}}}
            with patch.object(control, "preflight", return_value=checked), \
                 patch.object(control, "clone_one", side_effect=RuntimeError("synthetic COW stop")), \
                 patch.object(control, "capacity", side_effect=RuntimeError("synthetic floor breach")), \
                 patch.object(control, "verify_source", side_effect=RuntimeError("synthetic seal error")):
                result = control.run(output)
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(saved["failure"], "synthetic COW stop")
            self.assertEqual(saved["capacity_after_error"], "synthetic floor breach")
            self.assertEqual(saved["source_postcheck_error"], "synthetic seal error")
            self.assertFalse(saved["source_unchanged"])


if __name__ == "__main__":
    unittest.main()
