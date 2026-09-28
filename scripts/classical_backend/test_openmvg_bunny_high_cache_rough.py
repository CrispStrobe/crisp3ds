"""Synthetic-only guards for the sealed bunny rough cache continuation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from scripts.classical_backend import openmvg_bunny_high_cache_rough as control


class BunnyCacheRoughTest(unittest.TestCase):
    def test_commands_preserve_source_dense_settings_and_rough_scope(self):
        stages = control.commands(Path("/synthetic"),
                                  {"DensifyPointCloud": "/bin/dense",
                                   "ReconstructMesh": "/bin/mesh"})
        self.assertEqual([name for name, _, _ in stages], ["densify", "mesh"])
        dense = stages[0][1]
        self.assertEqual(dense[dense.index("--resolution-level") + 1], "3")
        self.assertEqual(dense[dense.index("--geometric-iters") + 1], "2")
        self.assertEqual(dense[dense.index("--min-resolution") + 1], "640")
        self.assertEqual(dense[dense.index("--max-resolution") + 1], "1280")

    def test_early_cache_checkpoint_requires_unchanged_files_and_new_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cached = output / "depth0000.dmap"
            cached.write_bytes(b"cached")
            rows = {cached.name: {"sha256": control.sha(cached)}}
            stats = {cached.name: cached.stat()}
            self.assertFalse(control.cache_reuse_checkpoint(output, stats, rows, 100, 150))
            with self.assertRaisesRegex(RuntimeError, "90 real seconds"):
                control.cache_reuse_checkpoint(output, stats, rows, 100, 191)
            (output / "depth0046.dmap").write_bytes(b"new")
            self.assertTrue(control.cache_reuse_checkpoint(output, stats, rows, 100, 150))
            rows[cached.name]["sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                control.cache_reuse_checkpoint(output, stats, rows, 100, 150)
            rows[cached.name]["sha256"] = control.sha(cached)
            cached.write_bytes(b"recomputed")
            with self.assertRaisesRegex(RuntimeError, "rewritten"):
                control.cache_reuse_checkpoint(output, stats, rows, 100, 150)

    def test_existing_output_blocks_before_source_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "exists"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                control.preflight(output)

    def test_ps_elapsed_parser_catches_sleep_spanning_day(self):
        with patch.object(control.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "1-02:03:04\n", "")):
            self.assertEqual(control.elapsed_ps_seconds(123), 93784)
        with patch.object(control.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "05:06\n", "")):
            self.assertEqual(control.elapsed_ps_seconds(123), 306)

    def test_failed_run_persists_capacity_and_source_postcheck_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fresh"
            checked = {"schema": "openmvg_bunny_high_cache_rough_v1",
                       "status": "read_only_preflight", "output": str(output),
                       "sealed_source": {"images": {}, "dmaps": {}}}
            with patch.object(control, "preflight", return_value=checked), \
                 patch.object(control.shutil, "copy2", side_effect=RuntimeError("synthetic copy stop")), \
                 patch.object(control, "capacity", side_effect=RuntimeError("synthetic floor breach")), \
                 patch.object(control, "verify_source", side_effect=RuntimeError("synthetic seal error")):
                result = control.run(output)
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(saved["failure"], "synthetic copy stop")
            self.assertEqual(saved["capacity_after_error"], "synthetic floor breach")
            self.assertEqual(saved["source_postcheck_error"], "synthetic seal error")
            self.assertFalse(saved["source_unchanged"])


if __name__ == "__main__":
    unittest.main()
