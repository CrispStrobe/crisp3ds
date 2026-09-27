"""Source-free runner isolation checks; no OpenMVS binary is invoked."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.classical_backend import openmvs_fusion_execute as executor
from scripts.classical_backend import openmvs_fusion_plan as planner
from scripts.classical_backend.run import ROOT, RESERVE, digest, tool_path


class FusionIsolationTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".local-tools/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.binary_dir = self.root / "bin"
        (self.source / "dense/images").mkdir(parents=True)
        (self.source / "masks").mkdir()
        self.binary_dir.mkdir()
        binary = tool_path(self.binary_dir, "DensifyPointCloud")
        binary.write_bytes(b"not-an-executable; mocked stage")
        (self.source / "scene.mvs").write_bytes(b"arm-relative dense/images paths")
        for i in range(60):
            stem = f"NP3_{i:03d}"
            (self.source / "dense/images" / f"{stem}.jpg").write_bytes(f"image{i}".encode())
            (self.source / "masks" / f"{stem}.mask.png").write_bytes(f"mask{i}".encode())
            (self.source / f"depth{i+1:04d}.dmap").write_bytes(f"depth{i}".encode())
        producer = {"schema": "classical_masked_dense_v1", "status": "complete",
                    "mask_count": 60, "native_options": planner.EXPECTED_OPTIONS,
                    "binary_hashes": {"DensifyPointCloud": digest(binary)},
                    "stages": [{"name": "densify", "status": "complete"}]}
        result = self.source / "result.json"
        result.write_text(json.dumps(producer))
        self.result_sha256 = digest(result)
        self.control = self.root / "control"
        self.ablation = self.root / "ablation"
        self.plan = planner.build_plan(self.source, self.control, self.ablation,
                                       self.binary_dir,
                                       expected_result_sha256=self.result_sha256,
                                       free_bytes=RESERVE + 2 * planner.ARM_CAP + planner.BUFFER)

    @staticmethod
    def _fake_stage(*, mutate_map):
        def run(output, name, command, deadline, max_bytes, max_log, max_rss):
            if mutate_map:
                (output / "depth0001.dmap").write_bytes(b"native changed copied depth")
            (output / "dense.mvs").write_bytes(b"dense scene")
            (output / "dense.ply").write_bytes(b"mock point cloud")
            return {"name": name, "status": "complete", "command": command}
        return run

    def _arm(self, label, output, *, mutate_map):
        with patch.object(executor, "stage", side_effect=self._fake_stage(mutate_map=mutate_map)), \
             patch.object(executor, "checked_ply", return_value=123), \
             patch.object(executor.shutil, "disk_usage", return_value=SimpleNamespace(free=RESERVE + 10**10)):
            return executor._run_arm(label, output, self.plan["commands"][label],
                                     self.plan, deadline=float("inf"))

    def test_native_success_mutating_copy_fails_without_mutating_source(self):
        original = digest(self.source / "depth0001.dmap")
        result = self._arm("control_dense_fuse", self.control, mutate_map=True)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["source_inputs_unchanged"])
        self.assertFalse(result["copied_inputs_unchanged"])
        self.assertIn("depth0001.dmap", result["copied_postcheck_failure"])
        self.assertEqual(digest(self.source / "depth0001.dmap"), original)
        self.assertNotEqual(digest(self.control / "depth0001.dmap"), original)
        self.assertEqual(json.loads((self.control / "fusion-result.json").read_text())["status"], "failed")

    def test_clean_arm_succeeds_and_planned_commands_only_change_filter(self):
        result = self._arm("control_dense_fuse", self.control, mutate_map=False)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["source_inputs_unchanged"])
        self.assertTrue(result["copied_inputs_unchanged"])
        self.assertEqual(result["stages"][0]["artifact"]["points"], 123)
        self.assertEqual(self.plan["sole_option_difference"], {"--fusion-filter": [2, 1]})
        commands = self.plan["commands"]
        left = [arg.replace(str(self.control), "<ARM>") for arg in commands["control_dense_fuse"]]
        right = [arg.replace(str(self.ablation), "<ARM>") for arg in commands["ablation_simple_fuse"]]
        differences = [(a, b) for a, b in zip(left, right) if a != b]
        self.assertEqual(differences, [("2", "1")])
        self.assertEqual(digest(self.control / "depth0001.dmap"),
                         digest(self.source / "depth0001.dmap"))


if __name__ == "__main__":
    unittest.main()
