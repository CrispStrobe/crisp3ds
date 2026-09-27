from __future__ import annotations

from pathlib import Path
import json
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.classical_backend import openmvs_fusion_execute as target
from scripts.classical_backend.run import RESERVE, digest


class FusionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        disk = patch.object(target.shutil, "disk_usage",
                            return_value=types.SimpleNamespace(free=RESERVE + 4 * target.ARM_CAP))
        disk.start()
        self.addCleanup(disk.stop)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.binary = self.root / "native-tool"
        self.binary.write_bytes(b"tool")
        for name, data in (("scene.mvs", b"scene"),
                           ("dense/images/photo.jpg", b"photo"),
                           ("masks/photo.mask.png", b"mask"),
                           ("depth0001.dmap", b"depth"),
                           ("result.json", b"sealed report")):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.hashes = {name: digest(self.source / name) for name in
                       ("scene.mvs", "dense/images/photo.jpg", "masks/photo.mask.png",
                        "depth0001.dmap")}
        self.control = self.root / "control"
        self.ablation = self.root / "ablation"
        self.commands = {
            "control_dense_fuse": [str(self.binary), "-i", str(self.control / "scene.mvs"),
                                   "--fusion-filter", "2"],
            "ablation_simple_fuse": [str(self.binary), "-i", str(self.ablation / "scene.mvs"),
                                     "--fusion-filter", "1"],
        }
        self.plan = {"source": str(self.source), "source_result_sha256": digest(self.source / "result.json"),
                     "binary_sha256": digest(self.binary), "input_sha256": self.hashes,
                     "control": str(self.control), "ablation": str(self.ablation),
                     "commands": self.commands, "sole_option_difference": {"--fusion-filter": [2, 1]}}

    @staticmethod
    def _fake_stage(output, name, command, deadline, max_bytes, max_log_bytes, max_rss_bytes):
        (output / "dense.mvs").write_bytes(b"native scene")
        (output / "dense.ply").write_bytes(b"native cloud")
        return {"name": name, "status": "complete", "command": command, "seconds": 0.01}

    def test_pair_copies_independently_and_preserves_inputs(self):
        with (patch.object(target, "build_plan", return_value=self.plan),
              patch.object(target, "stage", side_effect=self._fake_stage),
              patch.object(target, "checked_ply", return_value=42)):
            report = target.execute_pair(self.source, self.control, self.ablation)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(json.loads((self.control / "fusion-pair.json").read_text())["status"], "complete")
        self.assertEqual(report["arms"]["control_dense_fuse"]["stages"][0]["artifact"]["points"], 42)
        for output in (self.control, self.ablation):
            self.assertTrue((output / "dense/images/photo.jpg").is_file())
            self.assertFalse((output / "dense/images/photo.jpg").is_symlink())
            self.assertNotEqual((output / "depth0001.dmap").stat().st_ino,
                                (self.source / "depth0001.dmap").stat().st_ino)
            target._verify_inputs(output, self.hashes)

    def test_mutated_cached_map_fails_after_native_stage(self):
        def mutating_stage(output, *args):
            result = self._fake_stage(output, *args)
            (output / "depth0001.dmap").write_bytes(b"changed")
            return result
        with (patch.object(target, "stage", side_effect=mutating_stage),
              patch.object(target, "checked_ply", return_value=42)):
            report = target._run_arm("control", self.control, self.commands["control_dense_fuse"],
                                     self.plan, target.time.monotonic() + 10)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["copied_inputs_unchanged"])
        self.assertTrue(report["source_inputs_unchanged"])

    def test_source_or_binary_tamper_rejected(self):
        self.binary.write_bytes(b"tampered")
        with (patch.object(target, "stage") as stage,
              patch.object(target, "checked_ply", return_value=42)):
            report = target._run_arm("control", self.control, self.commands["control_dense_fuse"],
                                     self.plan, target.time.monotonic() + 10)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_result_and_binary_unchanged"])
        self.assertFalse(self.control.exists())
        stage.assert_not_called()

    def test_config_override_rejected_before_native_stage(self):
        original_copy = target._copy_inputs
        def copy_with_cfg(*args):
            original_copy(*args)
            (args[1] / "DensifyPointCloud.cfg").write_text("fusion-filter=0")
        with (patch.object(target, "_copy_inputs", side_effect=copy_with_cfg),
              patch.object(target, "stage") as stage):
            report = target._run_arm("control", self.control, self.commands["control_dense_fuse"],
                                     self.plan, target.time.monotonic() + 10)
        self.assertEqual(report["status"], "failed")
        self.assertIn("config override", report["failure"])
        stage.assert_not_called()

    def test_binary_changed_during_copy_rejected_before_launch(self):
        original_copy = target._copy_inputs
        def copy_then_tamper(*args):
            original_copy(*args)
            self.binary.write_bytes(b"changed while copying")
        with (patch.object(target, "_copy_inputs", side_effect=copy_then_tamper),
              patch.object(target, "stage") as stage):
            report = target._run_arm("control", self.control, self.commands["control_dense_fuse"],
                                     self.plan, target.time.monotonic() + 10)
        self.assertEqual(report["status"], "failed")
        self.assertIn("changed before launch", report["failure"])
        stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
