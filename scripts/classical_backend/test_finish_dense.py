import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from scripts.classical_backend import finish_dense
from scripts.classical_backend.run import ROOT


class FinishDenseTests(unittest.TestCase):
    def test_rejects_wrong_or_incomplete_source_without_creating_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as tmp:
            folder = Path(tmp)
            source, output = folder / "source", folder / "output"
            source.mkdir()
            (source / "result.json").write_text(json.dumps({"schema": "wrong", "status": "failed"}))
            with self.assertRaisesRegex(ValueError, "not a calibrated completed dense cloud"):
                finish_dense.finish(source, output)
            self.assertFalse(output.exists())
            (source / "result.json").write_text(json.dumps({
                "schema": "classical_calibrated_control_v1", "status": "failed",
                "registered_images": 60, "sources_unchanged": True, "stages": []}))
            with self.assertRaisesRegex(ValueError, "not a calibrated completed dense cloud"):
                finish_dense.finish(source, output)
            self.assertFalse(output.exists())

    def test_local_preserved_013_passes_read_only_gate(self):
        source = ROOT / "build-opencv/classical-ycb-calibrated-013"
        if not (source / "result.json").is_file():
            self.skipTest("preserved calibrated 013 artifact unavailable")
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-tools/tmp") as tmp:
            output = Path(tmp) / "output"
            with mock.patch.object(finish_dense.shutil, "disk_usage",
                                   return_value=types.SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "10 GiB disk floor"):
                    finish_dense.finish(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
