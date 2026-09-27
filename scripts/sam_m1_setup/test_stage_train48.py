"""No remote transfer is performed by these pure staging contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.sam_m1_setup import stage_train48 as stage


class Train48StageTests(unittest.TestCase):
    @staticmethod
    def synthetic_package():
        return {"schema": "ycb_object_evaluation_package_v1", "object_id": "006_mustard_bottle",
                "manifest_sha256": stage.ACQUISITION_SHA256,
                "training_inputs": [{"path": f"photos/{name}", "bytes": 1,
                                     "sha256": hashlib.sha256(b"x").hexdigest()}
                                    for name in stage.TRAIN_NAMES],
                "heldout_photos": [{"path": f"photos/{name}"} for name in stage.HELDOUT_NAMES]}

    def test_frozen_package_exact_train_and_heldout_names(self):
        package = self.synthetic_package()
        expected = stage.expected_photos(package)
        self.assertEqual(list(expected), list(stage.TRAIN_NAMES))
        self.assertEqual(len(expected), 48)
        package["training_inputs"][-1]["path"] = "photos/NP3_024.jpg"
        with self.assertRaisesRegex(ValueError, "exact 48-train"):
            stage.expected_photos(package)
        package = self.synthetic_package()
        package["heldout_photos"].pop()
        with self.assertRaisesRegex(ValueError, "exact 48-train"):
            stage.expected_photos(package)

    def test_remote_source_inventory_is_exact(self):
        package = self.synthetic_package()
        expected = stage.expected_photos(package)
        stage.validate_remote_inventory(expected.copy(), expected)
        for changed in (
            {name: row for name, row in expected.items() if name != stage.TRAIN_NAMES[0]},
            {**expected, stage.HELDOUT_NAMES[0]: expected[stage.TRAIN_NAMES[0]]},
            {**expected, stage.TRAIN_NAMES[0]: {"bytes": 1, "sha256": "0" * 64}},
        ):
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                stage.validate_remote_inventory(changed, expected)
        with patch.object(stage, "remote_inventory", return_value=expected), \
             patch.object(stage.Path, "is_mount", return_value=True), \
             patch.object(stage.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)):
            # A same-device temporary directory must fail the external-mount gate.
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with self.assertRaisesRegex(ValueError, "external"):
                    stage.preflight(root / "fresh", mount=root, root=root)

    def test_success_and_heldout_copy_rejection_with_mock_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder) / "package.json"
            package.write_text(json.dumps(self.synthetic_package()))
            package_sha = stage.sha256(package)
            output = Path(folder) / "train"
            value = b"x"
            digest = hashlib.sha256(value).hexdigest()
            base = {"schema": "sam21_m1_train48_photos_v1", "status": "preflight",
                    "package_sha256": package_sha,
                    "runner_sha256": stage.sha256(Path(stage.__file__)),
                    "photo_sha256": {name: digest for name in stage.TRAIN_NAMES},
                    "photo_bytes": {name: 1 for name in stage.TRAIN_NAMES},
                    "train_names_sha256": hashlib.sha256(("\n".join(stage.TRAIN_NAMES)+"\n").encode()).hexdigest(),
                    "train_names": list(stage.TRAIN_NAMES), "output": str(output)}
            inventory = {name: {"bytes": 1, "sha256": digest} for name in stage.TRAIN_NAMES}

            def transfer(run, *args, **kwargs):
                for name in stage.TRAIN_NAMES:
                    (run / "photos" / name).write_bytes(value)
                return {"name": "rsync-train48", "status": "complete"}

            with patch.object(stage, "PACKAGE", package), \
                 patch.object(stage, "PACKAGE_SHA256", package_sha), \
                 patch.object(stage, "preflight", return_value=base.copy()), \
                 patch.object(stage.bounded, "stage", side_effect=transfer), \
                 patch.object(stage, "remote_inventory", return_value=inventory), \
                 patch.object(stage.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)):
                self.assertEqual(stage.execute(output)["status"], "complete")

            second = Path(folder) / "with-heldout"
            def bad_transfer(run, *args, **kwargs):
                transfer(run, *args, **kwargs)
                (run / "photos" / stage.HELDOUT_NAMES[0]).write_bytes(value)
                return {"name": "rsync-train48", "status": "complete"}
            with patch.object(stage, "PACKAGE", package), \
                 patch.object(stage, "PACKAGE_SHA256", package_sha), \
                 patch.object(stage, "preflight", return_value={**base, "output": str(second)}), \
                 patch.object(stage.bounded, "stage", side_effect=bad_transfer), \
                 patch.object(stage.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)):
                with self.assertRaisesRegex(ValueError, "held-out"):
                    stage.execute(second)
            self.assertEqual(json.loads((second / "stage-report.json").read_text())["status"], "failed")

            third = Path(folder) / "native-failure"
            failed_stage = {"name": "rsync-train48", "status": "failed", "exit_code": 12,
                            "failure": "mock transfer failure"}
            with patch.object(stage, "PACKAGE", package), \
                 patch.object(stage, "preflight", return_value={**base, "output": str(third)}), \
                 patch.object(stage.bounded, "stage", side_effect=stage.bounded.StageError(failed_stage)):
                with self.assertRaises(stage.bounded.StageError):
                    stage.execute(third)
            saved = json.loads((third / "stage-report.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["stages"], [failed_stage])


if __name__ == "__main__":
    unittest.main()
