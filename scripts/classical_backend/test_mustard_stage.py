import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.classical_backend import mustard_stage as stage
from scripts.classical_backend import run as classical_run


class MustardStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / "train-photos"
        self.frames = self.root / "sam" / "frames"
        self.photos.mkdir()
        self.frames.mkdir(parents=True)
        training, mask_rows, mask_hashes = [], [], {}
        for index, name in enumerate(stage.TRAIN_NAMES):
            photo = self.photos / name
            Image.new("RGB", (4, 4), (index + 2, 30, 40)).save(photo)
            clean = self.frames / name / "clean.png"
            clean.parent.mkdir()
            alpha = Image.new("L", (4, 4), 0)
            alpha.putpixel((1, 1), 255)
            alpha.save(clean)
            photo_hash = stage.sha256(photo)
            mask_hash = stage.sha256(clean)
            training.append({"path": "photos/" + name, "sha256": photo_hash,
                             "bytes": photo.stat().st_size})
            mask_rows.append({"name": name, "source_sha256": photo_hash,
                              "cleaned_mask_sha256": mask_hash,
                              "cleaned_mask_path": f"frames/{name}/clean.png"})
            mask_hashes[name] = mask_hash
        self.package = self.root / "package.json"
        package_data = {"schema": "ycb_object_evaluation_package_v1", "object_id": "006_mustard_bottle",
                        "manifest_sha256": stage.ACQUISITION_SHA256,
                        "training_inputs": training,
                        "heldout_photos": [{"path": "photos/" + name} for name in stage.HELDOUT_NAMES]}
        self.package.write_text(json.dumps(package_data))
        self.inventory = self.root / "sam" / "complete_inventory.json"
        self.inventory.write_text(json.dumps({"schema": "sam21_mustard_point_mask_inventory_v1",
                                              "status": "generated_unreviewed",
                                              "package_sha256": stage.sha256(self.package),
                                              "prompt_sha256": stage.PROMPTS_SHA256,
                                              "images": mask_rows}))
        self.qa = self.root / "qa.json"
        self.qa.write_text(json.dumps({"schema": "mustard_sam_mask_qa_v1", "status": "accepted",
                                       "decision": "accepted_for_coarse_pose_support",
                                       "inventory_sha256": stage.sha256(self.inventory),
                                       "reviewed_names": list(stage.TRAIN_NAMES),
                                       "cleaned_mask_sha256": mask_hashes,
                                       "reviewer": "test reviewer", "reviewed_at": "2026-09-27"}))
        self.output = self.root / "out"

    def options(self):
        return {"output": self.output, "mount": self.root, "require_mount": False,
                "package_sha256": stage.sha256(self.package), "expected_size": (4, 4)}

    def args(self):
        return (self.package, self.photos, self.inventory, self.inventory.parent,
                stage.sha256(self.inventory), self.qa, stage.sha256(self.qa))

    def preflight(self):
        with patch.object(stage.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)):
            return stage.preflight(*self.args(), **self.options())

    def test_exact_48_training_package_and_prepare(self):
        plan = self.preflight()
        self.assertEqual(plan["train_names"], list(stage.TRAIN_NAMES))
        self.assertEqual(set(plan["heldout_names_excluded"]), set(stage.HELDOUT_NAMES))
        with patch.object(stage.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)):
            result = stage.prepare(*self.args(), **self.options())
        self.assertEqual(result["status"], "complete")
        self.assertEqual({p.name for p in (self.output / "images").iterdir()}, set(stage.TRAIN_NAMES))
        self.assertEqual({p.name for p in (self.output / "masks").iterdir()},
                         {name + ".png" for name in stage.TRAIN_NAMES})
        self.assertEqual(stage.sha256(self.output / "train-names.txt"), plan["train_names_sha256"])

    def test_missing_and_heldout_photo_rejected(self):
        (self.photos / stage.TRAIN_NAMES[0]).unlink()
        with self.assertRaisesRegex(ValueError, "missing|extra"):
            self.preflight()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.photos / stage.TRAIN_NAMES[0])
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.photos / stage.HELDOUT_NAMES[0])
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.preflight()

    def test_partial_inventory_and_unaccepted_qa_rejected(self):
        data = json.loads(self.inventory.read_text())
        data["images"].pop()
        self.inventory.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.preflight()
        data["images"].append({"name": stage.TRAIN_NAMES[-1],
                               "source_sha256": stage.sha256(self.photos / stage.TRAIN_NAMES[-1]),
                               "cleaned_mask_sha256": stage.sha256(self.frames / stage.TRAIN_NAMES[-1] / "clean.png"),
                               "cleaned_mask_path": f"frames/{stage.TRAIN_NAMES[-1]}/clean.png"})
        self.inventory.write_text(json.dumps(data))
        qa = json.loads(self.qa.read_text())
        qa["inventory_sha256"] = stage.sha256(self.inventory)
        qa["status"] = "pending"
        self.qa.write_text(json.dumps(qa))
        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.preflight()

    def test_tampered_cleaned_mask_and_qa_hash_rejected(self):
        clean = self.frames / stage.TRAIN_NAMES[0] / "clean.png"
        clean.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mask hash"):
            self.preflight()
        clean.unlink()
        self.qa.write_text(self.qa.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "sealed JSON"):
            stage.preflight(*self.args()[:-1], "0" * 64, **self.options())

    def test_changed_photo_and_mask_path_rejected(self):
        photo = self.photos / stage.TRAIN_NAMES[0]
        original = photo.read_bytes()
        Image.new("RGB", (4, 4), (100, 20, 30)).save(photo)
        with self.assertRaisesRegex(ValueError, "photo hash"):
            self.preflight()
        photo.write_bytes(original)
        inventory = json.loads(self.inventory.read_text())
        inventory["images"][0]["cleaned_mask_path"] = "frames/other/clean.png"
        self.inventory.write_text(json.dumps(inventory))
        qa = json.loads(self.qa.read_text())
        qa["inventory_sha256"] = stage.sha256(self.inventory)
        self.qa.write_text(json.dumps(qa))
        with self.assertRaisesRegex(ValueError, "relative path"):
            self.preflight()

    def test_external_output_requires_internal_reserve_monitor(self):
        self.assertEqual(classical_run.reserve_paths_for_devices(2, 1), (classical_run.ROOT,))
        self.assertEqual(classical_run.reserve_paths_for_devices(1, 1), ())


if __name__ == "__main__":
    unittest.main()
