"""Synthetic report-schema tests; never invoke the sealed receipt worker."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.object_motion.mustard_verified_track_receipt import (
    ReceiptError, _bounded_json, _child_failure, _safe_failure,
    sources_from_reports,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReceiptSchemaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images = Path(self.temp.name) / "images"
        self.images.mkdir()
        names = [f"TRAIN_{i:03d}.jpg" for i in range(48)]
        photos, masks, inputs, pose_masks = {}, {}, [], []
        for name in names:
            path = self.images / name
            Image.new("RGB", (8, 6), (20, 30, 40)).save(path)
            photo_sha = digest(path)
            mask_sha = hashlib.sha256((name + "-mask").encode()).hexdigest()
            photos[name] = photo_sha
            masks[name] = {"source": "/synthetic/" + name + ".png", "sha256": mask_sha,
                           "kept_pixels": 35, "ignored_pixels": 13}
            inputs.append({"name": name, "sha256": photo_sha})
            pose_masks.append({"name": name + ".png", "sha256": mask_sha})
        names_digest = hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()
        self.stage = {"schema": "mustard_train_only_stage_v1", "status": "complete",
                      "mask_role": "accepted coarse pose support, not object silhouette or ground truth",
                      "train_names": names, "train_names_sha256": names_digest,
                      "train_photo_sha256": photos, "cleaned_masks": masks,
                      "heldout_names_excluded": ["HELDOUT.jpg"],
                      "copy_bytes": 12345, "copied_bytes": 12345}
        self.producer = {"schema": "classical_backend_v1", "status": "sparse_complete",
                         "sfm_source": {"kind": "internal_image_only_pycolmap"},
                         "sfm_intrinsics_policy": "fixed-initial",
                         "inputs": inputs, "pose_masks": pose_masks}

    def parse(self, stage=None, producer=None):
        return sources_from_reports(stage or self.stage, producer or self.producer, self.images)

    def test_extracts_exact_image_sources_from_real_v1_fields(self):
        sources = self.parse()
        self.assertEqual(len(sources), 48)
        self.assertEqual((sources[0].width, sources[0].height), (8, 6))
        self.assertEqual(sources[0].mask_sha256, self.stage["cleaned_masks"][sources[0].name]["sha256"])

    def test_rejects_changed_producer_hash_and_order(self):
        producer = deepcopy(self.producer)
        producer["inputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReceiptError, "source hash"):
            self.parse(producer=producer)
        producer = deepcopy(self.producer)
        producer["pose_masks"].reverse()
        with self.assertRaisesRegex(ReceiptError, "inventory"):
            self.parse(producer=producer)

    def test_rejects_mask_role_and_pixel_count(self):
        stage = deepcopy(self.stage)
        stage["mask_role"] = "object-only"
        with self.assertRaisesRegex(ReceiptError, "stage report"):
            self.parse(stage=stage)
        stage = deepcopy(self.stage)
        stage["cleaned_masks"][stage["train_names"][0]]["kept_pixels"] = 34
        with self.assertRaisesRegex(ReceiptError, "pixel counts"):
            self.parse(stage=stage)

    def test_rejects_non_train_names_and_changed_photo(self):
        stage = deepcopy(self.stage)
        stage["heldout_names_excluded"] = [stage["train_names"][0]]
        with self.assertRaisesRegex(ReceiptError, "overlap"):
            self.parse(stage=stage)
        (self.images / self.stage["train_names"][0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ReceiptError, "RGB bytes"):
            self.parse()

    def test_bounded_json_requires_exact_hash(self):
        path = Path(self.temp.name) / "report.json"
        path.write_text('{"schema":"fixture"}')
        self.assertEqual(_bounded_json(path, digest(path))["schema"], "fixture")
        with self.assertRaisesRegex(ReceiptError, "hash"):
            _bounded_json(path, "0" * 64)

    def test_child_failure_is_bounded_and_path_redacted(self):
        error = ReceiptError("source mismatch: /private/secret folder/report.json\n" + "x" * 2000)
        safe = _safe_failure(error)
        self.assertEqual(safe["error_class"], "ReceiptError")
        self.assertNotIn("private", safe["error_message"])
        self.assertNotIn("secret", safe["error_message"])
        self.assertLessEqual(len(safe["error_message"]), 768)
        rendered = _child_failure(json.dumps(safe).encode())
        self.assertIn("source mismatch", rendered)
        self.assertLessEqual(len(rendered), 1024)
        self.assertEqual(_child_failure(b"Traceback /private/secret"),
                         "bounded diagnostic unavailable")


if __name__ == "__main__":
    unittest.main()
