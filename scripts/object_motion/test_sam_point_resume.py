import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import sam_point_resume as resume
from scripts.object_motion import ycb_object_masks as common


def fixture():
    names = [f"NP3_{i * 6:03}.jpg" for i in range(48)]
    rows = [{"path": "photos/" + name, "sha256": f"{i:064x}"} for i, name in enumerate(names)]
    prompts = [{"name": name, "points_xy_label": [[600, 525, 1], [720, 620, 0], [520, 630, 0]]}
               for name in names]
    return rows, prompts


def mask_bytes():
    array = np.zeros((1024, 1280), dtype=np.uint8)
    array[470:580, 550:670] = 255
    output = io.BytesIO()
    Image.fromarray(array, "L").save(output, "PNG")
    return output.getvalue()


class SAMPointResumeTests(unittest.TestCase):
    def test_fake_predictions_publish_completed_frames_across_interruption(self):
        rows, prompts = fixture()
        raw = np.zeros((1024, 1280), dtype=np.uint8)
        raw[470:580, 550:670] = 255
        calls = 0

        def fake_predict(_photo, box, points):
            nonlocal calls
            self.assertEqual(list(box), [480, 300, 760, 650])
            self.assertEqual(points, prompts[16]["points_xy_label"])
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt("simulated worker interruption")
            return raw

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": "a" * 64,
                       "cleaned_mask_sha256": "a" * 64} for row in rows[:16]]
            with self.assertRaises(KeyboardInterrupt):
                for index in range(16, 20):
                    mask, cleaned, stats = resume.candidate_masks(fake_predict, None, prompts[index])
                    output = io.BytesIO()
                    Image.fromarray(mask, "L").save(output, "PNG")
                    raw_bytes = output.getvalue()
                    output = io.BytesIO()
                    Image.fromarray(cleaned, "L").save(output, "PNG")
                    clean_bytes = output.getvalue()
                    import hashlib
                    record = resume.frame_record(Path(rows[index]["path"]).name, rows[index]["sha256"],
                                                 prompts[index]["points_xy_label"],
                                                 hashlib.sha256(raw_bytes).hexdigest(),
                                                 hashlib.sha256(clean_bytes).hexdigest(),
                                                 {"kind": "resume_inference", "invocation_id": "001",
                                                  "runner_sha256": resume.dependencies()["runner_sha256"]}, stats)
                    resume.publish_frame(root, "001", record, raw_bytes, clean_bytes)
            records = resume.verified_frames(root, rows, prompts, sealed)
            self.assertEqual(set(records), {Path(rows[16]["path"]).name, Path(rows[17]["path"]).name})
            self.assertFalse(list((root / "failures").iterdir()))
            self.assertEqual(calls, 3)
            name = Path(rows[16]["path"]).name
            path = root / "frames" / name / "frame.json"
            record = json.loads(path.read_text())
            record["producer"]["runner_sha256"] = "0" * 64
            path.write_text(json.dumps(record))
            receipt_path = root / "receipts" / f"{name}.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["frame_manifest_sha256"] = common.digest(path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "frozen resume inference"):
                resume.verified_frames(root, rows, prompts, sealed)
            path.write_text(json.dumps({**record, "producer": {**record["producer"],
                                                           "runner_sha256": resume.dependencies()["runner_sha256"]}}))
            receipt["frame_manifest_sha256"] = common.digest(path)
            receipt_path.write_text(json.dumps(receipt))
            (root / "frames" / "HELDOUT.jpg").mkdir()
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                resume.verified_frames(root, rows, prompts, sealed)

    def test_self_consistent_later_frame_replacement_rejected_by_prior_invocation(self):
        rows, prompts = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            data = mask_bytes()
            digest = __import__("hashlib").sha256(data).hexdigest()
            name = Path(rows[16]["path"]).name
            record = resume.frame_record(name, rows[16]["sha256"], prompts[16]["points_xy_label"], digest, digest,
                                         {"kind": "resume_inference", "invocation_id": "001",
                                          "runner_sha256": resume.dependencies()["runner_sha256"]})
            resume.publish_frame(root, "001", record, data, data)
            sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": "a" * 64,
                       "cleaned_mask_sha256": "a" * 64} for row in rows[:16]]
            records = resume.verified_frames(root, rows, prompts, sealed)
            resume.exclusive_json(root / "invocations" / "001.json",
                                  {"schema": "sam21_mustard_point_resume_invocation_v1",
                                   "frame_sha256_map": resume.frame_hash_map(root, records)})
            resume.verify_prior_invocations(root, records)
            manifest = root / "frames" / name / "frame.json"
            record["diagnostics"] = {"self_consistent_but_changed": True}
            manifest.write_text(json.dumps(record))
            receipt = root / "receipts" / f"{name}.json"
            updated = json.loads(receipt.read_text())
            updated["frame_manifest_sha256"] = common.digest(manifest)
            receipt.write_text(json.dumps(updated))
            records = resume.verified_frames(root, rows, prompts, sealed)
            with self.assertRaisesRegex(ValueError, "prior invocation seal"):
                resume.verify_prior_invocations(root, records)

    def test_sealed_first_frame_tamper_rejected_on_resume(self):
        rows, prompts = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            data = mask_bytes()
            digest = __import__("hashlib").sha256(data).hexdigest()
            sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": digest,
                       "cleaned_mask_sha256": digest} for row in rows[:16]]
            record = resume.frame_record(sealed[0]["name"], rows[0]["sha256"], prompts[0]["points_xy_label"],
                                         digest, digest, {"kind": "sealed_prior", "parent_sha256": resume.PRIOR_SHA,
                                                         "batch_sha256": resume.PRIOR_BATCH_SHA[0]})
            resume.publish_frame(root, "import-001", record, data, data)
            self.assertEqual(len(resume.verified_frames(root, rows, prompts, sealed)), 1)
            path = root / "frames" / sealed[0]["name"] / "frame.json"
            record["producer"]["batch_sha256"] = "0" * 64
            path.write_text(json.dumps(record))
            receipt_path = root / "receipts" / f"{sealed[0]['name']}.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["frame_manifest_sha256"] = common.digest(path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "sealed prior"):
                resume.verified_frames(root, rows, prompts, sealed)
            record["producer"]["batch_sha256"] = resume.PRIOR_BATCH_SHA[0]
            record["raw_mask_sha256"] = "0" * 64
            path.write_text(json.dumps(record))
            receipt["frame_manifest_sha256"] = common.digest(path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "sealed prior"):
                resume.verified_frames(root, rows, prompts, sealed)

    def test_interrupted_staging_does_not_overwrite_and_new_invocation_can_publish(self):
        rows, prompts = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            data = mask_bytes()
            digest = __import__("hashlib").sha256(data).hexdigest()
            sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": digest,
                       "cleaned_mask_sha256": digest} for row in rows[:16]]
            name = sealed[0]["name"]
            (root / "staging" / f"import-001-{name}").mkdir()  # interrupted before atomic publish
            record = resume.frame_record(name, rows[0]["sha256"], prompts[0]["points_xy_label"], digest, digest,
                                         {"kind": "sealed_prior", "parent_sha256": resume.PRIOR_SHA,
                                          "batch_sha256": resume.PRIOR_BATCH_SHA[0]})
            resume.publish_frame(root, "import-002", record, data, data)
            self.assertEqual(len(resume.verified_frames(root, rows, prompts, sealed)), 1)
            with self.assertRaises(FileExistsError):
                resume.publish_frame(root, "import-003", record, data, data)
            self.assertTrue((root / "staging" / f"import-001-{name}").exists())

    def test_receipt_recovers_interrupted_atomic_publication(self):
        rows, prompts = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name in ("frames", "staging", "failures", "invocations", "receipts"):
                (root / name).mkdir()
            data = mask_bytes()
            digest = __import__("hashlib").sha256(data).hexdigest()
            name = Path(rows[0]["path"]).name
            record = resume.frame_record(name, rows[0]["sha256"], prompts[0]["points_xy_label"], digest, digest,
                                         {"kind": "sealed_prior", "parent_sha256": resume.PRIOR_SHA,
                                          "batch_sha256": resume.PRIOR_BATCH_SHA[0]})
            resume.publish_frame(root, "import-001", record, data, data)
            stage = root / "staging" / f"import-001-{name}"
            (root / "frames" / name).rename(stage)  # death after receipt, before rename
            self.assertFalse((root / "frames" / name).exists())
            resume.recover_receipts(root, [name])
            self.assertTrue((root / "frames" / name).exists())
            sealed = [{"name": Path(row["path"]).name, "raw_mask_sha256": digest,
                       "cleaned_mask_sha256": digest} for row in rows[:16]]
            self.assertEqual(len(resume.verified_frames(root, rows, prompts, sealed)), 1)

    def test_complete_inventory_requires_exact_48_and_binds_relative_masks(self):
        rows, _ = fixture()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            (root / "contract.json").write_text("{}")
            self.assertIsNone(resume.complete_inventory(root, rows, {}, {"package_sha256": "a" * 64}))
            for row in rows:
                name = Path(row["path"]).name
                path = root / "frames" / name
                path.mkdir(parents=True)
                (path / "frame.json").write_text("{}")
            records = {Path(row["path"]).name: {"cleaned_mask_sha256": "a" * 64} for row in rows}
            resume.complete_inventory(root, rows, records, {"package_sha256": "b" * 64})
            report = json.loads((root / "complete_inventory.json").read_text())
            self.assertEqual(len(report["images"]), 48)
            self.assertEqual(report["images"][0]["cleaned_mask_path"], "frames/NP3_000.jpg/clean.png")
            self.assertFalse(report["full_training_package_ready"])


if __name__ == "__main__":
    unittest.main()
