"""Hermetic contract tests for the unreviewed 43+5 mustard candidate."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.classical_backend import mustard_candidate48 as candidate
from scripts.classical_backend import mustard_stage as stage
from scripts.object_motion import sam_point_trial as point


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return stage.sha256(path)


def mask(path, pixel):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (4, 4), 0)
    image.putpixel(pixel, 255)
    image.save(path)
    return stage.sha256(path)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__import__("os").environ.get("TMPDIR", stage.ROOT / ".local-tools/tmp"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.correction = self.root / "correction"
        self.output = self.root / "candidate"
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.base.mkdir()
        self.correction.mkdir()
        train, old_rows, old_hashes, corrected_rows, prompt_rows = [], [], {}, [], []
        for index, name in enumerate(stage.TRAIN_NAMES):
            photo = self.photos / name
            Image.new("RGB", (4, 4), (index + 1, 30, 40)).save(photo)
            source_hash = stage.sha256(photo)
            train.append({"path": "photos/" + name, "sha256": source_hash,
                          "bytes": photo.stat().st_size})
            frame = self.base / "frames" / name
            raw_hash = mask(frame / "raw.png", (1, 1))
            clean_hash = mask(frame / "clean.png", (1, 1))
            record_hash = write_json(frame / "frame.json", {
                "name": name, "source_sha256": source_hash,
                "prompt_sha256": stage.PROMPTS_SHA256,
                "points_xy_label": [[1, 1, 1], [2, 2, 0], [3, 3, 0]],
                "raw_mask_sha256": raw_hash, "cleaned_mask_sha256": clean_hash})
            old_rows.append({"name": name, "source_sha256": source_hash,
                             "cleaned_mask_sha256": clean_hash,
                             "cleaned_mask_path": f"frames/{name}/clean.png",
                             "frame_manifest_sha256": record_hash})
            old_hashes[name] = clean_hash
            if name in candidate.CORRECTED:
                raw = f"raw_masks/{name}.png"
                clean = f"cleaned_masks/{name}.png"
                corrected_rows.append({"name": name, "source_sha256": source_hash,
                                       "points_xy_label": [[1, 1, 1], [2, 2, 0], [3, 3, 0], [620, 330, 0]],
                                       "raw_mask_path": raw,
                                       "cleaned_mask_path": clean,
                                       "raw_mask_sha256": mask(self.correction / raw, (2, 1)),
                                       "cleaned_mask_sha256": mask(self.correction / clean, (2, 1))})
                prompt_rows.append({"name": name, "source_sha256": source_hash,
                                    "points_xy_label": corrected_rows[-1]["points_xy_label"]})
        self.package = self.root / "package.json"
        self.package_sha = write_json(self.package, {
            "schema": "ycb_object_evaluation_package_v1", "object_id": "006_mustard_bottle",
            "manifest_sha256": stage.ACQUISITION_SHA256,
            "training_inputs": train,
            "heldout_photos": [{"path": "photos/" + name} for name in stage.HELDOUT_NAMES]})
        self.base_sha = write_json(self.base / "complete_inventory.json", {
            "schema": "sam21_mustard_point_mask_inventory_v1", "status": "generated_unreviewed",
            "package_sha256": self.package_sha, "prompt_sha256": stage.PROMPTS_SHA256,
            "images": old_rows})
        self.rejected = self.root / "rejected.json"
        self.rejected_sha = write_json(self.rejected, {
            "status": "rejected", "inventory_sha256": self.base_sha,
            "rejected_names": list(candidate.CORRECTED),
            "reviewed_names": list(stage.TRAIN_NAMES),
            "approved_for_sfm": False, "cleaned_mask_sha256": old_hashes})
        self.prompts = self.root / "prompts.json"
        self.prompts_sha = write_json(self.prompts, {
            "schema": "sam21_mustard_m1_board_negative5_prompts_v1",
            "original_prompt_sha256": stage.PROMPTS_SHA256,
            "base_inventory_sha256": self.base_sha,
            "rejected_full_qa_sha256": self.rejected_sha, "images": prompt_rows})
        self.manifest = self.correction / "manifest.json"
        self.manifest_sha = write_json(self.manifest, {
            "schema": "sam21_mustard_m1_board5_candidate_v1", "status": "generated_unreviewed",
            "package_sha256": self.package_sha,
            "original_prompt_sha256": stage.PROMPTS_SHA256,
            "correction_prompt_sha256": self.prompts_sha,
            "base_inventory_sha256": self.base_sha,
            "rejected_full_qa_sha256": self.rejected_sha,
            "checkpoint_sha256": point.MODEL_SHA,
            "source_inventory_sha256": point.SOURCE_SHA,
            "runner_sha256": stage.sha256(candidate.ROOT / "scripts/object_motion/sam_m1_board5.py"),
            "images": corrected_rows})
        write_json(self.correction / "supervisor.json", {
            "schema": "sam21_mustard_m1_board5_supervisor_v1",
            "status": "generated_unreviewed", "reason": None})
        self.board_qa = self.root / "board-qa.json"
        self.board_qa_sha = write_json(self.board_qa, {
            "schema": "sam21_mustard_m1_board5_review_v1", "status": "accepted",
            "decision": "accepted_for_coarse_pose_support", "manifest_sha256": self.manifest_sha,
            "correction_prompt_sha256": self.prompts_sha,
            "reviewed_names": list(candidate.CORRECTED),
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in corrected_rows},
            "reviewer": "fixture", "reviewed_at": "2026-09-27"})
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(stage, "PACKAGE_SHA256", self.package_sha))
        self.patches.enter_context(patch.object(candidate, "BASE_SHA", self.base_sha))
        self.patches.enter_context(patch.object(candidate, "REJECTED_QA_SHA", self.rejected_sha))
        self.patches.enter_context(patch.object(candidate, "BOARD5_PROMPTS_SHA", self.prompts_sha))
        # The runtime is Mac-only and pins LF source bytes. Windows checkouts may
        # convert this Python file to CRLF; fixture lineage uses the actual bytes.
        self.patches.enter_context(patch.object(
            candidate, "BOARD5_RUNNER_SHA",
            stage.sha256(candidate.ROOT / "scripts/object_motion/sam_m1_board5.py")))
        self.patches.enter_context(patch.object(candidate.shutil, "disk_usage", return_value=SimpleNamespace(free=1 << 50)))

    def args(self):
        return (self.base, self.correction, self.manifest_sha, self.prompts,
                self.prompts_sha, self.board_qa, self.board_qa_sha)

    def options(self):
        return {"output": self.output, "mount": self.root, "package_path": self.package,
                "rejected_qa": self.rejected, "expected_size": (4, 4), "require_mount": False}

    def compose(self):
        return candidate.compose(*self.args(), **self.options())

    def test_43_base_5_corrected_and_full_qa_gate(self):
        inventory = self.compose()
        self.assertEqual(len(inventory["images"]), 48)
        self.assertEqual([r["origin"] for r in inventory["images"]].count("base43"), 43)
        for row in inventory["images"]:
            chosen = self.output / row["cleaned_mask_path"]
            self.assertEqual(stage.sha256(chosen), row["cleaned_mask_sha256"])
            if row["origin"] == "base43":
                self.assertEqual(row["prompt_sha256"], stage.PROMPTS_SHA256)
                self.assertEqual(chosen.read_bytes(), (self.base / row["cleaned_mask_path"]).read_bytes())
            else:
                self.assertEqual(row["prompt_sha256"], self.prompts_sha)
                self.assertNotEqual(chosen.read_bytes(), (self.base / row["cleaned_mask_path"]).read_bytes())
        self.assertEqual(inventory["status"], "generated_unreviewed")
        full_qa = self.root / "full-qa.json"
        write_json(full_qa, {"schema": "mustard_sam_composed_candidate_qa_v1",
            "status": "pending", "decision": "pending", "inventory_sha256": stage.sha256(self.output / "complete_inventory.json"),
            "reviewed_names": list(stage.TRAIN_NAMES),
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in inventory["images"]}})
        params = (self.package, self.photos, self.output / "complete_inventory.json", self.output,
                  stage.sha256(self.output / "complete_inventory.json"), full_qa, stage.sha256(full_qa))
        opts = {"output": self.root / "stage", "mount": self.root, "require_mount": False,
                "package_sha256": self.package_sha, "expected_size": (4, 4)}
        with self.assertRaisesRegex(ValueError, "full-48.*QA"):
            stage.preflight(*params, **opts)
        qa = json.loads(full_qa.read_text())
        qa.update(status="accepted", decision="accepted_for_coarse_pose_support",
                  reviewer="fixture", reviewed_at="2026-09-27")
        write_json(full_qa, qa)
        params = (*params[:-1], stage.sha256(full_qa))
        self.assertEqual(stage.preflight(*params, **opts)["inventory_schema"],
                         "sam21_mustard_m1_composed_candidate_v1")

    def test_failed_supervisor_blocks_worker_manifest(self):
        write_json(self.correction / "supervisor.json", {
            "schema": "sam21_mustard_m1_board5_supervisor_v1", "status": "failed", "reason": "resource cap"})
        with self.assertRaisesRegex(ValueError, "supervisor"):
            candidate.preflight(*self.args(), **self.options())

    def test_prompt_header_and_model_lineage_tamper_rejected(self):
        prompts = json.loads(self.prompts.read_text())
        prompts["base_inventory_sha256"] = "0" * 64
        prompt_sha = write_json(self.prompts, prompts)
        with self.assertRaisesRegex(ValueError, "lineage"):
            candidate.preflight(self.base, self.correction, self.manifest_sha, self.prompts,
                                prompt_sha, self.board_qa, self.board_qa_sha, **self.options())
        prompts["base_inventory_sha256"] = self.base_sha
        self.assertEqual(write_json(self.prompts, prompts), self.prompts_sha)
        manifest = json.loads(self.manifest.read_text())
        for key in ("source_inventory_sha256", "checkpoint_sha256", "runner_sha256"):
            changed = {**manifest, key: "0" * 64}
            changed_sha = write_json(self.manifest, changed)
            with self.assertRaisesRegex(ValueError, "lineage"):
                candidate.preflight(self.base, self.correction, changed_sha, self.prompts,
                                    self.prompts_sha, self.board_qa, self.board_qa_sha, **self.options())

    def test_composed_source_audit_rejects_late_base_change(self):
        inventory = self.compose()
        raw = self.base / "frames" / stage.TRAIN_NAMES[0] / "raw.png"
        mask(raw, (3, 1))
        qa = self.root / "full-qa.json"
        qa_sha = write_json(qa, {
            "schema": "mustard_sam_composed_candidate_qa_v1", "status": "accepted",
            "decision": "accepted_for_coarse_pose_support",
            "inventory_sha256": stage.sha256(self.output / "complete_inventory.json"),
            "reviewed_names": list(stage.TRAIN_NAMES),
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in inventory["images"]},
            "reviewer": "fixture", "reviewed_at": "2026-09-27"})
        with self.assertRaisesRegex(ValueError, "source or metadata changed"):
            stage.preflight(self.package, self.photos, self.output / "complete_inventory.json",
                            self.output, stage.sha256(self.output / "complete_inventory.json"),
                            qa, qa_sha, output=self.root / "stage", mount=self.root,
                            require_mount=False, package_sha256=self.package_sha,
                            expected_size=(4, 4))

    def test_missing_corrected_mask_or_changed_base_raw_rejected(self):
        (self.correction / "raw_masks" / (candidate.CORRECTED[0] + ".png")).unlink()
        with self.assertRaises(ValueError):
            candidate.preflight(*self.args(), **self.options())
        mask(self.correction / "raw_masks" / (candidate.CORRECTED[0] + ".png"), (2, 1))
        mask(self.base / "frames" / stage.TRAIN_NAMES[0] / "raw.png", (3, 1))
        with self.assertRaisesRegex(ValueError, "mask hash"):
            candidate.preflight(*self.args(), **self.options())


if __name__ == "__main__":
    unittest.main()
