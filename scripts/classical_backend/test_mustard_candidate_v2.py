"""Hermetic 43+3+2 subset-selection and independent-full-QA contracts."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image

from scripts.classical_backend import mustard_candidate48 as candidate
from scripts.classical_backend import mustard_stage as stage
from scripts.classical_backend.test_mustard_candidate48 import CandidateTests, mask, write_json
from scripts.object_motion import sam_point_trial as point


class CandidateV2Tests(unittest.TestCase):
    def setUp(self):
        CandidateTests.setUp(self)
        self.patches.enter_context(patch.object(candidate, "BOARD_POINT", [3, 0, 0]))
        self.patches.enter_context(patch.object(candidate, "BASE_BOX", [0, 0, 4, 4]))
        prompts = json.loads(self.prompts.read_text())
        prompts["box_xyxy_original_pixels"] = [0, 0, 4, 4]
        for row in prompts["images"]:
            row["points_xy_label"][-1] = [3, 0, 0]
        self.prompts_sha = write_json(self.prompts, prompts)
        self.patches.enter_context(patch.object(candidate, "BOARD5_PROMPTS_SHA", self.prompts_sha))
        manifest = json.loads(self.manifest.read_text())
        manifest["correction_prompt_sha256"] = self.prompts_sha
        for row in manifest["images"]:
            row["box_xyxy_original_pixels"] = [0, 0, 4, 4]
            row["points_xy_label"][-1] = [3, 0, 0]
            row["status"] = "complete_unreviewed" if row["name"] in candidate.PARTIAL_NAMES else "failed_unusable"
            if row["status"] == "failed_unusable":
                row["cleaned_mask_path"] = None
                row["cleaned_mask_sha256"] = None
        manifest["status"] = "partial_unusable"
        self.partial_sha = write_json(self.manifest, manifest)
        self.patches.enter_context(patch.object(candidate, "BOARD5_PARTIAL_MANIFEST_SHA", self.partial_sha))
        self.supervisor_sha = write_json(self.correction / "supervisor.json", {
            "schema": "sam21_mustard_m1_board5_supervisor_v1", "status": "failed",
            "reason": "worker exit 1", "seconds_total": 27.7,
            "seconds_worker": 26.9, "peak_worker_rss_kib": 1400000})
        Image.new("RGB", (8, 8), "white").save(self.correction / "review_sheet.jpg", format="JPEG")
        partial_sheet_sha = stage.sha256(self.correction / "review_sheet.jpg")
        self.patches.enter_context(patch.object(candidate, "BOARD5_PARTIAL_SUPERVISOR_SHA", self.supervisor_sha))
        self.patches.enter_context(patch.object(candidate, "ROI_V2_RUNNER_SHA",
            stage.sha256(candidate.ROOT / "scripts/object_motion/sam_m1_recipe_trial.py")))
        self.patches.enter_context(patch.object(candidate, "ROI_V2_RECIPE_HELPER_SHA",
            stage.sha256(candidate.ROOT / "scripts/object_motion/sam_prompt_recipe.py")))
        self.audit = self.root / "resource-audit.json"
        self.audit_sha = write_json(self.audit, {
            "schema": "sam21_partial_parent_resource_audit_v1", "status": "accepted",
            "parent_manifest_sha256": self.partial_sha,
            "parent_supervisor_sha256": self.supervisor_sha,
            "parent_manifest_status": "partial_unusable", "parent_supervisor_status": "failed",
            "parent_supervisor_reason": "worker exit 1", "selected_names": list(candidate.PARTIAL_NAMES),
            "resource_caps_not_triggered": True, "seconds_total": 27.7,
            "seconds_worker": 26.9, "peak_worker_rss_kib": 1400000,
            "reviewer": "fixture", "reviewed_at": "2026-09-27"})
        photos = {r["path"].split("/")[-1]: r["sha256"] for r in json.loads(self.package.read_text())["training_inputs"]}
        partial_rows = [r for r in manifest["images"] if r["name"] in candidate.PARTIAL_NAMES]
        self.partial_review = self.root / "partial-review.json"
        self.partial_review_sha = write_json(self.partial_review, {
            "schema": "sam21_mask_artifact_subset_qa_v1", "status": "accepted",
            "decision": "accepted_for_coarse_pose_support",
            "parent_manifest_sha256": self.partial_sha,
            "parent_supervisor_sha256": self.supervisor_sha,
            "parent_manifest_status": "partial_unusable", "parent_supervisor_status": "failed",
            "parent_supervisor_reason": "worker exit 1", "reviewed_names": list(candidate.PARTIAL_NAMES),
            "source_image_sha256": {r["name"]: photos[r["name"]] for r in partial_rows},
            "raw_mask_sha256": {r["name"]: r["raw_mask_sha256"] for r in partial_rows},
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in partial_rows},
            "resource_audit_sha256": self.audit_sha,
            "review_sheet_sha256": partial_sheet_sha, "reviewer": "fixture", "reviewed_at": "2026-09-27"})
        self.roi = self.root / "roi-v2"
        self.roi.mkdir()
        self.roi_recipe = self.root / "roi-recipe.json"
        recipe_rows = []
        roi_rows = []
        for name in candidate.ROI_NAMES:
            points = [[1, 1, 1], [2, 2, 0], [3, 3, 0], [3, 0, 0]]
            recipe_rows.append({"name": name, "source_sha256": photos[name],
                                "points_xy_label": points})
            raw_rel, clean_rel = f"raw_masks/{name}.png", f"cleaned_masks/{name}.png"
            roi_rows.append({"name": name, "source_sha256": photos[name],
                "status": "complete_unreviewed", "box_xyxy_original_pixels": [0, 0, 3, 3],
                "points_xy_label": points, "raw_mask_path": raw_rel,
                "cleaned_mask_path": clean_rel,
                "raw_mask_sha256": mask(self.roi / raw_rel, (2, 1)),
                "cleaned_mask_sha256": mask(self.roi / clean_rel, (2, 1)),
                "recipe_row_sha256": candidate.digest_object({"name": name,
                    "source_sha256": photos[name], "box_xyxy_original_pixels": [0, 0, 3, 3],
                    "points_xy_label": points})})
        self.roi_recipe_sha = write_json(self.roi_recipe, {
            "schema": "sam21_m1_prompt_recipe_v2", "box_xyxy_original_pixels": [0, 0, 3, 3],
            "package_sha256": self.package_sha, "original_prompt_sha256": stage.PROMPTS_SHA256,
            "base_inventory_sha256": self.base_sha, "rejected_full_qa_sha256": self.rejected_sha,
            "checkpoint_sha256": point.MODEL_SHA, "source_inventory_sha256": point.SOURCE_SHA,
            "parent_board5_manifest_sha256": self.partial_sha,
            "parent_board5_supervisor_sha256": self.supervisor_sha, "images": recipe_rows})
        self.roi_manifest_sha = write_json(self.roi / "manifest.json", {
            "schema": "sam21_mustard_m1_recipe_trial_v2", "status": "generated_unreviewed",
            "package_sha256": self.package_sha, "base_inventory_sha256": self.base_sha,
            "rejected_full_qa_sha256": self.rejected_sha,
            "checkpoint_sha256": point.MODEL_SHA, "source_inventory_sha256": point.SOURCE_SHA,
            "runner_sha256": candidate.ROI_V2_RUNNER_SHA,
            "recipe_helper_sha256": candidate.ROI_V2_RECIPE_HELPER_SHA,
            "recipe_sha256": self.roi_recipe_sha,
            "parent_board5_manifest_sha256": self.partial_sha,
            "parent_board5_supervisor_sha256": self.supervisor_sha, "images": roi_rows})
        self.roi_supervisor_sha = write_json(self.roi / "supervisor.json", {
            "schema": "sam21_mustard_m1_recipe_supervisor_v2",
            "status": "generated_unreviewed", "reason": None})
        Image.new("RGB", (8, 8), "blue").save(self.roi / "review_sheet.jpg", format="JPEG")
        roi_sheet_sha = stage.sha256(self.roi / "review_sheet.jpg")
        self.roi_review = self.root / "roi-review.json"
        self.roi_review_sha = write_json(self.roi_review, {
            "schema": "sam21_mask_artifact_subset_qa_v1", "status": "accepted",
            "decision": "accepted_for_coarse_pose_support",
            "parent_manifest_sha256": self.roi_manifest_sha,
            "parent_supervisor_sha256": self.roi_supervisor_sha,
            "parent_manifest_status": "generated_unreviewed",
            "parent_supervisor_status": "generated_unreviewed", "parent_supervisor_reason": None,
            "reviewed_names": list(candidate.ROI_NAMES),
            "source_image_sha256": {r["name"]: photos[r["name"]] for r in roi_rows},
            "raw_mask_sha256": {r["name"]: r["raw_mask_sha256"] for r in roi_rows},
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in roi_rows},
            "resource_audit_sha256": None, "review_sheet_sha256": roi_sheet_sha,
            "reviewer": "fixture", "reviewed_at": "2026-09-27"})
        selected = []
        for name in candidate.CORRECTED:
            partial = name in candidate.PARTIAL_NAMES
            parent = self.correction if partial else self.roi
            recipe = self.prompts if partial else self.roi_recipe
            recipe_sha = self.prompts_sha if partial else self.roi_recipe_sha
            box = [0, 0, 4, 4] if partial else [0, 0, 3, 3]
            parent_row = next(r for r in (partial_rows if partial else roi_rows) if r["name"] == name)
            selected.append({"name": name, "origin": "board5_partial3" if partial else "roi_v2_2",
                "source_sha256": photos[name], "parent_root": str(parent),
                "parent_manifest_sha256": self.partial_sha if partial else self.roi_manifest_sha,
                "parent_supervisor_sha256": self.supervisor_sha if partial else self.roi_supervisor_sha,
                "recipe_path": str(recipe), "recipe_sha256": recipe_sha,
                "recipe_row_sha256": candidate.digest_object({"name": name,
                    "source_sha256": photos[name], "box_xyxy_original_pixels": box,
                    "points_xy_label": parent_row["points_xy_label"]}),
                "review_path": str(self.partial_review if partial else self.roi_review),
                "review_sha256": self.partial_review_sha if partial else self.roi_review_sha,
                "resource_audit_path": str(self.audit) if partial else None,
                "resource_audit_sha256": self.audit_sha if partial else None})
        self.selection = self.root / "selection.json"
        self.selection_sha = write_json(self.selection, {
            "schema": "sam21_mustard_candidate_selection_v2", "status": "reviewed_per_view_not_full48",
            "package_sha256": self.package_sha, "base_inventory_sha256": self.base_sha,
            "rejected_full_qa_sha256": self.rejected_sha, "images": selected})
        self.output_v2 = self.root / "candidate-v2"

    def options(self):
        return {"output": self.output_v2, "base_root": self.base, "mount": self.root,
                "package_path": self.package, "train_photos": self.photos,
                "rejected_qa": self.rejected,
                "expected_size": (4, 4), "require_mount": False}

    def test_preflight_discloses_partial_parent_and_preserves_43(self):
        report = candidate.preflight_selection(self.selection, self.selection_sha, **self.options())
        self.assertEqual([r["origin"] for r in report["images"]].count("base43"), 43)
        self.assertEqual([r["origin"] for r in report["images"]].count("board5_partial3"), 3)
        self.assertEqual([r["origin"] for r in report["images"]].count("roi_v2_2"), 2)
        self.assertEqual(report["parent_status_disclosure"]["board5_partial3_supervisor_status"], "failed")

    def test_failed_parent_not_relabelled_and_subset_qa_required(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["status"] = "generated_unreviewed"
        write_json(self.manifest, manifest)
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())
        manifest["status"] = "partial_unusable"
        self.assertEqual(write_json(self.manifest, manifest), self.partial_sha)
        review = json.loads(self.partial_review.read_text())
        review["status"] = "pending"
        write_json(self.partial_review, review)
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())

    def test_changed_recipe_or_selected_png_rejected(self):
        recipe = json.loads(self.roi_recipe.read_text())
        recipe["images"][0]["points_xy_label"][3] = [0, 0, 0]
        write_json(self.roi_recipe, recipe)
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())
        recipe["images"][0]["points_xy_label"][3] = [3, 0, 0]
        self.assertEqual(write_json(self.roi_recipe, recipe), self.roi_recipe_sha)
        mask(self.correction / "cleaned_masks" / (candidate.PARTIAL_NAMES[0] + ".png"), (3, 1))
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())

    def test_review_sheet_and_roi_runner_tamper_rejected(self):
        (self.roi / "review_sheet.jpg").write_bytes(b"changed after visual review")
        with self.assertRaisesRegex(ValueError, "visual-review sheet"):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())
        Image.new("RGB", (8, 8), "blue").save(self.roi / "review_sheet.jpg", format="JPEG")
        roi = json.loads((self.roi / "manifest.json").read_text())
        roi["runner_sha256"] = "0" * 64
        write_json(self.roi / "manifest.json", roi)
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())
        roi["runner_sha256"] = candidate.ROI_V2_RUNNER_SHA
        roi["recipe_helper_sha256"] = "0" * 64
        write_json(self.roi / "manifest.json", roi)
        with self.assertRaises(ValueError):
            candidate.preflight_selection(self.selection, self.selection_sha, **self.options())

    def test_composition_still_requires_independent_full48_qa(self):
        inventory = candidate.compose_selection(self.selection, self.selection_sha, **self.options())
        self.assertEqual(inventory["schema"], "sam21_mustard_m1_composed_candidate_v2")
        self.assertEqual(stage.sha256(self.output_v2 / "full48-contact-sheet.png"),
                         inventory["contact_sheet_sha256"])
        full_qa = self.root / "full48-qa.json"
        full_qa_sha = write_json(full_qa, {"schema": "mustard_sam_composed_candidate_qa_v2",
            "status": "pending", "decision": "pending",
            "inventory_sha256": stage.sha256(self.output_v2 / "complete_inventory.json"),
            "contact_sheet_sha256": inventory["contact_sheet_sha256"],
            "reviewed_names": list(stage.TRAIN_NAMES),
            "cleaned_mask_sha256": {r["name"]: r["cleaned_mask_sha256"] for r in inventory["images"]}})
        params = (self.package, self.photos, self.output_v2 / "complete_inventory.json",
                  self.output_v2, stage.sha256(self.output_v2 / "complete_inventory.json"),
                  full_qa, full_qa_sha)
        opts = {"output": self.root / "stage", "mount": self.root, "require_mount": False,
                "package_sha256": self.package_sha, "expected_size": (4, 4)}
        with self.assertRaisesRegex(ValueError, "full-48 v2"):
            stage.preflight(*params, **opts)
        qa = json.loads(full_qa.read_text())
        qa.update(status="accepted", decision="accepted_for_coarse_pose_support",
                  reviewer="fixture", reviewed_at="2026-09-27")
        full_qa_sha = write_json(full_qa, qa)
        params = (*params[:-1], full_qa_sha)
        self.assertEqual(stage.preflight(*params, **opts)["inventory_schema"],
                         "sam21_mustard_m1_composed_candidate_v2")
        (self.output_v2 / "full48-contact-sheet.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "contact sheet"):
            stage.preflight(*params, **opts)


if __name__ == "__main__":
    unittest.main()
