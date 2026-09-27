"""Hermetic tests for the generic Mac recipe trial and partial-parent seal."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import sam_m1_recipe_trial as trial


FIXTURE = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_two_view_recipe.json"
FIXTURE_SHA = "38f2a40af084df4c177925bfcfad7b5ebe5f131388283d14d15f5e690a8d0367"


class MacRecipeTrialTests(TestCase):
    def test_parent_partial_status_and_exact_directory_sets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir, clean_dir = root / "raw_masks", root / "cleaned_masks"
            raw_dir.mkdir()
            clean_dir.mkdir()
            images = []
            for name in trial.legacy.NAMES:
                raw = raw_dir / f"{name}.png"
                raw.write_bytes(name.encode())
                item = {"name": name, "raw_mask_path": f"raw_masks/{name}.png",
                        "raw_mask_sha256": trial.common.digest(raw)}
                if name in ("NP3_318.jpg", "NP3_330.jpg", "NP3_348.jpg"):
                    clean = clean_dir / f"{name}.png"
                    clean.write_bytes(name.encode())
                    item.update({"status": "complete_unreviewed",
                                 "cleaned_mask_path": f"cleaned_masks/{name}.png",
                                 "cleaned_mask_sha256": trial.common.digest(clean)})
                else:
                    item["status"] = "failed_unusable"
                images.append(item)
            manifest = {"schema": "sam21_mustard_m1_board5_candidate_v1",
                        "status": "partial_unusable", "images": images}
            supervisor = {"schema": "sam21_mustard_m1_board5_supervisor_v1",
                          "status": "failed", "reason": "worker exit 1"}
            recipe = {"parent_board5_manifest_sha256": trial.PARENT_MANIFEST_SHA,
                      "parent_board5_supervisor_sha256": trial.PARENT_SUPERVISOR_SHA}

            def lineage():
                with patch.object(trial.resume, "bound_metadata",
                                  side_effect=[(manifest, trial.PARENT_MANIFEST_SHA),
                                               (supervisor, trial.PARENT_SUPERVISOR_SHA)]):
                    return trial.parent_lineage(root, recipe)

            self.assertEqual(lineage()["parent_board5_status"], "partial_unusable/failed")
            extra = raw_dir / "NP3_999.jpg.png"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "missing/extra"):
                lineage()
            extra.unlink()
            clean_dir.rename(root / "old_clean")
            clean_dir.symlink_to(root / "old_clean", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "missing/extra"):
                lineage()

    def test_fake_predictor_two_rows_and_canonical_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            args = SimpleNamespace(output=output, stage_root=root, package=root / "package",
                                   stage_report=root / "stage-report", original_prompts=root / "original-prompts",
                                   recipe=FIXTURE, recipe_sha256=FIXTURE_SHA,
                                   checkpoint=root / "checkpoint", sam_source=root / "source",
                                   base_root=root / "base", base_inventory=root / "base" / "complete_inventory.json",
                                   parent_board5=root / "parent")
            args.package.write_bytes(b"package")
            args.checkpoint.write_bytes(b"checkpoint")
            document = json.loads(FIXTURE.read_text())
            rows = [{"path": "photos/" + item["name"], "sha256": item["source_sha256"]}
                    for item in document["images"]]
            called = []

            def predict(image, box, points):
                called.append((box, points))
                mask = np.zeros((1024, 1280), np.uint8)
                mask[430:580, 550:650] = 1
                return mask

            with (patch.object(trial.mac, "host_preflight"),
                  patch.object(trial, "inputs", return_value=(rows, rows, document, "p", "s", {}, {})),
                  patch.object(trial.mac, "available_kib", return_value=3 * 1024**2),
                  patch.object(trial.mac, "rss_kib", return_value=1000),
                  patch.object(trial.mac, "disk_free_both", return_value=20 * 1024**3),
                  patch.object(trial.common, "verified_training_photo", return_value=Image.new("RGB", (1280, 1024))),
                  patch.object(trial, "dependencies", return_value={"runner_sha256": "r"}),
                  patch.object(trial.legacy, "source_seal", return_value={}),
                  patch.object(trial, "parent_lineage", return_value={}),
                  patch.object(trial.common, "load_package", return_value={"images": []}),
                  patch.object(trial.common, "training_records", return_value=rows),
                  patch.object(trial.smoke, "source_digest", return_value=trial.point.SOURCE_SHA),
                  patch.object(trial.common, "digest", side_effect=lambda p: {
                      str(args.package): "p", str(args.stage_report): "s",
                      str(args.recipe): FIXTURE_SHA,
                      str(args.original_prompts): trial.full.PROMPTS_SHA,
                      str(args.checkpoint): trial.point.MODEL_SHA}.get(str(p), "masksha"))):
                result = trial.worker(args, predictor=predict)
            self.assertEqual(result["status"], "generated_unreviewed")
            self.assertEqual(len(called), 2)
            self.assertTrue(all(box == [520, 365, 665, 600] for box, _ in called))
            self.assertEqual([r["name"] for r in result["images"]], document["selected_training_names"])
            for item, prompt in zip(result["images"], document["images"]):
                self.assertEqual(item["recipe_row_sha256"], trial.recipes.row_sha256(
                    prompt, document["box_xyxy_original_pixels"]))
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"],
                             "generated_unreviewed")

    def test_required_hash_and_runtime_caps(self):
        with self.assertRaises(ValueError):
            trial.recipes.load_v2(FIXTURE, "not-a-hash", [])
        self.assertIsNone(trial.legacy.terminal_guard(89, 239, 1000, 3 * 1024**2,
                                                       1000, 1000, 20 * 1024**3, 0))
        self.assertIsNotNone(trial.legacy.terminal_guard(91, 239, 1000, 3 * 1024**2,
                                                          1000, 1000, 20 * 1024**3, 0))
