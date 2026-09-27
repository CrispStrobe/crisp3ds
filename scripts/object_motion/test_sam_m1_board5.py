"""Hermetic contract tests for the separate five-view board-negative trial."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.object_motion import sam_m1_board5 as trial


class BoardFiveTests(TestCase):
    def test_frozen_prompts_and_fourth_negative(self):
        fixture = json.loads(trial.PROMPT_PATH.read_text())
        self.assertEqual([r["name"] for r in fixture["images"]], list(trial.NAMES))
        self.assertEqual(trial.common.digest(trial.PROMPT_PATH), trial.PROMPT_SHA)
        for row in fixture["images"]:
            xy, labels = trial.four_points(row["points_xy_label"])
            self.assertEqual(xy.tolist()[-1], [620.0, 330.0])
            self.assertEqual(labels.tolist(), [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            trial.four_points([[600, 525, 1], [720, 620, 0], [520, 630, 0]])
        with self.assertRaises(ValueError):
            trial.four_points([[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, True]])

    def test_membership_rejects_board_foreground(self):
        points = [[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, 0]]
        mask = np.zeros((1024, 1280), np.uint8)
        mask[525, 600] = 255
        self.assertEqual(trial.membership(mask, points), [1, 0, 0, 0])
        mask[330, 620] = 255
        with self.assertRaisesRegex(ValueError, "four-point"):
            trial.membership(mask, points)

    def test_base_inventory_rejects_missing_extra_and_tamper(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"path": f"photos/{name}", "sha256": f"{index:064x}"}
                    for index, name in enumerate((*trial.NAMES, "NP3_000.jpg"), 1)]
            base = root / "base"
            base.mkdir()
            inventory_path = base / "complete_inventory.json"
            images = []
            hashes = {}
            for row in rows:
                name = Path(row["path"]).name
                path = base / "frames" / name / "clean.png"
                path.parent.mkdir(parents=True)
                path.write_bytes(name.encode())
                digest = trial.common.digest(path)
                hashes[name] = digest
                images.append({"name": name, "source_sha256": row["sha256"],
                               "cleaned_mask_path": f"frames/{name}/clean.png",
                               "cleaned_mask_sha256": digest})
            inv = {"schema": "sam21_mustard_point_mask_inventory_v1", "status": "generated_unreviewed",
                   "package_sha256": "p", "prompt_sha256": trial.full.PROMPTS_SHA, "images": images}
            qa = {"schema": "mustard_sam_mask_qa_v1", "status": "rejected",
                  "decision": "rejected_checkerboard_leakage", "inventory_sha256": trial.INVENTORY_SHA,
                  "rejected_names": list(trial.NAMES), "cleaned_mask_sha256": hashes}
            inventory_path.write_text(json.dumps(inv))
            with patch.object(trial.resume, "bound_metadata", side_effect=[(inv, trial.INVENTORY_SHA),
                                                                       (qa, trial.QA_SHA)]):
                self.assertEqual(trial.source_seal(base, inventory_path, rows, "p"), hashes)
            bad = dict(inv, images=images[:-1])
            with patch.object(trial.resume, "bound_metadata", side_effect=[(bad, trial.INVENTORY_SHA),
                                                                       (qa, trial.QA_SHA)]):
                with self.assertRaisesRegex(ValueError, "missing/extra"):
                    trial.source_seal(base, inventory_path, rows, "p")
            (base / "frames" / trial.NAMES[0] / "clean.png").write_bytes(b"tamper")
            with patch.object(trial.resume, "bound_metadata", side_effect=[(inv, trial.INVENTORY_SHA),
                                                                       (qa, trial.QA_SHA)]):
                with self.assertRaisesRegex(ValueError, "differs"):
                    trial.source_seal(base, inventory_path, rows, "p")

    def test_fake_predictor_receives_four_points_and_preserves_partial(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "candidate"
            rows = [{"path": f"photos/{name}", "sha256": f"{i:064x}"}
                    for i, name in enumerate(trial.NAMES, 1)]
            points = [[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, 0]]
            prompts = [{"name": name, "source_sha256": row["sha256"], "points_xy_label": points}
                       for name, row in zip(trial.NAMES, rows)]
            args = SimpleNamespace(output=output, package=root / "package", stage_root=root,
                                   stage_report=root / "stage-report", original_prompts=root / "old-prompts",
                                   checkpoint=root / "checkpoint", sam_source=root / "source",
                                   base_root=root / "base", base_inventory=root / "base" / "complete_inventory.json",
                                   correction_prompts=trial.PROMPT_PATH)
            args.package.write_bytes(b"package")
            args.checkpoint.write_bytes(b"checkpoint")
            baseline = {name: "x" * 64 for name in trial.NAMES}
            calls = []

            def fake_predictor(image, box, actual_points):
                calls.append((tuple(box), actual_points))
                mask = np.zeros((1024, 1280), np.uint8)
                mask[425:590, 555:650] = 1  # includes positive, excludes all three negatives
                return mask

            with (patch.object(trial.mac, "host_preflight"),
                  patch.object(trial, "validated_inputs", return_value=(rows, prompts, "p", "s", trial.PROMPT_SHA, baseline)),
                  patch.object(trial.mac, "available_kib", return_value=3 * 1024**2),
                  patch.object(trial.mac, "rss_kib", return_value=1000),
                  patch.object(trial.mac, "disk_free_both", return_value=20 * 1024**3),
                  patch.object(trial.common, "verified_training_photo", return_value=Image.new("RGB", (1280, 1024))),
                  patch.object(trial.smoke, "source_digest", return_value=trial.point.SOURCE_SHA),
                  patch.object(trial, "source_seal", return_value=baseline),
                  patch.object(trial.common, "load_package", return_value={"images": []}),
                  patch.object(trial.common, "training_records", return_value=rows),
                  patch.object(trial, "dependency_hashes", return_value={"runner_sha256": "r"}),
                  patch.object(trial.common, "digest", side_effect=lambda p: {
                      str(args.package): "p", str(args.checkpoint): trial.point.MODEL_SHA,
                      str(args.original_prompts): trial.full.PROMPTS_SHA,
                      str(args.stage_report): "s",
                      str(trial.PROMPT_PATH): trial.PROMPT_SHA}.get(str(p), "masksha"))):
                report = trial.worker(args, predictor=fake_predictor)
            self.assertEqual(report["status"], "generated_unreviewed")
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(box == trial.smoke.BOX and points_arg == points
                                for box, points_arg in calls))
            self.assertEqual(len(report["images"]), 5)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"],
                             "generated_unreviewed")

    def test_terminal_caps_and_launch_memory(self):
        values = (89, 200, 1000, 3 * 1024**2, 1000, 1000, 20 * 1024**3, 0)
        self.assertIsNone(trial.terminal_guard(*values))
        for position, value in ((0, 91), (1, 241), (2, trial.mac.MAX_RSS_KIB + 1),
                                (3, trial.mac.MIN_AVAILABLE_KIB - 1), (4, trial.MAX_OUTPUT + 1),
                                (5, trial.MAX_LOG + 1), (6, trial.common.MIN_FREE_BYTES - 1), (7, 1)):
            changed = list(values)
            changed[position] = value
            self.assertIsNotNone(trial.terminal_guard(*changed))
        self.assertEqual(trial.MIN_LAUNCH_KIB, 4 * 1024**2)

    def test_one_board_foreground_keeps_raw_but_not_complete_mask(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "candidate"
            rows = [{"path": f"photos/{name}", "sha256": f"{i:064x}"}
                    for i, name in enumerate(trial.NAMES, 1)]
            points = [[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, 0]]
            prompts = [{"name": name, "source_sha256": row["sha256"], "points_xy_label": points}
                       for name, row in zip(trial.NAMES, rows)]
            args = SimpleNamespace(output=output, package=root / "package", stage_root=root,
                                   stage_report=root / "stage-report", original_prompts=root / "old-prompts",
                                   checkpoint=root / "checkpoint", sam_source=root / "source",
                                   base_root=root / "base", base_inventory=root / "base" / "complete_inventory.json",
                                   correction_prompts=trial.PROMPT_PATH)
            args.package.write_bytes(b"package")
            args.checkpoint.write_bytes(b"checkpoint")
            calls = 0

            def predict(image, box, actual_points):
                nonlocal calls
                calls += 1
                mask = np.zeros((1024, 1280), np.uint8)
                mask[425:590, 555:650] = 1
                if calls == 3:
                    mask[330, 620] = 1
                return mask

            with (patch.object(trial.mac, "host_preflight"),
                  patch.object(trial, "validated_inputs", return_value=(rows, prompts, "p", "s", trial.PROMPT_SHA, {})),
                  patch.object(trial.mac, "available_kib", return_value=3 * 1024**2),
                  patch.object(trial.mac, "rss_kib", return_value=1000),
                  patch.object(trial.mac, "disk_free_both", return_value=20 * 1024**3),
                  patch.object(trial.common, "verified_training_photo", return_value=Image.new("RGB", (1280, 1024))),
                  patch.object(trial.smoke, "source_digest", return_value=trial.point.SOURCE_SHA),
                  patch.object(trial, "source_seal", return_value={}),
                  patch.object(trial, "dependency_hashes", return_value={"runner_sha256": "r"}),
                  patch.object(trial.common, "load_package", return_value={"images": []}),
                  patch.object(trial.common, "training_records", return_value=rows),
                  patch.object(trial.common, "digest", side_effect=lambda p: {
                      str(args.package): "p", str(args.checkpoint): trial.point.MODEL_SHA,
                      str(args.original_prompts): trial.full.PROMPTS_SHA,
                      str(args.stage_report): "s",
                      str(trial.PROMPT_PATH): trial.PROMPT_SHA}.get(str(p), "masksha"))):
                report = trial.worker(args, predictor=predict)
            self.assertEqual(report["status"], "partial_unusable")
            bad = report["images"][2]
            self.assertEqual(bad["status"], "failed_unusable")
            self.assertTrue((output / bad["raw_mask_path"]).is_file())
            self.assertNotIn("cleaned_mask_sha256", bad)
            self.assertFalse((output / "cleaned_masks" / f"{bad['name']}.png").exists())
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "partial_unusable")
