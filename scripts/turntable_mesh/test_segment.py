"""Frozen photo prompts, component semantics, and sealed inference receipts."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    import scipy  # noqa: F401
except ImportError:  # the core test job runs without the scientific stack
    raise unittest.SkipTest("needs SciPy") from None

from scripts.turntable_mesh import segment


class SegmentTests(unittest.TestCase):
    def fixture(self, root):
        images, masks, source = root / "images", root / "coarse", root / "sam"
        images.mkdir()
        masks.mkdir()
        (source / "sam2/configs/sam2.1").mkdir(parents=True)
        (source / "sam2/build_sam.py").write_text("# test model source\n")
        (source / "sam2" / segment.MODEL_CONFIG).write_text("model: tiny\n")
        checkpoint = root / "tiny.pt"
        checkpoint.write_bytes(b"test weights")
        arrays = []
        for index, bounds in enumerate([(10, 12, 50, 55), (25, 15, 60, 50)]):
            name = f"capture_{index:04d}.png"
            Image.new("RGB", (80, 64), (30, 60, 90)).save(images / name)
            values = np.zeros((64, 80), np.uint8)
            x0, y0, x1, y1 = bounds
            values[y0:y1, x0:x1] = 255
            Image.fromarray(values).save(masks / (name + ".png"))
            arrays.append(values)
        return images, masks, source, checkpoint, arrays

    def board_fixture(self, root):
        try:
            from scripts.turntable_mesh.test_board_feedback import BoardFeedbackTests
        except ImportError:  # the board route is a separate, optional set of modules
            self.skipTest("board feedback modules are not available")

        images, coarse, _, cameras, calibration, _, support = (
            BoardFeedbackTests().fixture(root)
        )
        source = root / "sam"
        (source / "sam2/configs/sam2.1").mkdir(parents=True)
        (source / "sam2/build_sam.py").write_text("# test model source\n")
        (source / "sam2" / segment.MODEL_CONFIG).write_text("model: tiny\n")
        checkpoint = root / "tiny.pt"
        checkpoint.write_bytes(b"test weights")
        return images, coarse, cameras, calibration, source, checkpoint, support > 0

    def test_bounded_distance_equals_full_frame_at_edges_holes_and_islands(self):
        from scipy import ndimage

        for variant in range(8):
            mask = np.zeros((64, 96), bool)
            mask[:35, :45] = True
            if variant & 1:
                mask[20:22, 10:12] = False
            if variant & 2:
                mask[35:, 70:] = True
            if variant & 4:
                mask = np.fliplr(mask)
            yy, xx = np.nonzero(mask)
            bounds = [
                int(xx.min()),
                int(yy.min()),
                int(xx.max()) + 1,
                int(yy.max()) + 1,
            ]
            expected = ndimage.distance_transform_edt(mask)
            actual = segment.foreground_distance(mask, bounds)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(actual.argmax()), int(expected.argmax()))

    def test_board_feedback_reuses_embedding_and_preserves_baseline(self):
        for feedback in (False, True):
            with (
                self.subTest(feedback=feedback),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                images, coarse, cameras, calibration, source, checkpoint, support = (
                    self.board_fixture(root)
                )
                baseline = support.copy()
                if feedback:
                    baseline[30:100, 30:100] = True
                calls, embeddings = [], []

                class Predictor:
                    def set_image(self, rgb):
                        embeddings.append(rgb)

                    def predict(self, **kwargs):
                        calls.append(kwargs)
                        mask = baseline if len(calls) == 1 else support
                        return mask[None], np.array([0.75]), None

                with patch.object(
                    segment,
                    "_load_predictor",
                    return_value=(Predictor(), nullcontext, lambda: None, {}),
                ) as load:
                    report = segment.run(
                        images,
                        coarse,
                        root / "run",
                        source=source,
                        checkpoint=checkpoint,
                        multimask=True,
                        board_cameras=cameras,
                        board_calibration=calibration,
                        board_is_background=True,
                    )
                self.assertEqual(load.call_count, 1)
                self.assertEqual(len(embeddings), 1)
                self.assertEqual(len(calls), 2 if feedback else 1)
                self.assertEqual(
                    report["runtime_counts"],
                    {"model_load": 1, "set_image": 1, "predict": len(calls)},
                )
                self.assertEqual(
                    report["board_context_before"], report["board_context_after"]
                )
                self.assertTrue(report["geometry_read"])
                self.assertFalse(report["quality_accepted"])
                row = report["rows"][0]
                self.assertEqual(row["board_feedback"]["feedback_applied"], feedback)
                for directory, expected in (
                    ("baseline-masks", baseline),
                    ("masks", support),
                ):
                    with Image.open(
                        root / "run" / directory / (row["name"] + ".png")
                    ) as image:
                        np.testing.assert_array_equal(np.array(image) > 0, expected)
                if feedback:
                    self.assertEqual(len(calls[1]["point_labels"]), 9)
                    self.assertEqual(calls[1]["point_labels"].tolist(), [1] + [0] * 8)

    def test_board_feedback_failed_points_preserve_both_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, coarse, cameras, calibration, source, checkpoint, support = (
                self.board_fixture(root)
            )
            baseline = support.copy()
            baseline[30:100, 30:100] = True

            class Predictor:
                def set_image(self, rgb):
                    pass

                def predict(self, **kwargs):
                    return baseline[None], np.array([0.75]), None

            output = root / "run"
            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ):
                with self.assertRaisesRegex(ValueError, "feedback candidate"):
                    segment.run(
                        images,
                        coarse,
                        output,
                        source=source,
                        checkpoint=checkpoint,
                        multimask=True,
                        board_cameras=cameras,
                        board_calibration=calibration,
                        board_is_background=True,
                    )
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["rows"][0]["status"], "failed")
            self.assertEqual(
                report["rows"][0]["baseline"]["status"], "complete_unreviewed"
            )
            self.assertFalse(report["rows"][0]["selection_passed"])
            self.assertTrue((output / "baseline-masks/capture_0000.png.png").is_file())
            self.assertTrue((output / "masks/capture_0000.png.png").is_file())
            self.assertEqual(
                report["board_context_before"], report["board_context_after"]
            )

    def test_board_feedback_camera_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, coarse, cameras, calibration, source, checkpoint, support = (
                self.board_fixture(root)
            )

            class Predictor:
                def set_image(self, rgb):
                    cameras.write_text(cameras.read_text() + "\n")

                def predict(self, **kwargs):
                    return support[None], np.array([0.75]), None

            output = root / "run"
            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ):
                with self.assertRaisesRegex(
                    ValueError, "context implementation changed"
                ):
                    segment.run(
                        images,
                        coarse,
                        output,
                        source=source,
                        checkpoint=checkpoint,
                        multimask=True,
                        board_cameras=cameras,
                        board_calibration=calibration,
                        board_is_background=True,
                    )
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("provenance_error", report)

    def test_board_feedback_requires_explicit_unmixed_baseline_prior(self):
        for arguments in (
            {"board_cameras": Path("cameras.json")},
            {"board_is_background": "yes"},
            {
                "board_cameras": Path("cameras.json"),
                "board_calibration": Path("lens.json"),
                "board_is_background": True,
                "multimask": False,
            },
            {
                "board_cameras": Path("cameras.json"),
                "board_calibration": Path("lens.json"),
                "board_is_background": True,
                "multimask": True,
                "automatic_cues": True,
            },
            {
                "board_cameras": Path("cameras.json"),
                "board_calibration": Path("lens.json"),
                "board_is_background": True,
                "multimask": True,
                "prompts_json": Path("points.json"),
            },
        ):
            with (
                self.subTest(arguments=arguments),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                with patch.object(segment, "_load_predictor") as load:
                    with self.assertRaises(ValueError):
                        segment.run(
                            root,
                            root,
                            root / "run",
                            source=root,
                            checkpoint=root / "tiny.pt",
                            **arguments,
                        )
                    load.assert_not_called()

    def test_frozen_union_and_inscribed_positive_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, _, _, arrays = self.fixture(root)
            paths = segment.selected_images(images)
            prompts, dimensions = segment.frozen_prompts(paths, masks)
            self.assertEqual(dimensions, (80, 64))
            for prompt, support in zip(prompts, arrays):
                self.assertEqual(prompt["box_xyxy"], [10, 12, 60, 55])
                x, y = prompt["point_xy"]
                self.assertEqual(support[y, x], 255)
                distance = segment.ndimage.distance_transform_edt(support > 0)
                self.assertEqual(distance[y, x], distance.max())
                self.assertEqual(prompt["point_label"], 1)
            self.assertEqual(len(segment.selected_images(images, 1)), 1)
            for invalid in (-1, 256, True):
                with self.assertRaises(ValueError):
                    segment.selected_images(images, invalid)

    def test_best_prompt_component_and_explicit_hole_filling(self):
        values = np.zeros((2, 64, 80), bool)
        values[0, 10:40, 10:40] = True
        values[0, 20:25, 20:25] = False
        values[0, 50:55, 60:65] = True
        values[1, 1:8, 1:8] = True  # high-score unrelated component must lose
        raw, clean, metrics = segment.clean_prediction(
            values, [0.7, 0.99], [15, 15], (80, 64)
        )
        self.assertTrue(raw[52, 62])
        self.assertFalse(clean[52, 62])
        self.assertFalse(raw[22, 22])
        self.assertTrue(clean[22, 22])
        self.assertEqual(metrics["holes_filled_pixels"], 25)
        self.assertEqual(metrics["predicted_iou"], 0.7)
        with self.assertRaisesRegex(ValueError, "positive prompt"):
            segment.clean_prediction(values, [0.7, 0.99], [79, 63], (80, 64))

    def test_mocked_runner_seals_inputs_model_and_exact_binary_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, arrays = self.fixture(root)

            class Predictor:
                def set_image(self, rgb):
                    self.dimensions = rgb.shape

                def predict(self, **kwargs):
                    self_args.append(kwargs)
                    return arrays[len(self_args) - 1][None] > 0, np.array([0.75]), None

            self_args = []
            output = root / "run"
            prompts = root / "points.json"
            inventory = segment.input_inventory(segment.selected_images(images), masks)
            prompts.write_text(
                json.dumps(
                    {
                        "schema": "photo_only_segmentation_prompts_v1",
                        "images": [
                            {
                                "name": inventory[0]["name"],
                                "rgb_sha256": inventory[0]["rgb_sha256"],
                                "points_xy": [[20, 25], [75, 60]],
                                "point_labels": [1, 0],
                            }
                        ],
                    }
                )
            )
            with patch.object(
                segment,
                "_load_predictor",
                return_value=(
                    Predictor(),
                    nullcontext,
                    lambda: None,
                    {"torch": "mock", "device": "cpu"},
                ),
            ) as load:
                report = segment.run(
                    images,
                    masks,
                    output,
                    source=source,
                    checkpoint=checkpoint,
                    prompts_json=prompts,
                )
            self.assertEqual(report["status"], "complete_unreviewed")
            self.assertFalse(report["quality_accepted"])
            self.assertEqual(
                report["source_inputs_before"], report["source_inputs_after"]
            )
            self.assertEqual(report["model_before"], report["model_after"])
            self.assertEqual(
                report["prompt_overrides_sha256_before"],
                report["prompt_overrides_sha256_after"],
            )
            np.testing.assert_array_equal(
                self_args[0]["point_coords"], [[20, 25], [75, 60]]
            )
            np.testing.assert_array_equal(self_args[0]["point_labels"], [1, 0])
            self.assertEqual(len(self_args[1]["point_coords"]), 1)
            self.assertEqual(load.call_args.args[-1], "cpu")
            for row, original in zip(report["rows"], arrays):
                with Image.open(output / "masks" / (row["name"] + ".png")) as image:
                    self.assertEqual(image.mode, "L")
                    np.testing.assert_array_equal(np.array(image), original)
            for request in self_args:
                np.testing.assert_array_equal(request["box"], [10, 12, 60, 55])
                self.assertFalse(request["multimask_output"])
            with self.assertRaises(FileExistsError):
                segment.run(images, masks, output, source=source, checkpoint=checkpoint)

    def test_checkpoint_mutation_records_failure_and_before_after_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, arrays = self.fixture(root)

            class Predictor:
                def set_image(self, rgb):
                    checkpoint.write_bytes(b"changed weights during inference")

                def predict(self, **kwargs):
                    return arrays[0][None] > 0, np.array([0.5]), None

            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ):
                with self.assertRaisesRegex(ValueError, "changed during segmentation"):
                    segment.run(
                        images,
                        masks,
                        root / "run",
                        source=source,
                        checkpoint=checkpoint,
                    )
            report = json.loads((root / "run/result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertNotEqual(
                report["model_before"]["checkpoint_sha256"],
                report["model_after"]["checkpoint_sha256"],
            )

    def test_unavailable_mps_has_no_cpu_fallback(self):
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
                segment._load_predictor(
                    Path("sam"), Path("weights.pt"), segment.MODEL_CONFIG, "mps"
                )

    def test_override_hash_membership_bounds_and_labels_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, _, _, _ = self.fixture(root)
            paths = segment.selected_images(images)
            inventory = segment.input_inventory(paths, masks)
            path = root / "prompts.json"
            good = {
                "name": inventory[0]["name"],
                "rgb_sha256": inventory[0]["rgb_sha256"],
                "points_xy": [[20, 25]],
                "point_labels": [1],
            }
            for changed in [
                dict(good, rgb_sha256="wrong"),
                dict(good, name="../foreign.png"),
                dict(good, points_xy=[[80, 25]]),
                dict(good, points_xy=[[20.5, 25]]),
                dict(good, point_labels=[True]),
                dict(good, point_labels=[0]),
            ]:
                path.write_text(
                    json.dumps(
                        {
                            "schema": "photo_only_segmentation_prompts_v1",
                            "images": [changed],
                        }
                    )
                )
                prompts, dimensions = segment.frozen_prompts(paths, masks)
                with self.assertRaises(ValueError):
                    segment.apply_prompt_overrides(prompts, path, inventory, dimensions)

    def test_cleanup_cannot_discard_positive_or_fill_explicit_negative(self):
        values = np.zeros((1, 64, 80), bool)
        values[0, 10:40, 10:40] = True
        values[0, 20:25, 20:25] = False
        values[0, 50:55, 60:65] = True
        _, clean, _ = segment.clean_prediction(values, [0.8], [15, 15], (80, 64))
        for points, labels in [
            ([[15, 15], [62, 52]], [1, 1]),
            ([[15, 15], [22, 22]], [1, 0]),
        ]:
            with self.assertRaisesRegex(ValueError, "explicit positive/negative"):
                segment.verify_cleaned_points(
                    clean,
                    {"point_xy": [15, 15], "points_xy": points, "point_labels": labels},
                )

    def test_preserve_holes_retains_donut_and_discards_unconnected_island(self):
        values = np.zeros((1, 64, 80), bool)
        values[0, 10:40, 10:40] = True
        values[0, 20:25, 20:25] = False
        values[0, 50:55, 60:65] = True
        prompt = {
            "point_xy": [15, 15],
            "points_xy": [[15, 15], [22, 22]],
            "point_labels": [1, 0],
        }
        raw, default, default_report = segment.select_prediction(
            values, [0.8], prompt, (80, 64)
        )
        _, explicit_default, _ = segment.clean_prediction(
            values, [0.8], [15, 15], (80, 64), preserve_holes=False
        )
        np.testing.assert_array_equal(default, explicit_default)
        self.assertFalse(default_report["selection_passed"])
        self.assertEqual(default_report["holes_filled_pixels"], 25)
        _, preserved, report = segment.select_prediction(
            values, [0.8], prompt, (80, 64), preserve_holes=True
        )
        self.assertTrue(report["selection_passed"])
        self.assertEqual(report["holes_filled_pixels"], 0)
        self.assertFalse(preserved[22, 22])
        self.assertFalse(preserved[52, 62])
        self.assertTrue(raw[52, 62])
        expected = values[0].copy()
        expected[50:55, 60:65] = False
        np.testing.assert_array_equal(preserved, expected)
        for invalid in (0, 1, None, "yes"):
            with self.assertRaisesRegex(ValueError, "explicit boolean"):
                segment.clean_prediction(
                    values, [0.8], [15, 15], (80, 64), preserve_holes=invalid
                )

    def test_preserve_holes_run_policy_and_boolean_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, arrays = self.fixture(root)
            inventory = segment.input_inventory(segment.selected_images(images), masks)
            prompts = root / "points.json"
            prompts.write_text(
                json.dumps(
                    {
                        "schema": "photo_only_segmentation_prompts_v1",
                        "images": [
                            {
                                "name": inventory[0]["name"],
                                "rgb_sha256": inventory[0]["rgb_sha256"],
                                "points_xy": [[20, 25], [31, 26]],
                                "point_labels": [1, 0],
                            }
                        ],
                    }
                )
            )
            raw = arrays[0] > 0
            raw[25:28, 30:33] = False

            class Predictor:
                def set_image(self, rgb):
                    pass

                def predict(self, **kwargs):
                    return raw[None], np.array([0.9]), None

            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ) as load:
                with self.assertRaisesRegex(ValueError, "explicit boolean"):
                    segment.run(
                        images,
                        masks,
                        root / "invalid",
                        source=source,
                        checkpoint=checkpoint,
                        preserve_holes=1,
                    )
                load.assert_not_called()
                self.assertFalse((root / "invalid").exists())
                report = segment.run(
                    images,
                    masks,
                    root / "preserved",
                    source=source,
                    checkpoint=checkpoint,
                    prompts_json=prompts,
                    preserve_holes=True,
                )
            self.assertEqual(report["status"], "complete_unreviewed")
            self.assertTrue(report["configuration"]["preserve_holes"])
            self.assertIn("preserve background holes", report["cleanup"])
            self.assertEqual(report["rows"][0]["holes_filled_pixels"], 0)
            with Image.open(
                root / "preserved/masks" / (inventory[0]["name"] + ".png")
            ) as image:
                self.assertEqual(np.array(image)[26, 31], 0)

    def test_failed_membership_preserves_both_candidate_masks_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, arrays = self.fixture(root)
            raw = arrays[0] > 0
            raw[25:28, 30:33] = False
            inventory = segment.input_inventory(segment.selected_images(images), masks)
            prompts = root / "points.json"
            prompts.write_text(
                json.dumps(
                    {
                        "schema": "photo_only_segmentation_prompts_v1",
                        "images": [
                            {
                                "name": inventory[0]["name"],
                                "rgb_sha256": inventory[0]["rgb_sha256"],
                                "points_xy": [[20, 25], [31, 26]],
                                "point_labels": [1, 0],
                            }
                        ],
                    }
                )
            )

            class Predictor:
                def set_image(self, rgb):
                    pass

                def predict(self, **kwargs):
                    return raw[None], np.array([0.9]), None

            output = root / "run"
            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ):
                with self.assertRaisesRegex(ValueError, "explicit positive/negative"):
                    segment.run(
                        images,
                        masks,
                        output,
                        source=source,
                        checkpoint=checkpoint,
                        prompts_json=prompts,
                    )
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["rows"]), 1)
            row = report["rows"][0]
            self.assertEqual(row["status"], "failed")
            self.assertTrue(row["raw_point_membership"][1]["passed"])
            self.assertFalse(row["clean_point_membership"][1]["passed"])
            self.assertEqual(row["holes_filled_pixels"], 9)
            with Image.open(output / "raw-masks" / (row["name"] + ".png")) as image:
                self.assertEqual(np.array(image)[26, 31], 0)
            with Image.open(output / "masks" / (row["name"] + ".png")) as image:
                self.assertEqual(np.array(image)[26, 31], 255)

    def test_multimask_selects_lower_score_valid_photo_cues_and_reports_failures(self):
        masks = np.zeros((2, 64, 80), bool)
        masks[0, 10:40, 10:55] = True
        masks[1, 10:40, 10:40] = True
        prompt = {
            "point_xy": [15, 15],
            "points_xy": [[15, 15], [45, 25]],
            "point_labels": [1, 0],
        }
        _, clean, report = segment.select_prediction(
            masks, [0.99, 0.65], prompt, (80, 64)
        )
        self.assertEqual(report["selected_index"], 1)
        self.assertEqual(report["predicted_iou"], 0.65)
        self.assertEqual(report["candidate_count"], 2)
        self.assertTrue(report["selection_passed"])
        self.assertFalse(report["candidates"][0]["passed"])
        self.assertTrue(report["candidates"][1]["passed"])
        self.assertFalse(clean[25, 45])
        masks[1] = masks[0]
        raw, clean, report = segment.select_prediction(
            masks, [0.99, 0.65], prompt, (80, 64)
        )
        self.assertFalse(report["selection_passed"])
        self.assertEqual(report["selected_index"], 0)
        self.assertTrue(raw[25, 45])
        self.assertTrue(clean[25, 45])
        self.assertTrue(all(not row["passed"] for row in report["candidates"]))

    def test_explicit_multimask_reaches_predictor_and_selected_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, arrays = self.fixture(root)
            requests = []

            class Predictor:
                def set_image(self, rgb):
                    pass

                def predict(self, **kwargs):
                    requests.append(kwargs)
                    mask = arrays[len(requests) - 1] > 0
                    return np.stack([mask, mask]), np.array([0.6, 0.8]), None

            with patch.object(
                segment,
                "_load_predictor",
                return_value=(Predictor(), nullcontext, lambda: None, {}),
            ):
                report = segment.run(
                    images,
                    masks,
                    root / "run",
                    source=source,
                    checkpoint=checkpoint,
                    multimask=True,
                    automatic_cues=True,
                )
            self.assertTrue(report["configuration"]["multimask_output"])
            self.assertTrue(report["configuration"]["automatic_cues"])
            self.assertIn("Object lies inside", report["automatic_cue_prior"])
            self.assertTrue(all(request["multimask_output"] for request in requests))
            self.assertTrue(
                all(len(request["point_coords"]) > 1 for request in requests)
            )
            self.assertTrue(
                all(
                    row["candidate_count"] == 2 and row["selected_index"] == 1
                    for row in report["rows"]
                )
            )

    def test_automatic_positive_cues_have_clearance_and_spatial_coverage(self):
        support = np.zeros((200, 220), bool)
        support[60:140, 70:150] = True
        distance = segment.ndimage.distance_transform_edt(support)
        y, x = np.unravel_index(distance.argmax(), support.shape)
        points, clearance = segment.interior_positive_points(
            support, distance, [int(x), int(y)]
        )
        self.assertEqual(len(points), 5)
        self.assertEqual(len({tuple(point) for point in points}), 5)
        for px, py in points[1:]:
            self.assertGreaterEqual(distance[py, px], clearance)
            self.assertGreaterEqual(np.linalg.norm(np.array([px - x, py - y])), 16)
        self.assertEqual(
            {(px >= 110, py >= 100) for px, py in points[1:]},
            {(False, False), (False, True), (True, False), (True, True)},
        )
        thin = np.zeros_like(support)
        thin[100, 60:160] = True
        thin_distance = segment.ndimage.distance_transform_edt(thin)
        points, _ = segment.interior_positive_points(thin, thin_distance, [60, 100])
        self.assertEqual(points, [[60, 100]])

    def test_automatic_background_cues_guard_common_box_and_skip_edges(self):
        box = [70, 60, 150, 140]
        points, margin = segment.exterior_negative_points(box, (220, 200))
        self.assertEqual(margin, 24)
        self.assertEqual(len(points), 8)
        for x, y in points:
            self.assertTrue(
                x < box[0] - margin
                or x >= box[2] + margin
                or y < box[1] - margin
                or y >= box[3] + margin
            )
            self.assertTrue(0 <= x < 220 and 0 <= y < 200)
        points, _ = segment.exterior_negative_points([0, 0, 220, 200], (220, 200))
        self.assertEqual(points, [])

    def test_automatic_option_default_prompts_and_override_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, masks, source, checkpoint, _ = self.fixture(root)
            paths = segment.selected_images(images)
            before, _ = segment.frozen_prompts(paths, masks)
            explicit_default, _ = segment.frozen_prompts(
                paths, masks, automatic_cues=False
            )
            self.assertEqual(before, explicit_default)
            self.assertTrue(
                all(
                    set(prompt) == {"name", "point_xy", "point_label", "box_xyxy"}
                    for prompt in before
                )
            )
            generated, dimensions = segment.frozen_prompts(
                paths, masks, automatic_cues=True
            )
            inventory = segment.input_inventory(paths, masks)
            override = root / "points.json"
            override.write_text(
                json.dumps(
                    {
                        "schema": "photo_only_segmentation_prompts_v1",
                        "images": [
                            {
                                "name": inventory[0]["name"],
                                "rgb_sha256": inventory[0]["rgb_sha256"],
                                "points_xy": [[20, 25]],
                                "point_labels": [1],
                            }
                        ],
                    }
                )
            )
            corrected = segment.apply_prompt_overrides(
                generated, override, inventory, dimensions
            )
            self.assertEqual(corrected[0]["points_xy"], [[20, 25]])
            self.assertTrue(corrected[0]["automatic_cues_replaced_by_explicit_points"])
            self.assertGreater(len(corrected[1]["points_xy"]), 1)
            with patch.object(segment, "_load_predictor") as load:
                with self.assertRaisesRegex(
                    ValueError, "automatic_cues must be an explicit boolean"
                ):
                    segment.run(
                        images,
                        masks,
                        root / "invalid",
                        source=source,
                        checkpoint=checkpoint,
                        automatic_cues="yes",
                    )
                load.assert_not_called()
            self.assertEqual(
                json.loads((root / "invalid/result.json").read_text())["status"],
                "failed",
            )


if __name__ == "__main__":
    unittest.main()
