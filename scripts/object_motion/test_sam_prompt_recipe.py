"""Pure recipe contract tests; no model or external dataset required."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.object_motion import sam_prompt_recipe as recipe
from scripts.object_motion import sam_m1_board5 as board5


FIXTURE = Path(__file__).parents[2] / "tests/datasets/sam21_mustard_m1_two_view_recipe.json"
FIXTURE_SHA = "38f2a40af084df4c177925bfcfad7b5ebe5f131388283d14d15f5e690a8d0367"


class PromptRecipeTests(TestCase):
    def setUp(self):
        self.document = json.loads(FIXTURE.read_text())
        self.rows = [{"path": "photos/" + item["name"], "sha256": item["source_sha256"]}
                     for item in self.document["images"]]

    def test_approved_two_view_recipe_and_outside_box_negatives(self):
        self.assertEqual(recipe.common.digest(FIXTURE), FIXTURE_SHA)
        loaded, selected = recipe.load_v2(FIXTURE, FIXTURE_SHA, self.rows)
        self.assertEqual(loaded["selected_training_names"], ["NP3_336.jpg", "NP3_342.jpg"])
        self.assertEqual(loaded["box_xyxy_original_pixels"], [520, 365, 665, 600])
        self.assertEqual(len(selected), 2)
        for item in loaded["images"]:
            self.assertEqual(item["points_xy_label"],
                             [[600, 525, 1], [720, 620, 0], [520, 630, 0], [620, 330, 0]])
            self.assertTrue(recipe.validate_geometry(loaded["box_xyxy_original_pixels"],
                                                     item["points_xy_label"], 2))
            # The negative prompts are original-image coordinates, not crop coordinates.
            self.assertTrue(all(not (520 <= x < 665 and 365 <= y < 600)
                                for x, y, label in item["points_xy_label"] if label == 0))

    def test_v2_point_count_and_roi_contract(self):
        box = [520, 365, 665, 600]
        self.assertTrue(recipe.validate_geometry(box, [[600, 525, 1]], 2))
        self.assertTrue(recipe.validate_geometry(box, [[600, 525, 1]] + [[700 + i, 700, 0]
                                                       for i in range(15)], 2))
        for points in ([], [[720, 620, 0]], [[600, 525, True]], [[600, 525, 1], [600, 525, 0]],
                       [[600, 525, 1], [1280, 400, 0]], [[600, 525, 1]] * 17,
                       [[600, 525, 1], [720, 620, 2]], [[600, 525, 1], [700, 700, 1]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                recipe.validate_geometry(box, points, 2)
        for badbox in ([520, 365, 520, 600], [520, 365, 1281, 600], [True, 365, 665, 600]):
            with self.subTest(box=badbox), self.assertRaises(ValueError):
                recipe.validate_geometry(badbox, [[600, 525, 1]], 2)
        with self.assertRaises(ValueError):
            recipe.validate_geometry(box, [[600, 525, 1]], 3)

    def test_tamper_and_training_inventory_rejected(self):
        with TemporaryDirectory() as directory:
            altered = Path(directory) / "recipe.json"
            for mutate in (
                lambda doc: doc["images"][0].update(source_sha256="0" * 64),
                lambda doc: doc.update(selected_training_names=["NP3_336.jpg", "NP3_999.jpg"]),
                lambda doc: doc.update(box_xyxy_original_pixels=[520, 365, 1281, 600]),
            ):
                document = copy.deepcopy(self.document)
                mutate(document)
                altered.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    recipe.load_v2(altered, FIXTURE_SHA, self.rows)
            with self.assertRaises(ValueError):
                recipe.load_v2(FIXTURE, "0" * 64, self.rows)
            with self.assertRaises(ValueError):
                recipe.load_v2(FIXTURE, FIXTURE_SHA, self.rows[:1])

    def test_v1_adapter_matches_frozen_prompt_bytes_and_row_hashes(self):
        fixture = json.loads(board5.PROMPT_PATH.read_text())
        rows = [{"path": "photos/" + item["name"], "sha256": item["source_sha256"]}
                for item in fixture["images"]]
        canonical = recipe.v1_adapter(board5.PROMPT_PATH, board5.PROMPT_SHA, rows)
        self.assertEqual(len(canonical), 5)
        for item, source in zip(canonical, fixture["images"]):
            self.assertEqual(item["box_xyxy_original_pixels"], fixture["box_xyxy_original_pixels"])
            self.assertEqual(item["points_xy_label"], source["points_xy_label"])
            self.assertEqual(item["recipe_row_sha256"],
                             recipe.row_sha256(source, fixture["box_xyxy_original_pixels"]))
        with self.assertRaises(ValueError):
            recipe.validate_geometry([520, 365, 665, 600], fixture["images"][0]["points_xy_label"], 1)
