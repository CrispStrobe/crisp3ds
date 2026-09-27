import json
from pathlib import Path
import unittest

from scripts.object_motion import sam_point_full as full
from scripts.object_motion import ycb_object_masks as common


class SAMPointFullTests(unittest.TestCase):
    def test_exact_tracked_48_prompt_manifest_and_batch_contract(self):
        path = Path(__file__).parents[2] / "tests" / "datasets" / "sam21_mustard_point48_prompts.json"
        self.assertEqual(common.digest(path), full.PROMPTS_SHA)
        records = json.loads(path.read_text())["images"]
        rows = [{"path": "photos/" + item["name"], "sha256": item["source_sha256"]}
                for item in records]
        prompts = full.validated_prompts(path, rows)
        self.assertEqual(len(prompts), 48)
        self.assertEqual([len(prompts[i:i + full.BATCH_SIZE]) for i in range(0, 48, 8)], [8] * 6)
        self.assertEqual([item["name"] for item in prompts][0], "NP3_000.jpg")
        self.assertEqual([item["name"] for item in prompts][-1], "NP3_348.jpg")
        rows[5]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            full.validated_prompts(path, rows)

    def test_terminal_resource_guard_checks_exit_race_and_boundaries(self):
        floor = common.MIN_FREE_BYTES
        permitted = (full.BATCH_SECONDS, False, full.MAX_LOG, full.MAX_OUTPUT, floor)
        self.assertIsNone(full.terminal_guard(*permitted))
        cases = [
            ((full.BATCH_SECONDS + 0.01, False, 0, 0, floor), "90 seconds"),
            ((0, True, 0, 0, floor), "600 seconds"),
            ((0, False, full.MAX_LOG + 1, 0, floor), "1 MiB"),
            ((0, False, 0, full.MAX_OUTPUT + 1, floor), "60 MiB"),
            ((0, False, 0, 0, floor - 1), "10 GiB"),
        ]
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, full.terminal_guard(*arguments))


if __name__ == "__main__":
    unittest.main()
