"""Hand-calculated tests for dependency-free stereo quality diagnostics."""

from array import array
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from scripts import analyze_stereo_benchmarks as analyzer


ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".local-tools/tmp"


def pfm(path, rows, scale=-1.0):
    endian = "<" if scale < 0 else ">"
    with path.open("wb") as output:
        output.write(f"Pf\n{len(rows[0])} {len(rows)}\n{scale}\n".encode())
        for row in reversed(rows):
            output.write(struct.pack(endian + "f" * len(row), *row))


class DiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(SCRATCH).free < analyzer.RESERVE_BYTES:
            raise unittest.SkipTest("10 GiB storage reserve")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="c3ds-quality-", dir=SCRATCH)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_pfm_endian_scale_bottom_first_zero_and_infinity(self):
        rows = [[0.0, math.inf], [2.5, -1.0]]
        for scale in (-2.0, 2.0):
            path = self.root / f"{scale}.pfm"
            pfm(path, rows, scale)
            width, height, read_rows = analyzer.read_pfm(path)
            self.assertEqual((width, height), (2, 2))
            self.assertEqual(read_rows[0][0], 0.0)
            self.assertTrue(math.isinf(read_rows[0][1]))
            self.assertEqual(list(read_rows[1]), [2.5, -1.0])
        with self.assertRaisesRegex(ValueError, "payload length"):
            (self.root / "short.pfm").write_bytes(b"Pf\n2 2\n-1\n" + b"\0" * 12)
            analyzer.read_pfm(self.root / "short.pfm")
        bad = self.root / "bad.pfm"
        bad.write_bytes(b"Pf\n0 2\n-1\n")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analyzer.read_pfm(bad)
        huge = self.root / "huge.pfm"
        huge.write_bytes(b"Pf\n10000000 2\n-1\n")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analyzer.read_pfm(huge)

    def test_hand_calculated_categories_common_and_interior(self):
        # 10x10 leaves an 8x8 geometric interior. Three valid pixels:
        # border (0,0), interior (1,1), interior (2,2).
        truth = [[math.inf] * 10 for _ in range(10)]
        truth[0][0], truth[1][1], truth[2][2] = 0.0, 4.0, 8.0
        a = [[math.inf] * 10 for _ in range(10)]
        b = [[math.inf] * 10 for _ in range(10)]
        a[0][0], a[1][1], a[2][2] = 0.0, math.inf, 11.0
        b[0][0], b[1][1], b[2][2] = 3.0, 4.0, 10.0
        image = lambda rows: (10, 10, [array("f", row) for row in rows])
        result, maps = analyzer.analyze_scene(image(truth), {"a": image(a), "b": image(b)})
        self.assertEqual(result["interior_bounds_xyxy"], [1, 1, 9, 9])
        full_a = result["regions"]["full"]["a"]
        self.assertEqual(full_a["categories"], {"truth_invalid": 97, "missing": 1,
                                                  "correct": 1, "incorrect": 1})
        self.assertEqual(full_a["bad2_all_valid"], 2 / 3)
        self.assertEqual(full_a["mae_matched_px"], 1.5)
        self.assertEqual(result["regions"]["interior"]["a"]["categories"]["missing"], 1)
        self.assertEqual(result["regions"]["interior"]["a"]["categories"]["correct"], 0)
        self.assertEqual(result["regions"]["common_full"]["a"]["truth_valid"], 2)
        self.assertEqual(result["regions"]["common_full"]["b"]["truth_valid"], 2)
        self.assertEqual(result["regions"]["common_interior"]["a"]["truth_valid"], 1)
        self.assertEqual(result["regions"]["common_full"]["b"]["mae_matched_px"], 2.5)
        self.assertEqual(maps["a"][:3], bytes(analyzer.COLORS["correct"]))
        self.assertEqual(maps["a"][33:36], bytes(analyzer.COLORS["missing"]))
        self.assertEqual(maps["a"][66:69], bytes(analyzer.COLORS["incorrect"]))
        self.assertEqual(maps["a"][3:6], bytes(analyzer.COLORS["truth_invalid"]))
        with_search, _ = analyzer.analyze_scene(image(truth), {"a": image(a), "b": image(b)},
                                                search_count=3)
        self.assertEqual(with_search["interior_bounds_xyxy"], [3, 1, 9, 9])
        self.assertEqual(with_search["regions"]["left_search_strip"]["a"]["categories"]["missing"], 1)
        self.assertEqual(with_search["regions"]["interior"]["a"]["truth_valid"], 0)
        with self.assertRaisesRegex(ValueError, "mask dimensions"):
            analyzer.analyze_scene(image(truth), {"a": image(a)}, mask=b"\xff")

    def test_no_valid_truth_or_predictions_has_null_rates(self):
        image = (10, 10, [array("f", [math.inf] * 10) for _ in range(10)])
        result, _ = analyzer.analyze_scene(image, {"a": image})
        metrics = result["regions"]["full"]["a"]
        self.assertEqual(metrics["truth_valid"], 0)
        self.assertIsNone(metrics["coverage"])
        self.assertIsNone(metrics["mae_matched_px"])

    def test_dimensions_rejected_and_existing_output_preserved(self):
        image = (10, 10, [array("f", [0.0] * 10) for _ in range(10)])
        smaller = (9, 10, [array("f", [0.0] * 9) for _ in range(10)])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analyzer.analyze_scene(image, {"a": smaller})
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "keep"
        sentinel.write_text("untouched")
        report = self.root / "benchmark.json"
        report.write_text(json.dumps({"ok": True, "scenes": {"dummy": {}}}))
        with self.assertRaises(FileExistsError):
            analyzer.analyze_benchmark(report, existing)
        self.assertEqual(sentinel.read_text(), "untouched")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing-target")
        with self.assertRaises(FileExistsError):
            analyzer.analyze_benchmark(report, dangling)

    def test_manifest_mutation_and_unsafe_name_rejected_before_output(self):
        prepared = self.root / "prepared"
        prepared.mkdir()
        manifest = prepared / "inputs.json"
        manifest.write_text('{"truth":"truth.pfm"}')
        scene = {"prepared": str(prepared), "inputs_manifest_sha256": "wrong",
                 "inputs_sha256": {"truth": "wrong"}, "engines": {}}
        report = self.root / "benchmark.json"
        output = self.root / "new-output"
        report.write_text(json.dumps({"ok": True, "scenes": {"../escape": scene}}))
        with self.assertRaisesRegex(ValueError, "unsafe scene name"):
            analyzer.analyze_benchmark(report, output)
        self.assertFalse(output.exists())
        report.write_text(json.dumps({"ok": True, "scenes": {"safe": scene}}))
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            analyzer.analyze_benchmark(report, output)
        self.assertFalse(output.exists())
        scene["inputs_manifest_sha256"] = analyzer.sha256(manifest)
        pfm(prepared / "truth.pfm", [[0.0] * 10 for _ in range(10)])
        report.write_text(json.dumps({"ok": True, "scenes": {"safe": scene}}))
        with self.assertRaisesRegex(ValueError, "truth hash"):
            analyzer.analyze_benchmark(report, output)
        self.assertFalse(output.exists())
        scene["inputs_sha256"]["truth"] = analyzer.sha256(prepared / "truth.pfm")
        scene["engines"] = {"../escape": {"status": "ok"}}
        report.write_text(json.dumps({"ok": True, "scenes": {"safe": scene}}))
        with self.assertRaisesRegex(ValueError, "unsafe engine name"):
            analyzer.analyze_benchmark(report, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
