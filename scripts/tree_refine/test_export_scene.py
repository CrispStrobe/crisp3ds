import json
from pathlib import Path
import tempfile
import unittest

from scripts.tree_refine.export_scene import export


class ExportGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] /
                                                ".local-tools" / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        for i in range(10):
            (self.baseline / "views" / f"view_{i:04d}.mve").mkdir(parents=True)
        rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        self.views = [dict(id=i, camera=dict(width=768, height=512, fx=100,
                          fy=100, cx=383.5, cy=255.5), rotation=rotation,
                          translation=[-i, 0, 0]) for i in range(10)]
        (self.baseline / "conversion.json").write_text(json.dumps({"views": self.views}))
        self.refined = self.root / "refined.txt"
        self.summary = "SUMMARY 1 1 0 0 1 2 0 0 0 0 0 0 2 2 0"
        self.cameras = ["CAMERA " + str(i) + " " + " ".join(map(str,
                        view["rotation"] + view["translation"])) for i, view in enumerate(self.views)]
        self.point = "POINT 0 0 5 2 0 0 383.5 255.5 1 0 363.5 255.5"
        self.write_refined()

    def write_refined(self):
        self.refined.write_text("\n".join([self.summary, "LENGTHS 1 0 0 0 0 0 0 0 0",
                                            *self.cameras, self.point]) + "\n")

    def test_valid_export_and_existing_target(self):
        output = self.root / "scene"
        report = export(self.baseline, self.refined, output)
        self.assertEqual(report["tracks"], 1)
        self.assertEqual(report["fixed_anchor_max_parameter_delta"], 0)
        self.assertEqual(report["source_baseline"], report["final_baseline"])
        with self.assertRaises(FileExistsError):
            export(self.baseline, self.refined, output)

    def test_rejects_nonfinite_rms(self):
        self.summary = self.summary.replace(" 0 0 0 0 0 2 2 0", " nan 0 0 0 0 2 2 0")
        self.write_refined()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            export(self.baseline, self.refined, self.root / "scene")

    def test_rejects_anchor_change(self):
        self.cameras[0] = self.cameras[0][:-1] + "1"
        self.write_refined()
        with self.assertRaisesRegex(ValueError, "anchored poses changed"):
            export(self.baseline, self.refined, self.root / "scene")


if __name__ == "__main__":
    unittest.main()
