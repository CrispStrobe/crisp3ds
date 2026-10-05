"""Analytic checks for the multiscale stereo (needs Torch; CPU device)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.turntable_mesh import synthetic_scene
from scripts.turntable_mesh.dense_config import DenseConfig, build

try:
    import torch  # noqa: F401
except ImportError:  # the scientific interpreter has no Torch
    torch = None


class ConfigTest(unittest.TestCase):
    def test_defaults_validate_and_round_trip(self):
        config = DenseConfig().validate()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.json"
            path.write_text(json.dumps({"configuration": config.to_json()}))
            self.assertEqual(build(path), config)

    def test_overrides_and_levels(self):
        config = build(None, ["grid=128", "sizes=64,128", "repair-masks=false", "windows=5"])
        self.assertEqual((config.grid, config.sizes, config.repair_masks), (128, (64, 128), False))
        self.assertEqual(config.level("windows", 3), 5)

    def test_rejects_unknown_and_invalid(self):
        with self.assertRaises(ValueError):
            build(None, ["nonsense=1"])
        with self.assertRaises(ValueError):
            build(None, ["windows=4"])
        with self.assertRaises(ValueError):
            build(None, ["best_of=9", "neighbours=4"])


@unittest.skipIf(torch is None, "Torch not installed in this interpreter")
class StereoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.turntable_mesh import multiscale_stereo

        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        synthetic_scene.write(root / "inputs")
        cls.config = build(None, synthetic_scene.SMALL)
        cls.report = multiscale_stereo.run(root / "inputs", root / "stereo", device="cpu", config=cls.config,
                                           log=lambda *a: None)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_depth_matches_the_sphere(self):
        exact = np.load(self.root / "inputs/exact_depths.npz")
        found = np.load(self.root / "stereo/depths.npz")
        box_pad = self.config.crop_padding
        errors, covered = [], []
        for n in range(self.report["views"]):
            truth = exact[f"view_{n:03d}"]
            ys, xs = np.nonzero(truth)
            crop = truth[ys.min() - box_pad : ys.max() + box_pad + 1, xs.min() - box_pad : xs.max() + box_pad + 1]
            depth = found[f"depth_{n:03d}"]
            # The common canvas is centred on the mask box; even sizes may add one pixel.
            oy, ox = (depth.shape[0] - crop.shape[0]) // 2, (depth.shape[1] - crop.shape[1]) // 2
            depth = depth[oy : oy + crop.shape[0], ox : ox + crop.shape[1]]
            both = (depth > 0) & (crop > 0)
            covered.append(both.sum() / (crop > 0).sum())
            errors.append(np.abs(depth[both] - crop[both]) / crop[both])
        self.assertGreater(np.median(covered), 0.6)
        self.assertLess(np.median(np.concatenate(errors)), 0.003)
        self.assertLess(np.percentile(np.concatenate(errors), 90), 0.01)

    def test_volume_and_report(self):
        volume = np.load(self.root / "stereo/volume.npz")
        self.assertEqual(len(volume["index"]), len(volume["total"]))
        self.assertGreater((volume["weight"] > 0).mean(), 0.3)
        # Hull of a unit sphere seen from a ring: at least the sphere, not much more.
        hull_volume = len(volume["index"]) * float(volume["voxel"]) ** 3
        self.assertGreater(hull_volume, 4.0)
        self.assertLess(hull_volume, 7.0)
        self.assertFalse(self.report["reference_used"])
        self.assertTrue((self.root / "stereo/masks-repaired/view_000.png").is_file())

    def test_refuses_existing_output(self):
        from scripts.turntable_mesh import multiscale_stereo

        with self.assertRaises(FileExistsError):
            multiscale_stereo.run(self.root / "inputs", self.root / "stereo", device="cpu", config=self.config)


if __name__ == "__main__":
    unittest.main()
