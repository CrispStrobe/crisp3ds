"""Checks for the turntable camera constraint (NumPy only)."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.turntable_mesh import synthetic_scene, turntable_rig


def poses(root):
    rows = json.loads((Path(root) / "cameras.json").read_text())["views"]
    R = np.array([r["rotation"] for r in rows])
    C = np.array([-np.array(r["rotation"]).T @ np.array(r["translation"]) for r in rows])
    return rows, R, C


class RigTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        synthetic_scene.write(self.root / "exact", views=24, size=(16, 16))

    def tearDown(self):
        self.folder.cleanup()

    def test_exact_orbit_is_unchanged(self):
        report = turntable_rig.run(self.root / "exact", self.root / "out", "uniform")
        self.assertAlmostEqual(abs(report["step_degrees"]), 15.0, places=6)
        self.assertLess(report["change_from_input"]["rotation_degrees_maximum"], 1e-4)
        self.assertLess(report["change_from_input"]["centre_fraction_of_radius_maximum"], 1e-9)

    def test_noisy_orbit_is_pulled_back(self):
        rows, R, C = poses(self.root / "exact")
        generator = np.random.default_rng(3)
        data = {"views": rows}
        for row, r, c in zip(rows, R, C):
            wobble = turntable_rig.rotation(np.array([0.0, 0.0, 1.0]), np.radians(generator.normal(0, 1.0)))[0]
            noisy_c = wobble @ c + generator.normal(0, 0.02, 3)
            noisy_r = r @ wobble.T
            row["rotation"], row["translation"] = noisy_r.tolist(), (-noisy_r @ noisy_c).tolist()
            row["image"], row["mask"] = str(self.root / "exact" / row["image"]), str(self.root / "exact" / row["mask"])
        (self.root / "noisy").mkdir()
        (self.root / "noisy/cameras.json").write_text(json.dumps(data))
        np.save(self.root / "noisy/sparse_points.npy", np.zeros((4, 3)))
        turntable_rig.run(self.root / "noisy", self.root / "out", "uniform", step_degrees=15.0)
        _, fixed_R, fixed_C = poses(self.root / "out")
        _, noisy_R, noisy_C = poses(self.root / "noisy")
        # The fitted axis is only as good as the noisy poses, so the model ring is
        # slightly tilted against the true one; it must still be far flatter and
        # more even than the input.
        self.assertLess(np.ptp(fixed_C[:, 2]), 0.25 * np.ptp(noisy_C[:, 2]))
        self.assertLess(np.ptp(np.linalg.norm(fixed_C[:, :2], axis=1)), 0.005)
        steps = np.degrees(np.diff(np.unwrap(np.arctan2(fixed_C[:, 1], fixed_C[:, 0]))))
        noisy_steps = np.degrees(np.diff(np.unwrap(np.arctan2(noisy_C[:, 1], noisy_C[:, 0]))))
        self.assertLess(np.abs(np.abs(steps) - 15.0).max(), 0.05)
        self.assertGreater(np.abs(np.abs(noisy_steps) - 15.0).max(), 0.5)
        self.assertLess(abs(np.linalg.norm(fixed_C[0]) - 6.0), 0.02)

    def test_refuses_existing_output(self):
        (self.root / "out").mkdir()
        with self.assertRaises(FileExistsError):
            turntable_rig.run(self.root / "exact", self.root / "out")


if __name__ == "__main__":
    unittest.main()
