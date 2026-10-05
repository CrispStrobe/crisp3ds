"""Analytic checks for hull-bounded surface extraction (needs SciPy, scikit-image)."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts.turntable_mesh.dense_config import build

try:
    from scripts.turntable_mesh import tsdf_hull_mesh
except ImportError:  # the Torch interpreter has no SciPy / scikit-image
    tsdf_hull_mesh = None


def sphere_volume(path, *, observed=True, n=64, voxel=0.05, truncation=3.0):
    """Hull = ball of radius 1.15; evidence = exact TSDF of a unit ball."""
    grid = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) * voxel - n * voxel / 2
    radius = np.linalg.norm(grid, axis=1)
    inside = radius < 1.15
    index = np.indices((n, n, n)).reshape(3, -1).T[inside].astype(np.int32)
    sdf = (radius[inside] - 1.0) / (truncation * voxel)
    seen = (sdf > -1) & observed
    weight = np.where(seen, 6.0, 0).astype(np.float32)
    total = (np.clip(sdf, -1, 1) * weight).astype(np.float32)
    np.savez(path, index=index, total=total, weight=weight, shape=np.array([n, n, n]),
             origin=np.full(3, -n * voxel / 2, np.float32), voxel=np.float32(voxel),
             truncation=np.float32(truncation * voxel))


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    record = np.frombuffer(data, dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")], offset=84)
    assert len(record) == count
    return record["v"].astype(float)


@unittest.skipIf(tsdf_hull_mesh is None, "SciPy / scikit-image not installed in this interpreter")
class MeshTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_observed_sphere_is_closed_and_accurate(self):
        sphere_volume(self.root / "v.npz")
        report = tsdf_hull_mesh.run(self.root / "v.npz", self.root / "mesh", build(None, ["mesh_taubin_cycles=0"]))
        self.assertTrue(report["closed"])
        self.assertEqual(report["genus"], 0)
        self.assertGreater(report["signed_volume"], 0)
        radius = np.linalg.norm(read_stl(self.root / "mesh/mesh.stl").reshape(-1, 3), axis=1)
        self.assertLess(abs(np.median(radius) - 1.0), 0.03)  # voxel is 0.05
        self.assertLess(radius.max() - radius.min(), 0.08)

    def test_unobserved_volume_falls_back_to_the_hull(self):
        sphere_volume(self.root / "v.npz", observed=False)
        report = tsdf_hull_mesh.run(self.root / "v.npz", self.root / "mesh", build(None, ["mesh_taubin_cycles=0"]))
        self.assertTrue(report["closed"])
        self.assertEqual(report["observed_hull_fraction"], 0)
        radius = np.linalg.norm(read_stl(self.root / "mesh/mesh.stl").reshape(-1, 3), axis=1)
        self.assertLess(abs(np.median(radius) - 1.15), 0.05)

    def test_refuses_existing_output(self):
        sphere_volume(self.root / "v.npz")
        (self.root / "mesh").mkdir()
        with self.assertRaises(FileExistsError):
            tsdf_hull_mesh.run(self.root / "v.npz", self.root / "mesh")


if __name__ == "__main__":
    unittest.main()
