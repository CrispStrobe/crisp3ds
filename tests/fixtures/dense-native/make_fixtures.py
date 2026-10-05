"""Reference data for the tests of crates/dense (the native dense pipeline).

Run once from the repository root with NumPy and SciPy installed:

    python tests/fixtures/dense-native/make_fixtures.py

Writes, next to this file:

  arrays-deflated.npz   numpy.savez_compressed: every element type and shape the reader supports
  arrays-stored.npz     numpy.savez of the same arrays (members stored, not deflated)
  scipy-reference.json  scipy.ndimage.distance_transform_edt and gaussian_filter on small arrays

The committed files were produced with NumPy 1.26.4 and SciPy 1.17.1.
"""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

here = Path(__file__).parent

arrays = dict(
    index=np.array([[0, 1, 2], [3, 4, 5], [-6, 7, 8], [9, 10, 2**31 - 1]], np.int32),
    total=np.array([-1.5, 0.0, 0.25, 3e-7], np.float32),
    shape=np.array([5, 6, 7], np.int64),
    voxel=np.float32(0.05),
    support_down=np.array([0.0, -0.6, 0.8], np.float64),
    support_height=np.float32(np.nan),
    grid=np.arange(24, dtype=np.float64).reshape(2, 3, 4),
    empty=np.zeros((0, 3), np.int32),
)
np.savez_compressed(here / "arrays-deflated.npz", **arrays)
np.savez(here / "arrays-stored.npz", **arrays)

rng = np.random.default_rng(225)

# Distance transform: a sparse scatter of zeros (so distances get large), one plane without
# any zero and one all-zero line.
shape = (6, 7, 5)
mask = rng.random(shape) > 0.12
mask[3] = True
mask[1, 2, :] = False
distance = ndimage.distance_transform_edt(mask)

# Gaussian filter on float32, as tsdf_hull_mesh.py calls it: the sigmas of the default
# configuration (1, 2, 4 and the final 0.6). With sigma 4 the kernel radius (16) exceeds
# every side of the array, so the reflected boundary wraps more than once.
shape_g = (9, 6, 7)
data = rng.uniform(-1, 1, shape_g).astype(np.float32)
filtered = {str(sigma): ndimage.gaussian_filter(data, sigma) for sigma in (0.6, 1.0, 2.0, 4.0)}
assert all(v.dtype == np.float32 for v in filtered.values())

reference = {
    "produced_by": "tests/fixtures/dense-native/make_fixtures.py",
    "distance_transform_edt": {
        "shape": list(shape),
        "input": mask.astype(int).ravel().tolist(),
        "output": distance.ravel().tolist(),
    },
    "gaussian_filter": {
        "shape": list(shape_g),
        "input": [float(v) for v in data.ravel()],
        "outputs": {sigma: [float(v) for v in out.ravel()] for sigma, out in filtered.items()},
    },
}
(here / "scipy-reference.json").write_text(json.dumps(reference) + "\n")
print({name: (here / name).stat().st_size for name in ("arrays-deflated.npz", "arrays-stored.npz", "scipy-reference.json")})
