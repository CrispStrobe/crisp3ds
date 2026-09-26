# COLMAP usage audit

2026-09-26. Triggered by the user's request to compare our use with upstream.
Root inspected the pinned official sources and installed bindings; a separate
Sol agent independently audited the workflow. We use PyCOLMAP **3.11.1**, not
the moving latest COLMAP release. This is a local sparse-only research lane.

## Findings

The extraction → exhaustive matching → incremental mapping sequence is correct
and follows the [official example](https://raw.githubusercontent.com/colmap/colmap/3.11.1/pycolmap/examples/example.py).
Our first configuration and validation, however, were not an adequate baseline:

| Item | Initial custom run | Correction / interpretation |
| --- | --- | --- |
| Feature records | Treated first three columns as x,y,scale | Actual six-column records are x,y plus affine shape. The split leaked orientation variants; its held-out claims are invalid. |
| Focal initialization | Explicit camera params with heuristic f=921.6 | Explicit params set `prior_focal_length=true`. Unknown focal must use the default-factor path, or genuine EXIF/calibration evidence. |
| Input | Prepared 768×512 PNGs, metadata lost | Also test existing original JPEGs with EXIF and normal internal resizing. |
| Feature budget | 3,000 plus removal of held-out groups | Standard baseline retains all features and default 8,192 budget. |
| Camera model | Shared SIMPLE_RADIAL on already-undistorted images | Pinhole is the recommended starting point for undistorted images; original JPEG baseline uses standard SIMPLE_RADIAL/AUTO. |
| Mapper | One model, minimum model size 2 | Baseline retains default multi-model settings and minimum model size. |
| Randomness | No explicit global seed | Follow upstream example with seed 0; record settings and repeat before claiming reproducibility. |

Evidence: [database format](https://raw.githubusercontent.com/colmap/colmap/3.11.1/doc/database.rst),
[image-reader implementation](https://raw.githubusercontent.com/colmap/colmap/3.11.1/src/colmap/controllers/image_reader.cc),
[camera models](https://raw.githubusercontent.com/colmap/colmap/3.11.1/doc/cameras.rst),
and installed option serialization retained with new runs. The numeric heuristic
itself is COLMAP's usual 1.2×maximum-dimension fallback; classifying it as a known
prior was the mistake. Unverified supplied scene focal estimates are not truth.

An independent checker initially also rejected a point with distinct observations
from the same image. COLMAP permits this representation: its
[export code](https://raw.githubusercontent.com/colmap/colmap/3.11.1/src/colmap/scene/reconstruction_io.cc)
filters such observations for VisualSfM. Report repeated-image tracks separately;
do not call the model corrupt solely for this reason. They are not extra
independent viewpoints.

## Frozen corrective experiments

1. Reduced images: no feature holdout, default SIFT and mapper settings, unknown
   focal initialization, shared SIMPLE_PINHOLE camera.
2. Original JPEGs: no preprocessing by our converter, AUTO camera assignment,
   default SIMPLE_RADIAL, EXIF/default-factor initialization, default 3,200-pixel
   feature-extraction limit and 8,192 feature budget, default mapper.

Both use seed 0, CPU execution, two configured extraction/matching/mapping
threads, fresh databases, 300-second timeout, 1 GiB output guard and 10 GiB
free-space floor. Settings and hashes are retained. No new photos are required.
These change several factors together and are **not single-factor ablations**.
Model registration, track consistency and training reprojection establish
workflow behavior, not independent surface accuracy. Neither run has an
independent feature holdout. No dense or mesh result follows merely from
successful sparse reconstruction. Results belong in [COLMAP-SPARSE.md](COLMAP-SPARSE.md).
