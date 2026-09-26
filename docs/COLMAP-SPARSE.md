# Camera-estimated tree sparse research oracle

This is a local, CPU-only research comparison. It estimates camera poses and a
shared camera from the ten existing 768 × 512 tree PNGs. It does not read the
source COLMAP poses, focal lengths, point cloud, or full-size photos. Its 3D
scale is arbitrary. The input PNGs were previously resized and undistorted by
the source scene preparation; that image preprocessing is inherited. This lane
is separate from the production implementation.

The original, frozen run was launched from the repository root with:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.colmap_sparse.run
```

The command copies the resized PNG bytes into
`.local-tools/colmap-sparse/images/` under names derived from the original image
stems. The original run is preserved under `build-opencv/colmap-sparse/run-001/`.
The holdout entry point is now quarantined after the validation flaw described
below. Fresh baseline runs accept `--output /absolute/path/to/new-directory`;
the directory must not already exist.
The shared `SIMPLE_RADIAL` camera starts at focal 921.6 px, principal point
(384, 256), and radial term zero. The mapper refines focal and radial terms;
the principal point stays fixed. All poses start unknown. CPU SIFT configures
3,000 primary features per image and two worker threads; multiple orientation
rows can make the database count exceed 3,000. Matching is exhaustive.

The first holdout run removed selected SIFT rows before matching and exported
the compact mapper ID mapping, training matches, and held-out candidate pairs
in `holdout.json`. Candidate matching used mutual nearest neighbors and a
**forward-only** (image1 to image2) 0.8 Euclidean distance ratio, with no
reverse ratio or geometry filter. It cannot serve as independent validation.
The holdout code is disabled until near-coincident keypoints are grouped
robustly and the camera initialization no longer supplies a focal prior.

The frozen `run-001` completed in 8.995 seconds. It registered 3 of 10 images
and retained 12 sparse points. The fitted shared camera had focal 769.40 px
and radial term -0.0715. The manifest contains 14,208 training matches across
36 image pairs and 1,314 held-out descriptor candidate pairs. The mapper log
shows it relaxed its initial-pair constraints. This is a partial reconstruction;
the remaining seven images have no estimated pose and cannot be assessed by
its model. The database actually held six-column affine keypoints, while the
v1 split treated column three as scale. A post-run audit found 2,760 image
locations split between training and held-out orientations. **The v1 heldout
set is contaminated and does not provide independent validation.** It remains
unchanged as an audit artifact. The v1 camera had `prior_focal_length=1` because
explicit camera parameters were supplied. V1 used PyCOLMAP's implicit random
seed; future runs set the seed explicitly and record it. No provenance file was
captured during v1, so later observations must not be described as if they were.

Two conventional train-only baselines are prepared and use PyCOLMAP's default
feature limit of 8,192, default matcher and mapper settings, CPU execution,
two threads, and `set_random_seed(0)`. Neither removes features or supplies
camera parameters:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.colmap_sparse.run --baseline
.local-tools/colmap-sparse/venv/bin/python -m scripts.colmap_sparse.run --originals
```

The resized baseline uses the same ten prepared PNGs, one shared
`SIMPLE_PINHOLE` camera, and initial focal from the 1.2 default factor. The
originals baseline reads the ten original JPEGs directly with
`CameraMode.AUTO`, `SIMPLE_RADIAL`, and the default image size limit of 3,200.
It retains EXIF information; COLMAP may use EXIF focal length as a focal prior.
Their defaults and inputs differ, so comparing them is a workflow check, not a
single-factor ablation or an accuracy ranking.
Root executed both baselines in fresh supervisor directories:

| Run | Registered images | Points | Wall time | Independent model audit |
| --- | ---: | ---: | ---: | --- |
| `originals-supervisor-001` | 7/10 | 0 | 149.037 s | Consistent empty model, not usable reconstruction |
| `resized-supervisor-001` | 9/10 | 2,369 | 19.496 s | 10,310 positive-depth observations, reciprocal links and rotations pass |
| `resized-supervisor-replay-001` | 9/10 | 2,376 | 24.852 s | 10,425 positive-depth observations, reciprocal links and rotations pass |

The resized model's training reprojection median/p90 is 0.568/1.270 px.
Thirty-eight tracks have multiple observations in one image, which COLMAP
supports; 2,331 points have at most one observation per image. These are not
independent accuracy measurements. The successful reduced baseline shows that
the first custom run cannot represent COLMAP's capability on these images.
The original JPEG baseline correctly read EXIF focal 4371.2 px, but its final
binary model and text export both contain zero points. Its failure remains
unresolved; more image resolution alone is not evidence of better geometry.
No output is promoted into production. Before/after input and software hashes
matched for both completed runs. See [the upstream usage audit](COLMAP-USAGE-AUDIT.md).

The fresh reduced-image replay retained the same nine registered images but
did **not** reproduce the identical model: 2,376 points and training median/p90
0.582/1.282 px. Seed 0 and two-thread settings therefore do not establish
bitwise determinism. Both runs remain recorded; neither is selected as an
independent accuracy winner.

The original-photo log confirms that points existed during intermediate
registrations. Upstream refinement includes reprojection/parallax and
negative-depth filtering, but default logs do not identify the precise step
or criterion that removed the final points. Do not attribute this failure
solely to the dataset, camera model, EXIF, or image resolution without an
isolated experiment. Next diagnosis should reuse the database and instrument
mapper stages, rather than repeat expensive feature extraction blindly.

Artifacts are `database.db`, `model_text/{cameras,images,points3D}.txt`,
`points.ply`, and `summary.json`. `child.log` and `status.json` remain even if
the run fails. Only the quarantined run has `holdout.json`; baselines have no
independent observation holdout. The job guard caps wall time at 300 seconds, output plus temporary
files at 1 GiB, log at 10 MiB, and preserves at least 10 GiB free.
Future runs also write `provenance.json` before extraction and update it after
mapping with installed versions, effective option dictionaries, explicit
seeds, camera focal-prior flags, and before/after SHA-256 hashes for inputs,
the runner script, and the native PyCOLMAP binary. Changed hashes fail the run.

## Local dependency provenance

- PyCOLMAP 3.11.1 official PyPI macOS 11 arm64 CPython 3.11 wheel:
  SHA-256 `7c8f9679a78fa68e3b718f5efc87d520012330c32b1c0c1321766addda7903e5`,
  8,487,240 bytes. Wheel metadata declares BSD-3-Clause and only `numpy` as a
  Python dependency.
- NumPy 1.26.4 official PyPI macOS 11 arm64 CPython 3.11 wheel:
  SHA-256 `edd8b5fe47dab091176d21bb6de568acdd906d1887a4584a15a9a96a1dca06ef`,
  13,997,127 bytes. Its wheel carries its own license information.
- Both wheels are retained in `.local-tools/colmap-sparse/wheels/`, installed
  without package index access or dependencies into the isolated
  `.local-tools/colmap-sparse/venv/`. Nothing is installed globally.
- The pinned [COLMAP 3.11.1 macOS wheel build recipe](https://raw.githubusercontent.com/colmap/colmap/3.11.1/pycolmap/ci/install-colmap-macos.sh)
  has SHA-256 `17d6e0e3ed6f93d405ff748d99b0cba5955e9483c12884999e2d22ad6e59a910`.
  It disables CUDA, GUI, CGAL, and LSD. This is build-recipe evidence, not a
  complete binary dependency audit. The PyCOLMAP wheel contains no bundled
  license text for its native dependencies; upstream COLMAP explicitly notes
  that dependency licenses are separate. No redistribution or shipping approval
  is inferred from this local research run.
