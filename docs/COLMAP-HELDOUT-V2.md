# Frozen clean observation holdout, resized tree images

This is one predeclared validation run using the ten corrected 768×512 tree
PNGs. The new runner is `scripts/colmap_sparse/heldout_v2.py`; its output is
`build-opencv/colmap-sparse/heldout-v2-001`. It uses the **existing full-feature**
database from `resized-supervisor-001`, copied through SQLite's backup API.
The database camera is one shared `SIMPLE_PINHOLE` camera with initial
`f=921.6, cx=384, cy=256`, `prior_focal_length=0`, and zero pose priors.
The runner checks these fields and the baseline extraction provenance. It
imports no camera poses, fitted focal length, fitted distortion, points, or
matches from the baseline reconstruction. The staged PNG hashes must equal
those used to extract the source features.

Before matching, it deletes all copied `matches` and `two_view_geometries`
rows and verifies both tables are empty. It assigns **all** original keypoint
rows per image using the fixed `colmap-spatial-components-v1-20260926` seed and
0.25 px connected components from `feature_split.py`. It checks that each
held-out row list equals the independent read-only audit at
`feature-split-audit-001.json`. All held-out keypoint and descriptor rows are
removed from the copied mapper database; the remaining rows are compacted in
original order. The JSON manifest saves every original full keypoint row, its
held-out IDs, and the compact-to-original training row map. The provenance
records rows removed, rows retained, and old pair tables cleared.

The held-out candidate recipe uses only descriptors belonging to held-out
rows. For all 45 image pairs it requires mutual nearest neighbours and a
strict Euclidean distance ratio below 0.8 **in both directions**. It does
not inspect poses, intrinsics, epipolar residuals, geometric inliers, or
future registration. All raw candidate matches remain in the manifest,
including those whose cameras do not register or whose geometry is poor.
This is an observation holdout within shared photographs and scene content,
not an independent capture or ground truth.

The training lane runs PyCOLMAP 3.11.1 (the same pinned binary as the baseline),
CPU only. Exhaustive SIFT matching and incremental mapping use PyCOLMAP
defaults except for two threads. `set_random_seed(0)` is called separately
immediately before matching and immediately before mapping. The model is
written to COLMAP text for the independent verifier. There is one run with
no post-result option tuning. The runner writes all training match references
using original IDs and the untouched raw held-out matches to `holdout.json`.
The independent verifier should inspect that file, the final database, and
the text model. The guard requires a fresh output directory, limits the run
to 300 seconds and 1 GiB of output, and reserves at least 10 GiB free space.

The development camera screen is frozen before the run: require independent
integrity checks to pass, at least 8 of 10 cameras registered, at least 70%
of all raw held-out candidate matches scoreable, and an overall scored
held-out square-root Sampson median at most 1 px and p90 at most 4 px.
Report all pair coverage, unavailable matches, three-view cycles, and sparse
positive-depth counts regardless of this screen. A screen pass is useful for
deciding whether to inspect a sparse model for later dense work; it is not
physical depth accuracy or product acceptance. The runner leaves
`quality_accepted=false` and the screen pending until independent verification.

To run after protocol review:

```sh
.local-tools/colmap-sparse/venv/bin/python -m unittest scripts.colmap_sparse.test_heldout_v2
.local-tools/colmap-sparse/venv/bin/python -m scripts.colmap_sparse.heldout_v2
```

## Frozen run outcome

Root launched the unchanged protocol once. `heldout-v2-001` completed in
9.136 seconds, reusing cached features (this time excludes feature extraction).
It registered 9/10 images and retained 1,773 points. Before rematching, it
deleted all 45 old raw-match rows and 45 old verified-geometry rows and
confirmed both tables empty. It retained 42,882 training features and excluded
10,664 held-out features. Source database, images and software hashes remained
unchanged. The old contaminated run was not modified.

Root's independent `verification-supervisor.json` confirms zero spatial split
overlap, exact agreement with the database's 14,651 training match references,
reciprocal model tracks and 7,046 positive-depth observations. It reports:

| Frozen screen check | Actual result | Threshold |
| --- | ---: | ---: |
| Registered images | 9/10 | At least 8/10 |
| Scoreable raw held-out matches | 1,104/1,121 (98.48%) | At least 70% |
| Held-out square-root Sampson median | 0.385 px | At most 1 px |
| Held-out square-root Sampson p90 | 1.867 px | At most 4 px |

The development camera screen **passes**, recorded separately in
`assessment-supervisor.json`. Of the scored matches, 1,021 are within 4 px;
83 exceed that threshold and 17 additional matches cannot be scored because
their camera did not register. All remain in the report.

The independent three-view diagnostic finds 181 closed raw match cycles:
179 have registered cameras, 73 of those have less than 1° parallax, and
106 are usable. None of the usable triangles has nonpositive depth. Third-view
reprojection median/p90 is **1.355/16.261 px**; 85/106 are within 4 px and
21 exceed it. These outliers prevent treating the epipolar screen as evidence
of reliable dense surfaces. This subset also excludes unregistered and
low-parallax cycles; its median is not a missing-inclusive accuracy score.

The model is a promising camera-estimation research candidate, not production
acceptance or ground-truth agreement. Next inspect the bad three-view cycles
and their image support, without deleting them from this frozen evaluation.
Any resulting changes must be assessed on a newly reserved capture or
predeclared validation population; this result has now informed development.
