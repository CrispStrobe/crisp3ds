# Frozen foreground-centre sparse experiment

This is one preregistered diagnostic on the same ten 768 × 512 tree PNGs used
by the clean heldout-v2 reconstruction. Its input is the manually reviewed
`tests/datasets/tree-envelope-v1.json` contract, SHA-256
`4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853`.
The polygons are approximate target-tree envelopes; gaps may contain
background. Coordinates are COLMAP pixel-centre coordinates, with polygon
boundaries included and no added half-pixel shift.

The frozen source has 53,546 original SIFT rows, including 10,664 held-out
rows; 42,882 training rows remain in its mapper database. The runner makes a
logical SQLite backup of that compact training database, validates every
keypoint row against its original six-column manifest entry, clears both
cached pair tables, then keeps only training rows whose centres lie inside
the corresponding polygon. It rematches and remaps with the source's pinned
PyCOLMAP 3.11.1 CPU options, two threads, and seed 0 before each stage. No
features are extracted again. The original held-out IDs, all 1,121 raw
held-out descriptor pairs, and full original feature coordinates are copied
unchanged into the new manifest. The old cached two-view geometries cannot
enter mapping.

Before running, the reviewed polygons select 17,884 of 42,882 existing
training rows and define a fixed object subset of 82 held-out pairs whose
two keypoint centres both fall inside their polygons. The runner records
these exact original IDs and pair indices in `roi-eligibility.json`. The
same 82 pairs are scored against both the frozen baseline model and the new
foreground model; the full 1,121-pair global population is also scored
against both. Each score reports registered, finite, and unscored counts,
plus a denominator-preserving count of errors above 4 px **or missing**.
The independent verifier should check the new model and database separately.

The approved command ran once into a fresh output directory:

```sh
.local-tools/colmap-sparse/venv/bin/python -m scripts.colmap_sparse.foreground \
  --roi tests/datasets/tree-envelope-v1.json \
  --approved-roi-sha256 4f28b90259ae84ee2475e46c17aa1300a9715c077700edc68351691ffd9dc853 \
  --output build-opencv/colmap-sparse/foreground-001
```

The guarded run has a 300-second wall limit, 1 GiB output and temporary-file
cap, 10 MiB log cap, and 10 GiB free-space reserve. The source database,
manifest, model, verification report, options provenance, original PNGs,
polygon contract, helper scripts, runner, and PyCOLMAP binary are checked
against frozen hashes or before/after hashes. `child.log` and `status.json`
remain on failure.

The guarded run succeeded in 3.53 seconds. It registered 5 of 10 cameras
and retained 192 sparse points, compared with 9 cameras and 1,773 points in
the frozen baseline. The independent verifier passed integrity, found zero
heldout/train spatial overlaps, and checked 594 positive-depth track
observations. A separate read-only check confirmed all 594 point observations
reference training keypoint centres inside the reviewed polygons.

The heldout outcomes below preserve the original denominators. Finite median
and p90 values describe only pairs whose two cameras registered. The last
column counts pairs above 4 px **or missing**, so reduced coverage cannot
improve it by dropping difficult pairs.

| Population | Model | Finite pairs | Median / p90, px | >4 px or missing |
| --- | --- | ---: | ---: | ---: |
| All 1,121 heldout pairs | Frozen baseline | 1,104 | 0.385 / 1.867 | 100 |
| All 1,121 heldout pairs | Foreground | 596 | 1.046 / 3.743 | 576 |
| Fixed 82 object pairs | Frozen baseline | 82 | 0.534 / 1.957 | 8 |
| Fixed 82 object pairs | Foreground | 61 | 0.192 / 0.552 | 23 |

As a post hoc paired check on the same 61 object pairs finite in both models,
the baseline has median/p90 0.454/1.368 px and the foreground model
0.192/0.552 px; 44 pair residuals improve and 17 worsen. Both models place
59 of these 61 pairs within 4 px. This paired subset was selected after
registration and excludes 21 object pairs missing from the foreground model.
The fixed 82-pair penalty worsens from 8 to 23, and global coverage and
residuals worsen. The experiment shows no overall improvement and is not
promoted into the production pipeline.

This is a feature-centre filter, not a pixel mask: SIFT descriptors centred
inside a polygon can still see image content outside it. The polygons are not
ground-truth tree segmentation. Sparse reconstruction coverage, epipolar
residuals, and point counts remain diagnostics, not scene accuracy claims.
